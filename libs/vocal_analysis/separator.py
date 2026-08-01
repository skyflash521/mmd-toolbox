"""S1 ボーカル抽出。

S0 の出力(入力レベル正規化済み。ステレオ・元サンプルレート)からボーカルを分離し、ボーカルWAVを得る。
どのアダプタで分離するかは `separator`(登録アダプタの安定 id)で選ぶ。現行の登録は audio-separator
経由で Demucs v4 htdemucs_ft を in-process 実行するもの1つ。mode(always/never)はこの抽象が解釈する。
"""

import atexit
import inspect
import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import soundfile as sf

from .config import DEFAULT_SEPARATOR, SEPARATOR_CONFIG, SeparatorId
from .quiet import silence_third_party_output
from .types import AudioPcm


class SeparationError(Exception):
    """S1 の分離失敗(audio-separator 未導入など、原因が分かるエラー)。"""


def separate(
    pcm: AudioPcm, mode: Literal["always", "never"], *,
    separator: SeparatorId = DEFAULT_SEPARATOR,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """S0出力からボーカルWAVのパスを得る。

    separator は分離に使う登録アダプタの安定 id で、登録の無い id は `SeparationError`
    (既定のアダプタへ黙って倒さない)。mode="never" でも id は検証する(実際に分離しない指定でも、
    渡された id が無効であることに変わりはないため)。
    戻り値のWAVを格納する作業ディレクトリは呼び出し元に公開せず、プロセスの正常終了時に
    削除を試みる(強制終了時や削除失敗時は残置を許容する)。呼び出し元が削除
    タイミングを制御する手段は無い。on_progress はモデルの初回取得が実際にネットワーク
    ダウンロードを要した区間(vocal_analysis.recognizer.recognize の同名引数と同じ契約)に加え、
    モデルのロード開始時(ダウンロードの有無に関わらず、キャッシュ済みでディスクから読み込むだけの
    場合を含む。進捗文言`f"モデル読み込み中: {ファイル名}"`)、および分離処理(Demucs推論)が
    実際に進行している区間(進捗文言`f"分離中: {percent}%"`)にも、都度渡して呼ぶ。
    """
    if mode not in ("always", "never"):
        raise ValueError(f"未知の mode です: {mode!r}(always/never のいずれかを指定してください)")
    impl = _SEPARATOR_IMPLS.get(separator)
    if impl is None:
        raise SeparationError(f"未知の separator です: {separator!r}")

    work_dir = Path(tempfile.mkdtemp(prefix="vocal_analysis_s1_"))
    atexit.register(shutil.rmtree, work_dir, ignore_errors=True)
    input_wav = work_dir / "input.wav"
    sf.write(input_wav, pcm.samples, pcm.sample_rate)

    if mode == "never":
        return input_wav
    return impl(work_dir, input_wav, on_progress)


def _separate_audio_separator_htdemucs_ft(work_dir: Path, input_wav: Path, on_progress) -> Path:
    """audio-separator 経由の Demucs v4 htdemucs_ft でボーカルを分離する(安定 id の実装本体)。"""
    try:
        separator_obj = _build_separator(work_dir)
    except ImportError as e:
        raise SeparationError(
            "audio-separator が見つかりません。導入してください(pip install audio-separator onnxruntime)。"
        ) from e
    downloaded = _load_model_with_progress(separator_obj, on_progress)
    if downloaded:
        # ロード完了時点で通知を終える(ダウンロードが実際に発生した場合のみ)。以降の実際の
        # 分離処理(separator.separate)はダウンロードと無関係なので、ここで先にクリアする
        # (分離処理の完了まで「ダウンロード中」の補足を残すと、分離が進んでいるだけなのに
        # まだダウンロード中であるかのように誤認させる)。
        on_progress("")
    output_files = _separate_with_progress(separator_obj, input_wav, on_progress)
    output_path = Path(output_files[0])
    # S1のGPUメモリをS2(内容認識・音素モデル)のロード前に返す。保持したままだと、後続の
    # 内容認識パイプラインのロード・推論時にVRAMが逼迫し、処理時間が大きく悪化する。
    del separator_obj
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    # audio-separator は output_dir 相対のファイル名だけを返すことがある(絶対パスの
    # 保証はない)。相対パスの場合は分離器の output_dir(work_dir)を基準に解決する。
    return output_path if output_path.is_absolute() else work_dir / output_path


# 安定 id → 分離の実装。アダプタを増やすときは SeparatorId へ id を足すのと同時にここへ実装を
# 登録する。登録の無い id は separate() が弾くので、id だけが増えて分離器が既定のまま動く
# (選んだつもりで選べていない)状態にはならない。
_SEPARATOR_IMPLS = {
    "audio-separator-htdemucs-ft": _separate_audio_separator_htdemucs_ft,
}


def _load_model_with_progress(separator_obj, on_progress) -> bool:
    """separator_obj.load_model() を実行する。実行直前に on_progress(f"モデル読み込み中: {ファイル名}")
    を1回呼び(キャッシュ済みでディスクからの読み込み・GPU転送だけでも数秒かかり、それ自体は
    ダウンロードでないため下記の中継では捕捉できない空白区間を埋める)、実際にモデルダウンロードが
    発生した区間はさらに on_progress(f"ダウンロード中: {ファイル名} {percent}%") を呼ぶ(この間は
    「モデル読み込み中」の文言を上書きする)。戻り値は実際にダウンロードが発生したか(呼び出し元が
    ロード完了後の空文字列クリア通知を出すべきか)。

    audio-separator は huggingface_hub の snapshot_download と異なり tqdm_class 差し込み口を
    持たず、audio_separator.separator.separator モジュール内で `from tqdm import tqdm` した
    名前を直接インスタンス化してダウンロード進捗バーを作る(`download_file_if_not_exists`)。
    その名前だけをロード中だけ中継クラスへ差し替える。tqdm は quiet.silence_third_party_output()
    により既定 disable=True のため、self.n は更新されない(disable 時は内部状態を進めない
    実装のため)。中継は自前カウンタ(self._relay_n)で行う。

    モデル1件は複数ファイル(htdemucs_ft なら重み4分割+YAML)で構成され、`download_file_if_not_exists`
    はファイルごとに新しい tqdm インスタンスを作る(バイト集約された単一の合計進捗は無い)ため、
    0%→100% のサイクルがファイル数ぶん繰り返される。これを「壊れている」ように見せないため、
    通知に実際のファイル名(`download_file_if_not_exists` の `output_path` から取る)を含める。
    `Separator.download_file_if_not_exists` も一時的に差し替え、呼び出し直後の `output_path` を
    共有状態へ記録してから本来の処理へ委譲する(tqdm インスタンス自身は呼び出し元のファイル名を
    知らないため)。

    差し込み先はいずれも audio-separator の公開契約ではないため、上流の変更で失われうる。取得を
    まとめて試み、1つでも欠けていれば差し替えを行わずロードだけを実行する(進捗表示の可否を
    ロードの成否に連動させない)。ロード開始の通知自体は差し替えに依存しないのでこの場合も行う。
    この通知の後始末は通常経路と同じで、呼び出し元が次に出す文言で上書きされるか消される。
    """
    if on_progress is None:
        separator_obj.load_model(model_filename=SEPARATOR_CONFIG.model_filename)
        return False

    try:
        import audio_separator.separator.separator as _as_separator_module
        from tqdm import tqdm as _base_tqdm

        original_download = _as_separator_module.Separator.download_file_if_not_exists
        original_tqdm = _as_separator_module.tqdm
    except (ImportError, AttributeError):
        on_progress(f"モデル読み込み中: {SEPARATOR_CONFIG.model_filename}")
        separator_obj.load_model(model_filename=SEPARATOR_CONFIG.model_filename)
        return False

    state = {"shown": False, "current_filename": "モデルファイル"}

    class _RelayTqdm(_base_tqdm):
        def __init__(self, *args, **kwargs):
            self._relay_n = kwargs.get("initial") or 0
            self._relay_filename = state["current_filename"]
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            result = super().update(n)
            if self.total:
                self._relay_n += n or 0
                state["shown"] = True
                percent = min(100, int(self._relay_n * 100 / self.total))
                on_progress(f"ダウンロード中: {self._relay_filename} {percent}%")
            return result

    def _patched_download(self, *args, **kwargs):
        # 引数はそのまま素通しし、ファイル名の取り出しだけを行う。取り出しは位置・キーワードの
        # どちらの渡し方にも当たり、当たらなければ直前のラベルを保つ(呼び出し形が変わっても
        # ダウンロードを巻き込まない。ラベルが古いままになるだけで済ませる)。
        output_path = kwargs.get("output_path") or (args[1] if len(args) > 1 else None)
        if output_path is not None:
            state["current_filename"] = Path(output_path).name
        return original_download(self, *args, **kwargs)

    _as_separator_module.tqdm = _RelayTqdm
    _as_separator_module.Separator.download_file_if_not_exists = _patched_download
    try:
        on_progress(f"モデル読み込み中: {SEPARATOR_CONFIG.model_filename}")
        separator_obj.load_model(model_filename=SEPARATOR_CONFIG.model_filename)
    finally:
        _as_separator_module.tqdm = original_tqdm
        _as_separator_module.Separator.download_file_if_not_exists = original_download
    return state["shown"]


def _separate_with_progress(separator_obj, input_wav: Path, on_progress):
    """separator_obj.separate() を実行し、Demucs推論の実際の進捗を on_progress(f"分離中: {percent}%")
    で中継する(モデルダウンロードとは別区間。ダウンロード進捗は _load_model_with_progress)。

    audio-separator の DemucsSeparator.demix_demucs は vendored apply_model(...,
    set_progress_bar=None, ...) を固定引数で呼ぶため、コールバックの差し込み口が無い。
    demix_demucs が import した名前 `apply_model`(audio_separator.separator.architectures.
    demucs_separator モジュール内)だけを分離中だけ差し替え、set_progress_bar を注入して
    委譲する。apply_model はhtdemucs_ftのアンサンブル構成員・区間ごとに自身の名前空間の
    apply_model を再帰呼び出しするため、この差し替えは再帰呼び出しの名前解決には効かないが、
    差し込んだ set_progress_bar は呼び出しごとに構築される **kwargs を通じて再帰全体へそのまま
    伝播するため、最上位の1呼び出しを差し替えるだけで全区間・全構成員ぶんの進捗が中継される。
    set_progress_bar(step, fraction) の fraction は0から0.8まで単調増加する仕様(0.8-1.0は
    後処理向けの予約領域で進捗コールバックの対象外)なので、0.8を100%とみなして正規化する。

    差し替え先も注入するキーワードも audio-separator の公開契約ではないため、上流の変更で失われうる。
    差し替える前に、対象が取得でき、かつ set_progress_bar を受け取れることを確かめる(注入だけ先に
    行うと、受け取れなくなった構成では数分かかる分離処理の途中で落ちる)。確かめられなければ
    差し替えを行わず、進捗文言を省いたまま分離を実行する(進捗表示の可否を分離の成否に連動させない)。
    この場合、直前の「モデル読み込み中」を残すと分離が進む間ずっとロード中に見えるため、分離へ入る
    前に消す。
    """
    if on_progress is None:
        return separator_obj.separate(str(input_wav))

    try:
        import audio_separator.separator.architectures.demucs_separator as _demucs_module

        original_apply_model = _demucs_module.apply_model
        injectable = "set_progress_bar" in inspect.signature(original_apply_model).parameters
    except (ImportError, AttributeError, TypeError, ValueError):
        injectable = False
    if not injectable:
        on_progress("")
        return separator_obj.separate(str(input_wav))

    def _relay_progress(*args, **kwargs):
        # 呼び出し規約(引数の数・位置かキーワードか)が変わっても分離処理を巻き込まないよう、
        # 束縛が失敗しない形で受ける。期待する形で受け取れなければ、その回の中継を諦める。
        if len(args) < 2:
            return
        percent = min(100, round(args[1] / 0.8 * 100))
        on_progress(f"分離中: {percent}%")

    def _patched_apply_model(*args, **kwargs):
        kwargs["set_progress_bar"] = _relay_progress
        return original_apply_model(*args, **kwargs)

    _demucs_module.apply_model = _patched_apply_model
    try:
        return separator_obj.separate(str(input_wav))
    finally:
        _demucs_module.apply_model = original_apply_model


def _build_separator(output_dir: Path):
    """audio-separator の Separator を固定条件で構成する。

    log_level=CRITICAL でログ(モデルロード・処理進行等の逐次通知)を抑える。呼び出し側 CLI の
    進捗表示(改行なしで同じ行を上書きするライブ行)と同じ標準エラーへ抑制されないまま割り込み、
    行が連結して読めなくなるため。silence_third_party_output() は tqdm 由来の進捗バー(Demucs
    内部の推論進捗)を抑える(この Separator の log_level とは別経路のため個別に要る)。
    """
    from audio_separator.separator import Separator

    silence_third_party_output()
    return Separator(
        log_level=logging.CRITICAL,
        output_dir=str(output_dir),
        output_single_stem=SEPARATOR_CONFIG.output_single_stem,
        demucs_params={
            "segment_size": "Default",
            "shifts": SEPARATOR_CONFIG.shifts,
            "overlap": 0.25,
            "segments_enabled": True,
        },
    )
