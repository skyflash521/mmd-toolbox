"""S1 ボーカル抽出。

S0 の出力(入力レベル正規化済み。ステレオ・元サンプルレート)からボーカルを分離し、ボーカルWAVを得る。
分離は audio-separator 経由で Demucs v4 htdemucs_ft を in-process 実行する。mode(always/never)は
この抽象が解釈する。
"""

import atexit
import logging
import shutil
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import soundfile as sf

from .config import SEPARATOR_CONFIG
from .quiet import silence_third_party_output
from .types import AudioPcm


class SeparationError(Exception):
    """S1 の分離失敗(audio-separator 未導入など、原因が分かるエラー)。"""


def separate(
    pcm: AudioPcm, mode: Literal["always", "never"], *,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    """S0出力からボーカルWAVのパスを得る。

    戻り値のWAVを格納する作業ディレクトリは呼び出し元に公開せず、プロセスの正常終了時に
    削除を試みる(強制終了時や削除失敗時は残置を許容する)。呼び出し元が削除
    タイミングを制御する手段は無い。on_progress はモデルの初回取得が実際にネットワーク
    ダウンロードを要した区間だけ、進捗文言を渡して呼ぶ(vocal_analysis.recognizer.recognize の
    同名引数と同じ契約)。
    """
    if mode not in ("always", "never"):
        raise ValueError(f"未知の mode です: {mode!r}(always/never のいずれかを指定してください)")

    work_dir = Path(tempfile.mkdtemp(prefix="vocal_analysis_s1_"))
    atexit.register(shutil.rmtree, work_dir, ignore_errors=True)
    input_wav = work_dir / "input.wav"
    sf.write(input_wav, pcm.samples, pcm.sample_rate)

    if mode == "never":
        return input_wav

    try:
        separator = _build_separator(work_dir)
    except ImportError as e:
        raise SeparationError(
            "audio-separator が見つかりません。導入してください(pip install audio-separator onnxruntime)。"
        ) from e
    downloaded = _load_model_with_progress(separator, on_progress)
    if downloaded:
        # ロード完了時点で通知を終える(ダウンロードが実際に発生した場合のみ)。以降の実際の
        # 分離処理(separator.separate)はダウンロードと無関係なので、ここで先にクリアする
        # (分離処理の完了まで「ダウンロード中」の補足を残すと、分離が進んでいるだけなのに
        # まだダウンロード中であるかのように誤認させる)。
        on_progress("")
    output_files = separator.separate(str(input_wav))
    output_path = Path(output_files[0])
    # S1のGPUメモリをS2(内容認識・音素モデル)のロード前に返す。保持したままだと、後続の
    # 内容認識パイプラインのロード・推論時にVRAMが逼迫し、処理時間が大きく悪化する。
    del separator
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    # audio-separator は output_dir 相対のファイル名だけを返すことがある(絶対パスの
    # 保証はない)。相対パスの場合は分離器の output_dir(work_dir)を基準に解決する。
    return output_path if output_path.is_absolute() else work_dir / output_path


def _load_model_with_progress(separator_obj, on_progress) -> bool:
    """separator_obj.load_model() を実行し、実際にモデルダウンロードが発生した区間だけ
    on_progress(f"ダウンロード中: {ファイル名} {percent}%") を呼ぶ。戻り値は実際にダウンロードが
    発生したか(呼び出し元がロード完了後の空文字列クリア通知を出すべきか)。

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
    """
    if on_progress is None:
        separator_obj.load_model(model_filename=SEPARATOR_CONFIG.model_filename)
        return False

    import audio_separator.separator.separator as _as_separator_module
    from tqdm import tqdm as _base_tqdm

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

    original_download = _as_separator_module.Separator.download_file_if_not_exists

    def _patched_download(self, url, output_path, *args, **kwargs):
        state["current_filename"] = Path(output_path).name
        return original_download(self, url, output_path, *args, **kwargs)

    original_tqdm = _as_separator_module.tqdm
    _as_separator_module.tqdm = _RelayTqdm
    _as_separator_module.Separator.download_file_if_not_exists = _patched_download
    try:
        separator_obj.load_model(model_filename=SEPARATOR_CONFIG.model_filename)
    finally:
        _as_separator_module.tqdm = original_tqdm
        _as_separator_module.Separator.download_file_if_not_exists = original_download
    return state["shown"]


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
