"""SOFA(Singing-Oriented Forced Aligner)サブプロセス呼び出しコア。

利用者提供の専用venv・SOFAリポジトリ・チェックポイントをサブプロセスとして呼び、既知の音素記号列を
音声へ時刻合わせする。SOFA本体のコード・チェックポイントは本モジュールに一切同梱しない。ここでは
入力の書き出し・サブプロセス起動・終了コード/出力検証・タイムアウト時のプロセスツリーkill・HTK出力
(100ナノ秒単位)の秒への変換・Segment契約の検証・非ASCIIパスの拒否・単語単位分割(有効な単語列の
確定・gapの確定)・IPA写像(SOFA出力記号のSegment化)を扱う。
"""

import os
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Literal

import numpy as np
import soundfile as sf

from .config import SofaAlignerConfig
from .phonemes import _BLANK_G2P_SYMBOLS, _MIN_WORD_DURATION_SEC, RecognitionError, _classify_symbol, _G2P_TO_VOCAB_SYMBOL
from .types import Segment


# SOFAの語彙(チェックポイント学習時の音素セット)は無声化母音の専用記号を持たず、通常の母音記号
# (i/u)のみを認識する。pyopenjtalk-plus由来のG2P記号列に含まれる無声化マーカーI/Uは、SOFAへ渡す
# 直前にこの写像で正規化する(G2P記号→音素モデル語彙の写像表がI→i・U→ɯのIPA記号へ
# 既に集約しているのと同じ対応関係で、Segmentの最終的なphoneme値には影響しない)。
_SOFA_VOCAB_SYMBOL_NORMALIZE: dict[str, str] = {"I": "i", "U": "u"}

# SOFA サブプロセスの完了待ちを区切る周期(秒)。長時間 1 回の communicate(timeout=長時間) で
# 待つと、待機中に届いた中断(KeyboardInterrupt)が待機完了まで反映されない(OS 待機プリミティブは
# 完了かタイムアウトまで戻らない)。短周期に分割し待ち直すことで、中断の反映を最大この秒数まで縮める。
_COMMUNICATE_POLL_SEC = 1.0


def _normalize_for_sofa_vocab(symbol: str) -> str:
    return _SOFA_VOCAB_SYMBOL_NORMALIZE.get(symbol, symbol)


def _write_sofa_inputs(
    work_dir: Path, targets: list[tuple[np.ndarray, int, list[str]]]
) -> list[str]:
    """targets(音声サンプル・サンプルレート・G2P音素記号列の組)をASCII固定名で書き出す。

    basenameは0始まりの4桁ゼロ埋め連番(segment_0000, segment_0001, ...)。音素記号列は
    _normalize_for_sofa_vocab で無声化母音マーカーを正規化してから空白区切りで.labへ書き出し、
    SOFA自身のテキスト→音素変換を経由しない。書き出した順のbasename列を返す。
    """
    basenames = []
    for i, (samples, sample_rate, phoneme_symbols) in enumerate(targets):
        basename = f"segment_{i:04d}"
        basenames.append(basename)
        sf.write(work_dir / f"{basename}.wav", samples, sample_rate)
        normalized = " ".join(_normalize_for_sofa_vocab(p) for p in phoneme_symbols)
        (work_dir / f"{basename}.lab").write_text(normalized, encoding="utf-8")
    return basenames


def _build_sofa_command(config: SofaAlignerConfig, work_dir: Path) -> list[str]:
    return [
        str(config.sofa_python),
        "infer.py",
        "--ckpt",
        str(config.checkpoint_path),
        "--folder",
        str(work_dir),
        "--mode",
        "force",
        "--g2p",
        "Phoneme",
        "--out_formats",
        "htk",
    ]


def _popen_kwargs() -> dict:
    """子孫プロセスの受け皿(タイムアウト時にプロセスツリーごと終了するための土台)。"""
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


def _terminate_and_reap(proc: subprocess.Popen) -> None:
    """proc とその子孫を終了し、標準出力/標準エラーを回収する。呼び出し元が伝播させたい中断・
    エラーを上書きしないベストエフォート処理で、内部の失敗を外へ伝播させない(無期限待機も避ける
    ため回収は _COMMUNICATE_POLL_SEC 以内に区切る)。"""
    try:
        _kill_process_tree(proc.pid)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass
    try:
        proc.communicate(timeout=_COMMUNICATE_POLL_SEC)
    except Exception:
        pass


def _kill_process_tree(pid: int) -> None:
    if sys.platform == "win32":
        subprocess.run(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    else:
        os.killpg(os.getpgid(pid), signal.SIGKILL)


def _check_ascii_paths(work_dir: Path, config: SofaAlignerConfig) -> None:
    """SOFA自身が非ASCIIパスを扱えない技術的制約により、一時ディレクトリ・sofa_root・
    checkpoint_pathのいずれかに非ASCII文字が含まれる場合はSOFAを起動せず`RecognitionError`にする。
    `sofa_python`は検証対象に含まない。
    """
    for label, path in (
        ("一時ディレクトリ", work_dir),
        ("sofa_root", config.sofa_root),
        ("checkpoint_path", config.checkpoint_path),
    ):
        if not str(path).isascii():
            raise RecognitionError(f"{label}のパスに非ASCII文字が含まれています: {path}")


def _check_paths_exist(config: SofaAlignerConfig) -> None:
    """SOFA実行環境の各パスの実在と種別をSOFA起動前に検証する。

    不正な場合は、どのパスが・なぜ不正か(存在しない/期待する種別でない)を明示した
    `RecognitionError` にする(音声前段の外部依存の実行失敗として利用先が分類・提示できる
    形にするため。未検証のまま subprocess を起動すると、パス情報を持たない FileNotFoundError が
    想定外エラー扱いで漏れる)。
    """
    for label, path, check, kind in (
        ("sofa_python", config.sofa_python, Path.is_file, "ファイル"),
        ("sofa_root", config.sofa_root, Path.is_dir, "ディレクトリ"),
        ("sofa_root配下のinfer.py", config.sofa_root / "infer.py", Path.is_file, "ファイル"),
        ("checkpoint_path", config.checkpoint_path, Path.is_file, "ファイル"),
    ):
        if not check(path):
            reason = "存在しません" if not path.exists() else f"{kind}ではありません"
            raise RecognitionError(f"SOFAの実行環境パスが不正です: {label}={path}({reason})")


def _run_sofa_subprocess(basenames: list[str], config: SofaAlignerConfig, work_dir: Path) -> None:
    """infer.pyを1回呼ぶ。タイムアウト・異常終了・出力欠落/空はいずれも RecognitionError にする。"""
    cmd = _build_sofa_command(config, work_dir)
    # SOFA は独自のプロセスグループ(_popen_kwargs)で起動するため、呼び出し元プロセスへの中断は
    # 子プロセスへ伝わらない。Popen 成功直後(proc 取得直後)から中断(KeyboardInterrupt)捕捉の
    # 保護下に入るよう、起動から待機までを1つの try で畳む。proc は起動失敗時に None のままなので、
    # 中断時の後始末は取得できている場合だけ行う。
    proc = None
    try:
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(config.sofa_root),
                encoding="utf-8",
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                **_popen_kwargs(),
            )
        except FileNotFoundError as e:
            # 事前検証(_check_paths_exist)後にパスが消える競合等の受け皿。起動失敗も外部依存の
            # 実行失敗として扱い、候補パスを明示する。
            raise RecognitionError(
                "SOFAサブプロセスを起動できませんでした(実行ファイルまたは作業ディレクトリが"
                f"見つかりません): sofa_python={config.sofa_python}, sofa_root={config.sofa_root}"
            ) from e

        # communicate() を短周期ポーリングへ分割する(_COMMUNICATE_POLL_SEC 参照)。再呼び出しは
        # subprocess.communicate() の仕様上安全(TimeoutExpired 後の再呼び出しは公式に想定された経路)。
        deadline = time.monotonic() + config.timeout_sec
        stderr = None
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _terminate_and_reap(proc)
                raise RecognitionError(
                    f"SOFA呼び出しが{config.timeout_sec}秒以内に完了しませんでした(タイムアウト)"
                )
            try:
                _, stderr = proc.communicate(timeout=min(_COMMUNICATE_POLL_SEC, remaining))
                break
            except subprocess.TimeoutExpired:
                continue
    except KeyboardInterrupt:
        # 中断(KeyboardInterrupt)を待機中に受けた場合も、タイムアウト時と同様にプロセスツリーを
        # 終了・回収してから中断を再送出する(子プロセスを取り残さない)。後始末自体は失敗しても
        # 元の中断を上書きしないベストエフォート(_terminate_and_reap)。
        if proc is not None:
            _terminate_and_reap(proc)
        raise

    if proc.returncode != 0:
        raise RecognitionError(f"SOFA呼び出しが終了コード{proc.returncode}で失敗しました: {stderr}")

    for basename in basenames:
        lab_path = work_dir / "htk" / "phones" / f"{basename}.lab"
        if not lab_path.exists() or lab_path.stat().st_size == 0:
            raise RecognitionError(f"SOFAの出力ファイルが見つからないか空です: {lab_path}")


def _parse_htk_label_file(path: Path) -> list[tuple[float, float, str]]:
    """HTKラベル形式(100ナノ秒単位の整数)を秒単位の(start_sec, end_sec, label)列へ変換する。"""
    segments = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        start_100ns, end_100ns, label = line.split(maxsplit=2)
        segments.append((int(start_100ns) / 1e7, int(end_100ns) / 1e7, label))
    return segments


_SEGMENT_CONTRACT_TOLERANCE_SEC = 0.05  # 一次検証の許容誤差(SOFA内部リサンプリング由来の丸め誤差を含む)


def _check_segment_contract(
    segments: list[tuple[float, float, str]], trim_duration_sec: float, tolerance: float
) -> None:
    """Segment契約(隙間なく連続・非重複で全時間軸を被覆)の(a)〜(d)を検証する。

    (b)(c)(d)の一致判定は`tolerance`以内の絶対差で行う(呼び出し元が一次検証は許容誤差付き、
    再検証は0.0=厳密一致で使い分ける)。
    """
    if not segments:
        raise RecognitionError("SOFA出力のSegment列が空です(全時間軸を被覆できません)")

    for start, end, _ in segments:
        if start > end:
            raise RecognitionError(f"Segmentの開始時刻が終了時刻より後です: start={start}, end={end}")

    for (_, prev_end, _), (next_start, _, _) in zip(segments, segments[1:]):
        if abs(prev_end - next_start) > tolerance:
            raise RecognitionError(
                f"隣接するSegmentの境界が一致しません: {prev_end} != {next_start}"
            )

    first_start = segments[0][0]
    if abs(first_start - 0.0) > tolerance:
        raise RecognitionError(f"先頭Segmentの開始時刻が0.0と一致しません: {first_start}")

    last_end = segments[-1][1]
    if abs(last_end - trim_duration_sec) > tolerance:
        raise RecognitionError(
            f"末尾Segmentの終了時刻がトリム後区間の全長と一致しません: {last_end} != {trim_duration_sec}"
        )


def _normalize_segments(
    segments: list[tuple[float, float, str]], trim_duration_sec: float
) -> list[tuple[float, float, str]]:
    """許容誤差を残さない厳密値へ正規化する(先頭の開始時刻を0.0へ、末尾の終了時刻をトリム後区間の
    全長へ、各隣接ペアの後続の開始時刻を先行の終了時刻へそれぞれ上書きする)。
    """
    normalized = list(segments)

    start0, end0, label0 = normalized[0]
    normalized[0] = (0.0, end0, label0)

    for i in range(len(normalized) - 1):
        _, prev_end, _ = normalized[i]
        _, next_end, next_label = normalized[i + 1]
        normalized[i + 1] = (prev_end, next_end, next_label)

    last_start, _, last_label = normalized[-1]
    normalized[-1] = (last_start, trim_duration_sec, last_label)

    return normalized


def _validate_and_normalize_segments(
    segments: list[tuple[float, float, str]], trim_duration_sec: float
) -> list[tuple[float, float, str]]:
    """Segment契約を検証し、厳密値へ正規化した上で返す。

    一次検証(許容誤差50ミリ秒以内)→正規化→再検証(許容誤差なしの厳密な等号/不等号)の2段構成。
    再検証で1件でも違反すれば、一次検証の許容誤差設定がその音声には不適切だったとみなし
    `RecognitionError`にする(値を調整して通さず失敗として扱う)。
    """
    _check_segment_contract(segments, trim_duration_sec, tolerance=_SEGMENT_CONTRACT_TOLERANCE_SEC)
    normalized = _normalize_segments(segments, trim_duration_sec)
    _check_segment_contract(normalized, trim_duration_sec, tolerance=0.0)
    return normalized


def _align_batch(
    targets: list[tuple[np.ndarray, int, list[str]]], config: SofaAlignerConfig
) -> dict[str, list[tuple[float, float, str]]]:
    """targetsをまとめて1回のSOFA呼び出しで処理し、basenameごとの生セグメント列を返す。

    Segment契約の検証・IPA写像は含まない(呼び出し元が別途`_validate_and_normalize_segments`等で
    行う)。非ASCIIパスの拒否(`_check_ascii_paths`)と実行環境パスの実在検証
    (`_check_paths_exist`)は本関数がSOFA起動前に行う。configの相対パスはプロセスの
    作業ディレクトリ基準で絶対化してから検証・起動に使う。targetsが空ならサブプロセスを
    起動せず空の結果を返す。
    """
    if not targets:
        return {}

    # SOFAサブプロセスは作業ディレクトリをsofa_rootにして動くため、相対パスのままでは
    # 起動前検証(このプロセスの作業ディレクトリ基準)と実行時の解決先が食い違い、検証を
    # 通過したパスが実行時に見つからなくなる。検証・起動の前に絶対化して食い違いを断つ。
    config = replace(
        config,
        sofa_python=config.sofa_python.absolute(),
        sofa_root=config.sofa_root.absolute(),
        checkpoint_path=config.checkpoint_path.absolute(),
    )

    # ignore_cleanup_errors=True: 中断時にプロセスツリーを終了しても、Windows では子プロセスの
    # ファイルハンドル解放に遅延がありうる。片付け失敗を無視しないと、その例外が伝播中の
    # KeyboardInterrupt を上書きしてしまう。
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        work_dir = Path(tmp)
        _check_ascii_paths(work_dir, config)
        _check_paths_exist(config)
        basenames = _write_sofa_inputs(work_dir, targets)
        _run_sofa_subprocess(basenames, config, work_dir)
        return {
            basename: _parse_htk_label_file(work_dir / "htk" / "phones" / f"{basename}.lab")
            for basename in basenames
        }


def _clamp_words_to_valid_list(
    words: list[tuple[list[str], float, float]], trim_duration_sec: float
) -> list[tuple[list[str], float, float]]:
    """単語タイムスタンプ列から「有効な単語列」を確定する(cursorベースの逐次クランプ)。

    各wordは(音素記号列, 開始秒, 終了秒)。(1)各単語の終了時刻をトリム後区間の全長以下へ再クランプ
    する。(2) `cursor`を0.0で初期化し単語を時系列順に処理する。各単語の開始時刻を
    `max(元の開始時刻, cursor)`へクランプし、クランプ後の区間長が最小単語長未満(ゼロ・負長を含む)、
    または音素記号列が空の場合はその単語を無効とする(`cursor`は更新しない)。そうでなければ有効とし
    `cursor`をクランプ後の終了時刻へ更新する。
    """
    cursor = 0.0
    valid_words: list[tuple[list[str], float, float]] = []
    for phoneme_symbols, start, end in words:
        clamped_end = min(end, trim_duration_sec)
        clamped_start = max(start, cursor)
        if not phoneme_symbols or (clamped_end - clamped_start) < _MIN_WORD_DURATION_SEC:
            continue
        valid_words.append((phoneme_symbols, clamped_start, clamped_end))
        cursor = clamped_end
    return valid_words


def _determine_word_gaps(
    valid_words: list[tuple[list[str], float, float]], trim_duration_sec: float
) -> list[tuple[float, float]]:
    """「有効な単語列」の隙間からgap区間を確定する。

    先頭から最初の有効単語まで・有効単語同士の間・最後の有効単語からトリム後区間の終端まで、の
    3種類。長さ0の隙間は生成しない。有効な単語列が空なら、トリム後区間全体を単一のgapとする。
    """
    if not valid_words:
        return [(0.0, trim_duration_sec)]

    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for _, start, end in valid_words:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = end
    if cursor < trim_duration_sec:
        gaps.append((cursor, trim_duration_sec))
    return gaps


def _map_symbol_to_segment_fields(symbol: str) -> tuple[Literal["vowel", "consonant", "gap"], str | None]:
    """SOFAが返す生の音素記号(G2P由来。AP/SP挿入を含む)をSegmentのtype・phonemeへ変換する。

    pau・cl(blank記号)・AP(吸気音・呼吸音)・SP(無音)はいずれもgapへ倒す。
    それ以外はG2P記号→音素モデル語彙の写像表でIPA記号へ変換してから、写像後のIPA記号を対象と
    する分類基準でtypeを定める。写像表に無い記号は`RecognitionError`にする(黙って捨てない)。
    """
    if symbol in _BLANK_G2P_SYMBOLS or symbol in ("AP", "SP"):
        return "gap", None
    vocab_symbol = _G2P_TO_VOCAB_SYMBOL.get(symbol)
    if vocab_symbol is None:
        raise RecognitionError(f"SOFA出力記号 '{symbol}' の音素モデル語彙への写像が未定義です")
    return _classify_symbol(vocab_symbol), vocab_symbol


def _segments_from_raw(raw_segments: list[tuple[float, float, str]]) -> list[Segment]:
    """SOFA出力の生セグメント列(秒・生記号)をSegment列へ変換する(IPA写像を適用)。"""
    segments = []
    for start, end, symbol in raw_segments:
        seg_type, phoneme = _map_symbol_to_segment_fields(symbol)
        segments.append(
            Segment(type=seg_type, start_sec=start, end_sec=end, phoneme=phoneme, confidence=None)
        )
    return segments
