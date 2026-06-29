"""vpr2vmd CLI(vpr2vmd.md §4)。

vpr を入力に、口形イベント列と開き量を作って lipsync に渡し、口パク VMD を 1 コマンドで
出力する薄いラッパー。引数解析・検証・出力先解決と --dry-run の空実行を担い、変換パイプライン
(vpr 読み込み → 口形イベント確定 → 開き量 → lipsync → VMD 出力)は _convert() が束ねる。

終了コード(既存ツール shakevmd 等の規約に倣う):
0 正常 / 1 入力不正(入力 vpr の欠落・非vpr 等) / 2 引数エラー(未知オプション・範囲不正・
上書きガード) / 3 出力書き込み失敗。
"""

import argparse
import math
import os
import sys

from lipsync import generate_morph_keys
from mmd_toolbox.vmd import VmdDocument, ensure_frame0_neutral_keys, normalize, write_file
from vpr_io import VprFormatError, read

from . import openness, presets, timing
from .events import build_mouth_events, resolve_overlaps
from .io import TrackSelectionError, collect_notes, select_track
from .tempo_correction import apply_tempo_correction

# 口パクスタイルプリセット名(vpr2vmd.md §4.2)。具体値の解決は presets.resolve が担う。
STYLE_NAMES = ("pop", "ballad", "powerful", "whisper", "rap")

# VMD ヘッダのモデル名は固定 20 バイト・Shift-JIS(vpr2vmd.md §4.2・§5)。
_MODEL_NAME_MAX_BYTES = 20


def _model_name(text: str) -> str:
    """--model-name を検証する。cp932 で表現でき、20 バイト以内であること。"""
    try:
        encoded = text.encode("cp932")
    except UnicodeEncodeError:
        raise argparse.ArgumentTypeError(
            f"モデル名は Shift-JIS(cp932)で表現できる文字のみ: {text!r}"
        )
    if len(encoded) > _MODEL_NAME_MAX_BYTES:
        raise argparse.ArgumentTypeError(
            f"モデル名は cp932 で {_MODEL_NAME_MAX_BYTES} バイト以内"
            f"({len(encoded)} バイト): {text!r}"
        )
    return text


def _open_amount(text: str) -> float:
    """開き量(--open-max / --default-open)を検証する。0.0〜1.0 の有限 float。

    開き量は lipsync の口形開き量(0〜1)・開き量上限(open_cap, 0〜1)に対応する量なので、
    範囲外(負値・1 超)・inf/nan は引数エラーにする。
    """
    v = float(text)  # 非数値は ValueError → argparse が exit 2 にする
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"有限な数値が必要: {text!r}")
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"開き量は 0.0〜1.0 の範囲: {text!r}")
    return v


def _build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: 仕様外の前置き省略形を受理しない(未知/省略形は exit 2)。
    p = argparse.ArgumentParser(prog="vpr2vmd", allow_abbrev=False)
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--overwrite", action="store_true")
    # --track は整数なら 0-based INDEX、非整数なら Track.name(vpr2vmd.md §4.2)。解釈・解決は
    # io.select_track が行うため、ここでは生文字列のまま保持する(type=str)。
    p.add_argument("--track")
    p.add_argument("--model-name", dest="model_name", type=_model_name, default="")
    p.add_argument("--style", choices=STYLE_NAMES, default="pop")
    # 既定は「ん」モーフを使う。指定時は撥音「ん」を無音(閉口)へ倒す(vpr2vmd.md §4.2)。
    p.add_argument("--no-n-morph", dest="no_n_morph", action="store_true")
    # 既定はプリセット値。未指定センチネル(None)は presets.resolve がプリセットから解決する。
    p.add_argument("--open-max", dest="open_max", type=_open_amount)
    p.add_argument("--default-open", dest="default_open", type=_open_amount)
    p.add_argument("--report-json", dest="report_json")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


def _default_output(input_path: str) -> str:
    # vpr2vmd.md §4.1: 既定出力は <入力名(拡張子なし)>.vmd。元の拡張子に依らず常に .vmd。
    base, _ = os.path.splitext(input_path)
    return base + ".vmd"


def _same_path(a: str, b: str) -> bool:
    """2 パスが同一ファイルを指すか。未存在でも realpath 比較で判定する。"""
    # 実ファイルが同一かを優先(symlink・大小無視 FS でも inode で一致判定)。出力先が
    # 未存在だと samefile が立たないので、symlink 解決した realpath で比較する。
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


def _print_plan(args, output: str) -> None:
    """--dry-run の処理計画表示(出力は書かない)。"""
    print(f"input: {args.input}")
    print(f"output: {output}")
    print(f"track: {args.track if args.track is not None else '(先頭トラック)'}")
    print(f"style: {args.style}")
    # 既定は撥音「ん」に「ん」モーフを使う(on)。--no-n-morph 指定時は無音へ倒す(off)。
    print(f"n-morph: {'off (撥音→無音)' if args.no_n_morph else 'on (撥音→ん)'}")
    print(f"model-name: {args.model_name!r}")
    print(f"open-max: {args.open_max if args.open_max is not None else '(プリセット値)'}")
    print(
        f"default-open: "
        f"{args.default_open if args.default_open is not None else '(プリセット値)'}"
    )


def _convert(args, output: str) -> int:
    """vpr → 口パク VMD の変換本体(vpr2vmd.md §3〜§5)。

    vpr_io 解析 → 対象トラック選択 → 重なり解決 → 口形イベント確定 → 開き量 → lipsync →
    モーフキー VMD 出力を束ねる。終了コードは vpr2vmd.md §4.3 に従う(形式失敗・対象トラック
    皆無=1、`--track` の不正値=2、出力書き込み失敗=3、それ以外は 0)。
    """
    try:
        project, _warnings = read(args.input)
    except VprFormatError:
        return 1  # 読み込み・形式検証の失敗(非vpr など)=入力不正
    if not project.tracks:
        return 1  # 対象トラックが1件も無い=入力不正
    try:
        track = select_track(project, args.track)
    except TrackSelectionError:
        return 2  # --track の値が当該入力で有効な選択にならない=引数エラー

    adopted = resolve_overlaps(collect_notes(track))
    openness_params, gen_params = presets.resolve(args.style, args.open_max, args.default_open)
    # 曲の代表BPMから保持・アタック・リリースを縮める(高速テンポでの短母音消失を防ぐ)。
    rep_bpm = timing.representative_bpm(adopted, project.tempos, project.resolution)
    gen_params = apply_tempo_correction(gen_params, rep_bpm)
    open_by_note = openness.open_amounts(
        [note.velocity for note in adopted],
        lo=openness_params.lo,
        hi=openness_params.hi,
        open_max=openness_params.open_max,
        default_open=openness_params.default_open,
        gamma=openness_params.gamma,
    )
    mouth_events = build_mouth_events(
        adopted,
        project.tempos,
        project.resolution,
        use_n_morph=not args.no_n_morph,
        open_by_note=open_by_note,
    )
    morph_keys = generate_morph_keys(mouth_events, gen_params)

    document = VmdDocument(
        model_name_raw=args.model_name.encode("cp932").ljust(20, b"\x00"),
        morph=morph_keys,
    )
    # 使用モーフを 0F に中立登録してから(編集・MMD互換規約)フレーム順へ正規化する。
    document = ensure_frame0_neutral_keys(document, sections=("morph",))
    document, _warnings = normalize(document, sections=["morph"])
    try:
        write_file(document, output)
    except OSError:
        return 3  # 出力VMDの書き込み失敗
    return 0


def main(argv=None) -> int:
    """CLI エントリポイント。終了コードを返す(0/1/2/3)。"""
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse はエラー時 code 2 で sys.exit(--help は 0)。例外を握って終了コードに変換。
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)

    output = args.output if args.output is not None else _default_output(args.input)

    # 上書きガード(vpr2vmd.md §4.2): --overwrite なしでは既存出力を上書きしない。出力先が
    # 入力と同一パスのときは、出力がまだ無くても(=入力を上書きする指定なので)同様に弾く。
    # 同一パス判定を存在確認より先に置くことで、未存在でも入力上書き指定は引数エラーになる。
    if not args.overwrite and (_same_path(output, args.input) or os.path.exists(output)):
        return 2

    # 入力 vpr の存在確認(欠落は入力不正)。読み込み・形式検証は _convert が行う。
    if not os.path.isfile(args.input):
        return 1

    if args.dry_run:
        _print_plan(args, output)
        return 0

    return _convert(args, output)
