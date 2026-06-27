"""vpr2vmd CLI(vpr2vmd.md §4)。

vpr を入力に、口形イベント列と開き量を作って lipsync に渡し、口パク VMD を 1 コマンドで
出力する薄いラッパー。本ファイルは引数解析・検証・出力先解決と --dry-run の空実行を担う。
実際の変換パイプライン(vpr 読み込み → 口形イベント確定 → 開き量 → lipsync → VMD 出力)は
後続ステップで _convert() に実装する。

終了コード(既存ツール shakevmd 等の規約に倣う):
0 正常 / 1 入力不正(入力 vpr の欠落・非vpr 等) / 2 引数エラー(未知オプション・範囲不正・
上書きガード) / 3 出力書き込み失敗。
"""

import argparse
import math
import os
import sys

# 口パクスタイルプリセット名(vpr2vmd.md §4.2)。プリセットごとの開き量レンジ・タイミング等の
# 具体値の解決は後続の実装で行う。
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
    # 対象トラックを持つ後続の実装で行うため、ここでは生文字列のまま保持する(type=str)。
    p.add_argument("--track")
    p.add_argument("--model-name", dest="model_name", type=_model_name, default="")
    p.add_argument("--style", choices=STYLE_NAMES, default="pop")
    # 既定はプリセット値。未指定センチネル(None)を後続ステップでプリセットから解決する。
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
    print(f"model-name: {args.model_name!r}")
    print(f"open-max: {args.open_max if args.open_max is not None else '(プリセット値)'}")
    print(
        f"default-open: "
        f"{args.default_open if args.default_open is not None else '(プリセット値)'}"
    )


def _convert(args, output: str) -> int:
    """vpr → 口パク VMD の変換本体。後続ステップ(vpr 読み込み・口形イベント確定・開き量・
    lipsync・VMD 出力)で実装する。"""
    raise NotImplementedError("vpr→VMD 変換は未実装(後続ステップで実装)")


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

    # 入力 vpr の存在確認(欠落は入力不正)。読み込み・形式検証は後続ステップ。
    if not os.path.isfile(args.input):
        return 1

    if args.dry_run:
        _print_plan(args, output)
        return 0

    return _convert(args, output)
