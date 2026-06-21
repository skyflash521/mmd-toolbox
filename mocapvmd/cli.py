"""mocapvmd CLI(mocapvmd.md §3)。

引数解析 → VMD読み(mmd_toolbox.vmd.io)→ クリーニング → 疎化 → VMD書き。
ボーン選択は持たず、全ボーンを処理対象とする。対象外セクション(モーフ・カメラ・
照明・セルフ影)は無加工で透過する。

終了コード: 0 正常 / 1 入力不正(VMDでない等)/ 2 引数エラー / 3 出力書き込み失敗。
"""

import argparse
import os
import sys

from mmd_toolbox.vmd import io

from . import report


def _build_parser():
    p = argparse.ArgumentParser(prog="mocapvmd", allow_abbrev=False)
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--report-json", dest="report_json")
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    return p


def _default_output(input_path):
    base, _ = os.path.splitext(input_path)
    return base + "_mocap.vmd"


def _same_path(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


def main(argv=None):
    if argv is None:
        argv = sys.argv[1:]
    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)

    # 入力パス検証(§3.3)。不在・非通常ファイルは引数エラー。
    if not os.path.isfile(args.input):
        return 2

    # 出力先・上書きガード(§3.2)。入力と同一パスへの出力は --overwrite が必要。
    output = args.output if args.output is not None else _default_output(args.input)
    if not args.overwrite and _same_path(output, args.input):
        return 2

    # 入力読み込み(VMDでない等 → 入力不正)。
    try:
        doc, read_warnings = io.read(args.input)
    except Exception:
        return 1

    # 読み込み時の警告(デコード不能な名前フィールド等)を surface する。
    # 同一(コード・セクション・メッセージ)はキー毎の重複を避けて1行にまとめる。
    seen_warn = set()
    for w in read_warnings:
        key = (w.code, w.section, w.message)
        if key in seen_warn:
            continue
        seen_warn.add(key)
        where = f"({w.section})" if w.section else ""
        print(f"警告: {w.message}{where}", file=sys.stderr)

    # 診断レポート(dry-run 表示・report-json 出力)。どのボーンにどの処理が適用される予定かを
    # 出力を変更せずに確認できる。
    if args.dry_run or args.report_json:
        rep = report.build_report(doc.bone)
        if args.dry_run:
            print(report.format_dry_run(rep))
        if args.report_json:
            try:
                report.write_json(rep, args.report_json)
            except OSError:
                return 3

    # dry-run は出力を書かずに終える。
    if args.dry_run:
        return 0

    # 読み込んだドキュメントをそのまま書き出す(対象外セクション・ボーンとも無加工で透過する)。
    try:
        io.write_file(doc, output)
    except Exception:
        return 3
    return 0
