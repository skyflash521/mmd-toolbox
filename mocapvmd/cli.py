"""mocapvmd CLI(mocapvmd.md §3)。

引数解析 → VMD読み(mmd_toolbox.vmd.io)→ 全ボーンの一般ノイズ軽減(クリーニング)→
足IK・つま先IKの接地安定化 → 共通機構による疎化 → VMD書き。ボーン選択は持たず、一般ノイズ軽減と
疎化は全ボーン、足IK安定化は分類 foot_ik / toe_ik のボーンに適用する。対象外セクション(モーフ・
カメラ・照明・セルフ影)は無加工で透過する。既定では疎なキーとベジェ補間を出力し、--no-reduce 時のみ
クリーニング後の密キー(線形補間)を出力する。

終了コード: 0 正常 / 1 入力不正(VMDでない・値が非有限等)/ 2 引数エラー / 3 出力書き込み失敗。
"""

import argparse
import dataclasses
import math
import os
import sys

from mmd_toolbox.vmd import io
from mmd_toolbox.vmd.reduce import BONE_LINEAR_INTERP
from mmd_toolbox.vmd.types import BoneKey

from . import classify, denoise, footik, presets, progress, reduce, report


def _build_parser():
    p = argparse.ArgumentParser(prog="mocapvmd", allow_abbrev=False)
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--preset", choices=presets.PRESET_NAMES, default="balanced")
    p.add_argument("--denoise", dest="denoise", action="store_true", default=True)
    p.add_argument("--no-denoise", dest="denoise", action="store_false")
    p.add_argument("--foot-ik-stabilize", dest="foot_ik_stabilize", action="store_true", default=True)
    p.add_argument("--no-foot-ik-stabilize", dest="foot_ik_stabilize", action="store_false")
    p.add_argument(
        "--reduce-preset", dest="reduce_preset", choices=presets.REDUCTION_PRESET_NAMES, default="balanced"
    )
    p.add_argument("--reduce-error-bone-pos", dest="reduce_error_bone_pos", type=float, default=None)
    p.add_argument("--reduce-error-bone-rot", dest="reduce_error_bone_rot", type=float, default=None)
    p.add_argument("--curve-mode", dest="curve_mode", choices=("bezier", "linear"), default="bezier")
    p.add_argument("--no-reduce", dest="reduce", action="store_false", default=True)
    p.add_argument("--quiet", dest="quiet", action="store_true")
    p.add_argument("--list-bones", dest="list_bones", action="store_true")
    p.add_argument("--report-json", dest="report_json")
    p.add_argument("--preview-csv", dest="preview_csv")
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


def _print_read_warnings(read_warnings):
    """読み込み警告(デコード不能な名前フィールド等)を surface する。

    同一(コード・セクション・メッセージ)はキー毎の重複を避けて1行にまとめる。
    """
    seen_warn = set()
    for w in read_warnings:
        key = (w.code, w.section, w.message)
        if key in seen_warn:
            continue
        seen_warn.add(key)
        where = f"({w.section})" if w.section else ""
        print(f"警告: {w.message}{where}", file=sys.stderr)


def _list_bones_text(bone_keys):
    """各ボーンの名前と分類を初出順・名前ごとに1行で返す(--list-bones / §3.2)。"""
    order = []
    seen = set()
    for k in bone_keys:
        if k.name not in seen:
            seen.add(k.name)
            order.append(k.name)
    return "\n".join(f"{name} [{classify.classify(name)}]" for name in order)


def _validate_bones(bone_keys):
    """全ボーンキーの値の健全性(非有限・ゼロノルム quaternion)を検証する(§3.3 入力不正検出)。

    クリーニング/疎化の有無に依らず入力不正を弾くため、パイプライン前に全キーの位置・回転を検証する。
    不正があれば ValueError を送出する。
    """
    denoise.validate_bone_values([k.position for k in bone_keys], [k.rotation for k in bone_keys])


def _clean_bones(bone_keys, preset):
    """全ボーンを種別別パラメータで一般ノイズ軽減し、密キー(線形補間)で返す(§4.1, §4.2)。

    各トラックを名前ごとに時系列順へまとめ、クリーニング後の密サンプルを線形補間キーとして組み直す
    (§3.3 のクリーニング後の密キー形式)。キー1個以下のトラックは平滑化できないため逐語保持する。
    値の健全性は呼び出し前に _validate_bones で検証済みとする。
    """
    order = []
    groups = {}
    for k in bone_keys:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)

    out = []
    for name in order:
        ks = sorted(groups[name], key=lambda k: k.frame)
        positions = [k.position for k in ks]
        rotations = [k.rotation for k in ks]
        if len(ks) < 2:
            out.extend(ks)
            continue
        params = presets.resolve_cleaning(preset, classify.classify(name))
        cpos, crot = denoise.apply_denoise(
            positions,
            rotations,
            pos_window=params["pos_window"],
            rot_window=params["rot_window"],
            pos_strength=params["pos_strength"],
            rot_strength=params["rot_strength"],
        )
        name_raw = ks[0].name_raw
        for i, k in enumerate(ks):
            out.append(BoneKey(name_raw, k.frame, cpos[i], crot[i], BONE_LINEAR_INTERP))
    out.sort(key=lambda k: (k.name_raw, k.frame))
    return out


def _stabilize_bones(bone_keys, preset):
    """分類 foot_ik / toe_ik の密トラックに接地安定化を適用し、密キー(線形補間)で返す(§4.3)。

    左右ペアリング・接地検出・接地ロックは footik.stabilize_foot_ik に委譲する。foot_ik / toe_ik 以外の
    ボーンと、キー1個以下のトラックは逐語透過する。回転は接地ロック対象外なので元の値を保つ。
    """
    order = []
    groups = {}
    for k in bone_keys:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)

    tracks = {}
    for name in order:
        ks = sorted(groups[name], key=lambda k: k.frame)
        category = classify.classify(name)
        if category in ("foot_ik", "toe_ik") and len(ks) >= 2:
            tracks[name] = (category, [k.frame for k in ks], [k.position for k in ks])
    if not tracks:
        return bone_keys

    stabilized = footik.stabilize_foot_ik(tracks, preset)

    out = []
    for name in order:
        ks = sorted(groups[name], key=lambda k: k.frame)
        if name in stabilized:
            locked = stabilized[name].locked_positions
            name_raw = ks[0].name_raw
            for i, k in enumerate(ks):
                out.append(BoneKey(name_raw, k.frame, locked[i], k.rotation, BONE_LINEAR_INTERP))
        else:
            out.extend(ks)
    out.sort(key=lambda k: (k.name_raw, k.frame))
    return out


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

    # --list-bones は書き込み・疎化をしない診断モード。出力先・上書きガードや疎化許容値の検証(処理・
    # 書き込み固有)を行わず、読み込んでボーン一覧と分類を表示して終了する(§3.2)。通常経路の引数エラー
    # (終了コード2)優先順位を保つため、これらの検証は通常経路でのみ読み込み前に行う。
    if args.list_bones:
        try:
            doc, read_warnings = io.read(args.input)
        except Exception:
            return 1
        _print_read_warnings(read_warnings)
        print(_list_bones_text(doc.bone))
        return 0

    # 出力先・上書きガード(§3.2)。入力と同一パスへの出力は --overwrite が必要。
    output = args.output if args.output is not None else _default_output(args.input)
    if not args.overwrite and _same_path(output, args.input):
        return 2

    # 疎化の許容誤差 override は非有限・負を引数エラー(終了コード2)とする(§3.2 / §5.3)。
    for v in (args.reduce_error_bone_pos, args.reduce_error_bone_rot):
        if v is not None and (not math.isfinite(v) or v < 0.0):
            return 2

    # 入力読み込み(VMDでない等 → 入力不正)。
    try:
        doc, read_warnings = io.read(args.input)
    except Exception:
        return 1
    _print_read_warnings(read_warnings)

    # 入力ボーン値の健全性(非有限・ゼロノルム quaternion)は処理経路(クリーニング/疎化の有無・dry-run か)
    # に依らず、パイプライン前に全キーを検証する(§3.3 入力不正=終了コード1)。dry-run でも疎化レポート
    # (§4.4)のため疎化を実行するので、未検証の不正値が疎化へ流れて逐語透過・例外化するのを防ぐ。
    try:
        _validate_bones(doc.bone)
    except ValueError:
        return 1

    # クリーニング → 足IK安定化 → 疎化のパイプライン。dry-run でも疎化レポート(§4.4)の素データを得るため
    # 実行し、出力の書き出しだけを dry-run で省く。一般ノイズ軽減は全ボーン(--no-denoise 時は逐語透過)、
    # 足IK安定化は分類 foot_ik / toe_ik(§4.3 / §6)、疎化は全ボーン(--no-reduce 時は密キーのまま)に適用する。
    # 進捗のライブ表示。重い疎化の進行を端末へ出す(--quiet で無効、既定は stderr が端末のときだけ)。
    # 各段を begin_stage/end_stage で囲み、疎化は per-bone の reporter.update を progress に渡す。例外時も
    # finally でハートビートを止め行を確定するため try/finally で囲む。読み書きは速い I/O なので段にしない。
    reporter = progress.ProgressReporter(sys.stderr, enabled=False if args.quiet else None)
    # 疎化レポートを出すときだけ診断 diagnostics_out を集める(通常実行ではオーバーヘッドを避ける)。
    want_report = args.dry_run or args.report_json or args.preview_csv
    reduction_diag = {} if (args.reduce and want_report) else None
    try:
        if args.denoise:
            reporter.begin_stage("クリーニング")
            new_bone = _clean_bones(doc.bone, args.preset)
            reporter.end_stage()
        else:
            new_bone = doc.bone
        if args.foot_ik_stabilize:
            reporter.begin_stage("足IK安定化")
            new_bone = _stabilize_bones(new_bone, args.preset)
            reporter.end_stage()
        if args.reduce:
            reporter.begin_stage("疎化")
            new_bone = reduce.reduce_bones(
                new_bone,
                args.reduce_preset,
                override_pos=args.reduce_error_bone_pos,
                override_rot=args.reduce_error_bone_rot,
                curve_mode=args.curve_mode,
                diagnostics_out=reduction_diag,
                progress=reporter.update,
            )
            reporter.end_stage()
    finally:
        reporter.close()

    # 診断レポート(dry-run 表示・report-json 出力)。疎化したときは §4.4 の疎化レポート(reduction)も載せる。
    if want_report:
        rep = report.build_report(
            doc.bone,
            args.preset,
            denoise=args.denoise,
            foot_ik_stabilize=args.foot_ik_stabilize,
            reduction=reduction_diag,
        )
        if args.dry_run:
            print(report.format_dry_run(rep))
        if args.report_json:
            try:
                report.write_json(rep, args.report_json)
            except OSError:
                return 3
        # --preview-csv: 入力(クリーニング前)と出力(疎化後)のフレーム毎サンプル比較を CSV 出力する(§3.2)。
        if args.preview_csv:
            try:
                report.write_preview_csv(report.bone_preview_rows(doc.bone, new_bone), args.preview_csv)
            except OSError:
                return 3

    # dry-run は出力を書かずに終える。
    if args.dry_run:
        return 0

    out_doc = dataclasses.replace(doc, bone=new_bone)
    try:
        io.write_file(out_doc, output)
    except Exception:
        return 3
    return 0
