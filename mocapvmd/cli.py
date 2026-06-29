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

from mmd_toolbox.pmx.types import PmxFormatError
from mmd_toolbox.vmd import io
from mmd_toolbox.vmd.reduce import BONE_LINEAR_INTERP
from mmd_toolbox.vmd.types import BoneKey

from . import __version__, classify, denoise, footik, presets, progress, reduce, report
from .model_profile import MocapModelProfileError
from .pose_denoise import apply_pose_denoise


def _build_parser():
    p = argparse.ArgumentParser(prog="mocapvmd", allow_abbrev=False)
    p.add_argument("--version", action="version", version=f"mocapvmd {__version__}")
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--preset", choices=presets.PRESET_NAMES, default="medium")
    p.add_argument("--clean-strength", dest="clean_strength", type=float, default=1.0)
    p.add_argument("--denoise", dest="denoise", action="store_true", default=True)
    p.add_argument("--no-denoise", dest="denoise", action="store_false")
    p.add_argument("--denoise-mode", dest="denoise_mode", choices=("bone", "pose"), default="bone")
    p.add_argument("--pmx", dest="pmx", default=None)
    p.add_argument("--foot-ik-stabilize", dest="foot_ik_stabilize", action="store_true", default=True)
    p.add_argument("--no-foot-ik-stabilize", dest="foot_ik_stabilize", action="store_false")
    p.add_argument("--foot-slide-suppression", dest="foot_slide_suppression", type=float, default=1.0)
    p.add_argument("--reduce-error-bone-pos", dest="reduce_error_bone_pos", type=float, default=None)
    p.add_argument("--reduce-error-bone-rot", dest="reduce_error_bone_rot", type=float, default=None)
    p.add_argument("--curve-mode", dest="curve_mode", choices=("bezier", "linear"), default="bezier")
    p.add_argument("--no-reduce", dest="reduce", action="store_false", default=True)
    p.add_argument("--quiet", dest="quiet", action="store_true")
    p.add_argument("--list-bones", dest="list_bones", action="store_true")
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


def _clean_bones(bone_keys, clean_strength):
    """全ボーンを種別別パラメータで一般ノイズ軽減し、密キー(線形補間)で返す(§4.1, §4.2)。

    各トラックを名前ごとに時系列順へまとめ、クリーニング後の密サンプルを線形補間キーとして組み直す
    (§3.3 のクリーニング後の密キー形式)。クリーニング強度の倍率 clean_strength を種別別の基準ブレンド率へ
    掛ける(§5.2)。キー1個以下のトラックは平滑化できないため逐語保持する。値の健全性は呼び出し前に
    _validate_bones で検証済みとする。
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
        params = presets.resolve_cleaning(clean_strength, classify.classify(name))
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


def _stabilize_bones(bone_keys, suppression):
    """分類 foot_ik / toe_ik の密トラックに接地安定化を適用し、密キー(線形補間)で返す(§4.3)。

    左右ペアリング・接地検出・接地ロックは footik.stabilize_foot_ik に委譲する。横滑り抑制 S(0〜1)は
    検出・ロック強度・最大補正量を連動制御する。foot_ik / toe_ik 以外のボーンと、キー1個以下のトラックは
    逐語透過する。回転は接地ロック対象外なので元の値を保つ。
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

    stabilized = footik.stabilize_foot_ik(tracks, suppression)

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

    # クリーニング強度の倍率は非有限・負を引数エラー(終了コード2)とする(§3.2 / §5.2)。
    if not math.isfinite(args.clean_strength) or args.clean_strength < 0.0:
        return 2

    # 横滑り抑制は 0〜1 の有限値のみ許容(範囲外・非有限は引数エラー=終了コード2。§4.3 / §5.4)。
    s = args.foot_slide_suppression
    if not math.isfinite(s) or not 0.0 <= s <= 1.0:
        return 2

    # pose モードで --pmx 指定時は、パスの存在・通常ファイルを引数エラー(終了コード2)で検証する。
    if args.denoise and args.denoise_mode == "pose" and args.pmx is not None:
        if not os.path.isfile(args.pmx):
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
    # 足IK安定化は分類 foot_ik / toe_ik(§4.1 / §4.3)、疎化は全ボーン(--no-reduce 時は密キーのまま)に適用する。
    # 進捗のライブ表示。重い疎化の進行を端末へ出す(--quiet で無効、既定は stderr が端末のときだけ)。
    # 各段を stage で1本の行に切り替えて表示し、疎化は per-bone の reporter.update を progress に渡す。
    # 例外時もハートビートを止め行を消すため try/finally で囲む。読み書きは速い I/O なので段にしない。
    # 段ラベルは厳密な仕様用語でなく利用者向けの平易な文言にする(平滑化=クリーニング、足IK最適化=足IK
    # 安定化、キーフレーム圧縮=疎化)。
    reporter = progress.ProgressReporter(sys.stderr, enabled=False if args.quiet else None)
    # dry-run の診断表示を出すときだけ diagnostics_out を集める(通常実行ではオーバーヘッドを避ける)。
    want_report = args.dry_run
    reduction_diag = {} if (args.reduce and want_report) else None
    # pose モードの dry-run で表現空間ノイズ除去の診断素データを集める(§12)。
    pose_diag = {} if (args.denoise and args.denoise_mode == "pose" and want_report) else None
    try:
        if args.denoise:
            reporter.stage("平滑化")
            if args.denoise_mode == "pose":
                # 表現空間ノイズ除去。プロファイル不正・PMX形式不正は入力不正(終了コード1)。
                try:
                    new_bone = apply_pose_denoise(
                        doc.bone, pmx_path=args.pmx, diagnostics_out=pose_diag
                    )
                except (MocapModelProfileError, PmxFormatError):
                    return 1
            else:
                new_bone = _clean_bones(doc.bone, args.clean_strength)
        else:
            new_bone = doc.bone
        if args.foot_ik_stabilize:
            reporter.stage("足IK最適化")
            new_bone = _stabilize_bones(new_bone, args.foot_slide_suppression)
        if args.reduce:
            reporter.stage("キーフレーム圧縮")
            new_bone = reduce.reduce_bones(
                new_bone,
                args.preset,
                override_pos=args.reduce_error_bone_pos,
                override_rot=args.reduce_error_bone_rot,
                curve_mode=args.curve_mode,
                diagnostics_out=reduction_diag,
                progress=reporter.update,
            )
    finally:
        reporter.close()

    # dry-run 診断表示。疎化したときは §4.4 の疎化レポート(reduction)も載せる。
    if want_report:
        rep = report.build_report(
            doc.bone,
            args.preset,
            clean_strength=args.clean_strength,
            denoise=args.denoise,
            foot_ik_stabilize=args.foot_ik_stabilize,
            reduction=reduction_diag,
            suppression=args.foot_slide_suppression,
            pose_denoise=pose_diag,
        )
        if args.dry_run:
            print(report.format_dry_run(rep))

    # dry-run は出力を書かずに終える。
    if args.dry_run:
        return 0

    out_doc = dataclasses.replace(doc, bone=new_bone)
    try:
        io.write_file(out_doc, output)
    except Exception:
        return 3
    # 進捗表示が有効だった(端末・非 quiet)ときだけ、消した進捗行のあとに完了行を1行残す。
    reporter.summary(f"完了 {output}")
    return 0
