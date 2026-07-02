"""mocapvmd CLI(mocapvmd.md §3 / §9 / §10)。

引数解析 → VMD読み(vmd.io)→ 全ボーンの一般ノイズ軽減(クリーニング)→
足IK・つま先IKの接地安定化 → 共通機構による疎化 → VMD書き。ボーン選択は持たず、一般ノイズ軽減と
疎化は全ボーン、足IK安定化は分類 foot_ik / toe_ik のボーンに適用する。対象外セクション(モーフ・
カメラ・照明・セルフ影)は無加工で透過する。既定では疎なキーとベジェ補間を出力し、--no-reduce 時のみ
クリーニング後の密キー(線形補間)を出力する。

--machine 指定時は標準出力を JSON Lines のイベントストリーム(progress / warning / result / error)に
切り替える(mocapvmd.md §10)。既定(非機械)の人間向け表示・出力ファイル・終了コードは変えない。

終了コード(§9): 0 正常 / 1 入力不正(VMDでない・値が非有限・PMX形式不正・モデルプロファイル不正)/
2 引数エラー / 3 出力書き込み失敗 / 130 協調的な中断(Ctrl-C 等)。
"""

import argparse
import dataclasses
import math
import os
import sys
import time

from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    argparse_error_field,
    error_event,
)
from pmx.types import PmxFormatError
from vmd import io
from vmd.reduce import BONE_LINEAR_INTERP
from vmd.types import BoneKey

from . import __version__, classify, denoise, footik, presets, progress, reduce, report
from .model_profile import MocapModelProfileError
from .pose_denoise import apply_pose_denoise


def _build_parser(machine=False):
    # 構造化出力モード(--machine)は使用法エラーを error イベントへ振り替えるため、SystemExit の代わりに
    # ArgumentParseError を送出する MachineArgumentParser を使う(--help/--version は error() を経由しない
    # ので影響を受けず、従来どおり SystemExit で短絡する)。
    cls = MachineArgumentParser if machine else argparse.ArgumentParser
    p = cls(prog="mocapvmd", allow_abbrev=False)
    p.add_argument("--version", action="version", version=f"mocapvmd {__version__}",
                   help="バージョンを表示して終了する")
    p.add_argument("--machine", action="store_true",
                   help="出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・"
                        "標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない")
    p.add_argument("input", help="入力VMDファイル")
    p.add_argument("-o", "--output", help="出力先(既定: <入力名>_mocap.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="入力と同一パスへの出力を許可する(未指定で同一パスならエラー)")
    p.add_argument("--preset", choices=presets.PRESET_NAMES, default="medium",
                   help="疎化の許容誤差プリセット(速度観点): slower/slow/medium/fast/faster。種別スケールの基準値")
    p.add_argument("--clean-strength", dest="clean_strength", type=float, default=1.0,
                   help="クリーニング強度の倍率(ブレンド率に掛ける。0で無加工相当、上げるほど強く均す。窓幅は据え置き)")
    p.add_argument("--denoise", dest="denoise", action="store_true", default=True,
                   help="一般ノイズ軽減を有効化(既定 on)")
    p.add_argument("--no-denoise", dest="denoise", action="store_false",
                   help="一般ノイズ軽減を無効化")
    p.add_argument("--denoise-mode", dest="denoise_mode", choices=("bone", "pose"), default="bone",
                   help="一般ノイズ軽減の方式: bone(ボーン単位)/ pose(表現空間)")
    p.add_argument("--pmx", dest="pmx", default=None,
                   help="pose 方式が参照するモデルPMX(未指定時は内蔵の既定モデルプロファイル)")
    p.add_argument("--foot-ik-stabilize", dest="foot_ik_stabilize", action="store_true", default=True,
                   help="足IK接地安定化を有効化(既定 on)")
    p.add_argument("--no-foot-ik-stabilize", dest="foot_ik_stabilize", action="store_false",
                   help="足IK接地安定化を無効化")
    p.add_argument("--foot-slide-suppression", dest="foot_slide_suppression", type=float, default=1.0,
                   help="足IK接地中の横滑り抑制の強さ(0〜1)。既定1.0は接地中の横滑りを除去する")
    p.add_argument("--reduce-error-bone-pos", dest="reduce_error_bone_pos", type=float, default=None,
                   help="疎化の位置許容誤差の基準値(MMD単位)。明示値はプリセットに優先。種別スケールを掛ける")
    p.add_argument("--reduce-error-bone-rot", dest="reduce_error_bone_rot", type=float, default=None,
                   help="疎化の回転許容誤差の基準値(度)。明示値はプリセットに優先。種別スケールを掛ける")
    p.add_argument("--curve-mode", dest="curve_mode", choices=("bezier", "linear"), default="bezier",
                   help="出力補間曲線: bezier / linear")
    p.add_argument("--reduce", dest="reduce", action="store_true", default=True,
                   help="疎化を有効化(既定 on)。クリーニング後の信号を疎なキー+ベジェ補間で出力する")
    p.add_argument("--no-reduce", dest="reduce", action="store_false",
                   help="疎化せずクリーニング後の密キー(線形補間)を出力する(診断・比較用)")
    p.add_argument("--list-bones", dest="list_bones", action="store_true",
                   help="ボーン一覧と分類結果を表示して終了する")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力せず処理計画と診断を表示する(引数検証は実施する)")
    p.add_argument("--quiet", dest="quiet", action="store_true",
                   help="進捗表示を抑制する(警告・診断・終了コードは抑制しない)")
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                   help="通常実行でも 4.4 のレポートを標準出力へ表示する(出力VMDは書く)")
    return p


def _default_output(input_path):
    base, _ = os.path.splitext(input_path)
    return base + "_mocap.vmd"


def _same_path(a, b):
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


def _surface_warnings(read_warnings, machine, emitter):
    """読み込み警告(デコード不能な名前フィールド等)を surface する(§10.2)。

    同一(コード・セクション・メッセージ)はキー毎の重複を避けて 1 件にまとめる。機械モードは warning
    イベント(section は単一要素配列 or null)、非機械は標準エラーへ 1 行出す。
    """
    seen_warn = set()
    for w in read_warnings:
        key = (w.code, w.section, w.message)
        if key in seen_warn:
            continue
        seen_warn.add(key)
        if machine:
            emitter.warning(code=w.code, message=w.message,
                            section=[w.section] if w.section else None)
        else:
            where = f"({w.section})" if w.section else ""
            print(f"警告: {w.message}{where}", file=sys.stderr)


def _bone_order(bone_keys):
    """ボーン名を初出順・名前ごとに 1 つずつ返す(--list-bones / inspect)。"""
    order = []
    seen = set()
    for k in bone_keys:
        if k.name not in seen:
            seen.add(k.name)
            order.append(k.name)
    return order


def _list_bones_text(bone_keys):
    """各ボーンの名前と分類を初出順・名前ごとに1行で返す(--list-bones / §3.2)。"""
    return "\n".join(f"{name} [{classify.classify(name)}]" for name in _bone_order(bone_keys))


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


def _build_inspect(args, doc, reduction_diag, pose_diag):
    """--machine --dry-run の inspect result ペイロードを組む(§10.2)。VMD は書かない。

    keys/frame_range/duration/sections は入力から、preset〜reduce は解決済み実行計画(引数値)から、
    reduction は reduce.reduce_bones の診断(--no-reduce 時 None)、pose_denoise は apply_pose_denoise の
    診断の部分集合(pose 方式かつ有効時のみ、それ以外 None)。数値は JSON 安全な素の float/int へ寄せる。
    """
    order = []
    groups = {}
    for k in doc.bone:
        if k.name not in groups:
            groups[k.name] = []
            order.append(k.name)
        groups[k.name].append(k)
    all_frames = [k.frame for k in doc.bone]
    frame_range = [min(all_frames), max(all_frames)] if all_frames else None
    duration = (max(all_frames) / 30.0) if all_frames else None
    bones = []
    for name in order:
        frames = [k.frame for k in groups[name]]
        bones.append({
            "name": name,
            "category": classify.classify(name),
            "keys": len(groups[name]),
            "frame_range": [min(frames), max(frames)],
        })
    sections = [
        name for name, present in (
            ("bone", doc.bone), ("morph", doc.morph), ("camera", doc.camera),
            ("light", doc.light), ("self_shadow", doc.self_shadow), ("ik_property", doc.ik_property),
        ) if present
    ]
    pose = None
    if pose_diag:
        fit = pose_diag["fit"]
        md = pose_diag["marker_displacement"]
        mk = pose_diag["markers"]
        pose = {
            "pmx": pose_diag["pmx"],
            "frames": int(pose_diag["frames"]),
            "markers": {"available": int(mk["available"]),
                        "required_bones_ok": bool(mk["required_bones_ok"])},
            "marker_displacement": {"max": float(md["max"]), "mean": float(md["mean"])},
            "fit": {
                "mean_error_before": float(fit["mean_error_before"]),
                "mean_error_after": float(fit["mean_error_after"]),
                "fallback_frames": int(fit["fallback_frames"]),
                "max_bone_delta_deg": float(fit["max_bone_delta_deg"]),
                "max_center_delta": float(fit["max_center_delta"]),
            },
        }
    return {
        "output": None,
        "input_kind": "bone",
        "keys": len(doc.bone),
        "frame_range": frame_range,
        "duration_sec": duration,
        "sections": sections,
        "preset": args.preset,
        "clean_strength": args.clean_strength,
        "denoise": args.denoise,
        "denoise_mode": args.denoise_mode,
        "foot_ik_stabilize": args.foot_ik_stabilize,
        "foot_slide_suppression": args.foot_slide_suppression,
        "curve_mode": args.curve_mode,
        "reduce": args.reduce,
        "bones": bones,
        "reduction": reduction_diag,
        "pose_denoise": pose,
    }


def main(argv=None):
    """CLI エントリポイント。終了コードを返す(§9: 0/1/2/3、中断 130)。"""
    # 人間向け標準エラーはロケール符号化(cp932 等)で表せない文字を含んでも UnicodeEncodeError で
    # プロセスを落とさない(規約 §10)。エラーハンドラを緩め、表せない文字は退避表記へ置換して出す。
    # argparse 使用法エラー・fail() の error 行・警告ループの警告行の人間向け stderr を一様に覆う
    # (機械モードの stdout はバイナリ + UTF-8 の別経路 cli_events なので影響しない)。
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    if argv is None:
        argv = sys.argv[1:]

    # 機械モード判定(§10)。解析前に argv で先取りする: 引数エラー時も出力チャネルを決めるため。
    # emitter はバイナリ stdout へ UTF-8 で書く(ロケール符号化非依存)。非機械では None(従来経路)。
    machine = "--machine" in argv
    emitter = EventEmitter(sys.stdout.buffer) if machine else None

    def fail(code, message, exit_code, *, field=None, path=None):
        """失敗を報告して終了コードを返す(§10.4)。機械モードは error イベントでストリームを終端し、
        それ以外は理由を標準エラーへ 1 行出す(トレースバックは出さない)。"""
        if emitter is not None:
            emitter.error(**error_event(
                code=code, message=message, exit_code=exit_code, field=field, path=path))
        else:
            print(f"error: {message}", file=sys.stderr)
        return exit_code

    parser = _build_parser(machine)
    try:
        args = parser.parse_args(argv)
    except ArgumentParseError as e:
        # 機械モードの MachineArgumentParser は使用法エラーで例外を送出する(SystemExit の代わり)。
        return fail("bad_argument", e.message, 2, field=argparse_error_field(e.message))
    except SystemExit as e:
        # 非機械の使用法エラー(argparse が stderr へ出力済み・code 2)と、両モードの --help/--version
        # (メタ操作・code 0)。例外を握って終了コードへ変換する(§10.1)。
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)

    # 引数解析後の本体を畳む。KeyboardInterrupt は中断(§10.5)として cancelled へ、それ以外の想定外例外は
    # internal_error(§10.4)へ。どちらもトレースバックを漏らさない。
    try:
        return _run(args, machine, emitter, fail)
    except KeyboardInterrupt:
        # 協調的な中断(Ctrl-C / 親プロセスの中断)。書き込みは全計算後に 1 回だけで原子的なので、ここに
        # 来た時点で出力は未書き込みか原子置換済みのいずれかで、中途半端な出力は残らない。
        return fail("cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return fail("internal_error", f"{type(e).__name__}: {e}", 1)


def _run(args, machine, emitter, fail):
    """引数解析済みの本体処理(検証 → クリーニング → 足IK安定化 → 疎化 → 書き込み)。

    失敗は fail() 経由で終了コードを返す。機械モードは emitter で progress / warning / result を送出する。
    """
    # 入力パス検証(§3.3)。不在・非通常ファイルは引数エラー。
    if not os.path.isfile(args.input):
        return fail("input_not_file", f"入力が存在しないか通常ファイルでない: {args.input}", 2, field="input")

    # --list-bones は書き込み・疎化をしない診断モード。出力先・上書きガードや疎化許容値・ボーン値検証
    # (処理・書き込み固有)を行わず、読み込んでボーン一覧と分類を出して終了する(§3.2 / §10.2)。
    if args.list_bones:
        try:
            doc, read_warnings = io.read(args.input)
        except Exception as e:
            return fail("not_vmd", f"入力を VMD として読めない: {type(e).__name__}: {e}", 1, field="input")
        _surface_warnings(read_warnings, machine, emitter)
        if machine:
            emitter.result(mode="list_bones", bones=[
                {"name": n, "category": classify.classify(n)} for n in _bone_order(doc.bone)
            ])
        else:
            print(_list_bones_text(doc.bone))
        return 0

    # 出力先・上書きガード(§3.2)。入力と同一パスへの出力は --overwrite が必要。
    output = args.output if args.output is not None else _default_output(args.input)
    if not args.overwrite and _same_path(output, args.input):
        return fail("output_overwrites_input",
                    f"出力先が入力と同一パス。上書きには --overwrite が必要: {output}",
                    2, field="--output")

    # 疎化の許容誤差 override は非有限・負を引数エラー(§10.4 bad_argument)とする(§3.2 / §5.3)。
    for name, v in (("--reduce-error-bone-pos", args.reduce_error_bone_pos),
                    ("--reduce-error-bone-rot", args.reduce_error_bone_rot)):
        if v is not None and (not math.isfinite(v) or v < 0.0):
            return fail("bad_argument", f"{name} は有限の非負値が必要: {v}", 2, field=name)

    # クリーニング強度の倍率は非有限・負を引数エラー(§5.2)。
    if not math.isfinite(args.clean_strength) or args.clean_strength < 0.0:
        return fail("bad_argument", f"--clean-strength は有限の非負値が必要: {args.clean_strength}",
                    2, field="--clean-strength")

    # 横滑り抑制は 0〜1 の有限値のみ許容(範囲外・非有限は引数エラー。§4.3 / §5.4)。
    s = args.foot_slide_suppression
    if not math.isfinite(s) or not 0.0 <= s <= 1.0:
        return fail("bad_argument", f"--foot-slide-suppression は 0〜1 の有限値が必要: {s}",
                    2, field="--foot-slide-suppression")

    # pose モードで --pmx 指定時は、パスの存在・通常ファイルを引数エラー(§10.4 pmx_not_file)で検証する。
    if args.denoise and args.denoise_mode == "pose" and args.pmx is not None:
        if not os.path.isfile(args.pmx):
            return fail("pmx_not_file", f"--pmx が存在しないか通常ファイルでない: {args.pmx}",
                        2, field="--pmx")

    # 入力読み込み(VMDでない等 → 入力不正)。
    try:
        doc, read_warnings = io.read(args.input)
    except Exception as e:
        return fail("not_vmd", f"入力を VMD として読めない: {type(e).__name__}: {e}", 1, field="input")
    _surface_warnings(read_warnings, machine, emitter)

    # 入力ボーン値の健全性(非有限・ゼロノルム quaternion)は処理経路(クリーニング/疎化の有無・dry-run か)
    # に依らず、パイプライン前に全キーを検証する(§3.3 入力不正)。dry-run でも疎化レポート(§4.4)のため
    # 疎化を実行するので、未検証の不正値が疎化へ流れて逐語透過・例外化するのを防ぐ。
    try:
        _validate_bones(doc.bone)
    except ValueError as e:
        return fail("invalid_bone_values",
                    f"ボーン値が不正(非有限・ゼロノルム quaternion): {e}", 1, field="input")

    # クリーニング → 足IK安定化 → 疎化のパイプライン。dry-run でも疎化レポート(§4.4)の素データを得るため
    # 実行し、出力の書き出しだけを dry-run で省く。進捗は機械モードで progress イベント(端末非依存)、
    # 非機械は端末時のライブ表示(--quiet で無効)。段ラベルは利用者向けの平易な文言にする(平滑化=
    # クリーニング、足IK最適化=足IK安定化、キーフレーム圧縮=疎化)。例外時もハートビートを止め行を消す
    # ため try/finally で囲む。
    reporter = progress.ProgressReporter(sys.stderr, enabled=False if (args.quiet or machine) else None)
    # dry-run / verbose のときだけ診断素データを集める(通常実行のオーバーヘッドを避ける)。
    want_report = args.dry_run or args.verbose
    reduction_diag = {} if (args.reduce and want_report) else None
    pose_diag = {} if (args.denoise and args.denoise_mode == "pose" and want_report) else None
    try:
        if args.denoise:
            if machine:
                emitter.progress(stage="denoise", done=0, total=None, note="", elapsed=0.0)
            else:
                reporter.stage("平滑化")
            if args.denoise_mode == "pose":
                # 表現空間ノイズ除去。PMX形式不正・モデルプロファイル不正は入力不正(§10.4)。
                try:
                    new_bone = apply_pose_denoise(doc.bone, pmx_path=args.pmx, diagnostics_out=pose_diag)
                except PmxFormatError as e:
                    return fail("not_pmx", f"PMX 形式が不正: {type(e).__name__}: {e}", 1, field="--pmx")
                except MocapModelProfileError as e:
                    field = "--pmx" if args.pmx is not None else None
                    return fail("model_profile_invalid", f"モデルプロファイルが不正: {e}", 1, field=field)
            else:
                new_bone = _clean_bones(doc.bone, args.clean_strength)
        else:
            new_bone = doc.bone
        if args.foot_ik_stabilize:
            if machine:
                emitter.progress(stage="foot_ik", done=0, total=None, note="", elapsed=0.0)
            else:
                reporter.stage("足IK最適化")
            new_bone = _stabilize_bones(new_bone, args.foot_slide_suppression)
        if args.reduce:
            if machine:
                reduce_start = time.monotonic()
                emitter.progress(stage="reduce", done=0, total=None, note="", elapsed=0.0)

                def reduce_cb(done, total):
                    emitter.progress(stage="reduce", done=done, total=total, note="",
                                     elapsed=time.monotonic() - reduce_start)
            else:
                reporter.stage("キーフレーム圧縮")
                reduce_cb = reporter.update
            new_bone = reduce.reduce_bones(
                new_bone,
                args.preset,
                override_pos=args.reduce_error_bone_pos,
                override_rot=args.reduce_error_bone_rot,
                curve_mode=args.curve_mode,
                diagnostics_out=reduction_diag,
                progress=reduce_cb,
            )
    finally:
        reporter.close()

    # 人間向け診断レポート(非機械の dry-run / verbose)。機械モードは stdout をイベント専用にするので
    # 人間向けレポートは出さない(入力検査は inspect result で返す)。
    if not machine and want_report:
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
        print(report.format_dry_run(rep))

    # dry-run は出力を書かずに終える。機械モードは入力検査(inspect)の result でストリームを終端する。
    if args.dry_run:
        if machine:
            emitter.result(mode="inspect", **_build_inspect(args, doc, reduction_diag, pose_diag))
        return 0

    out_doc = dataclasses.replace(doc, bone=new_bone)
    try:
        io.write_file(out_doc, output)
    except Exception as e:
        return fail("write_failed", f"出力の書き込みに失敗: {type(e).__name__}: {e}",
                    3, field="--output", path=output)

    # 書き込み成功後に終端イベント/完了行を 1 回出す(進捗行は close で消えている)。
    if machine:
        emitter.result(mode="process", output=output,
                       input_keys=len(doc.bone), output_keys=len(new_bone))
    else:
        # 進捗表示が有効だった(端末・非 quiet)ときだけ、消した進捗行のあとに完了行を 1 行残す。
        reporter.summary(f"完了 {output}")
    return 0
