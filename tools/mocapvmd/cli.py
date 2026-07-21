"""mocapvmd CLI。

引数解析 → VMD読み(vmd.io)→ 全ボーンの一般ノイズ軽減(クリーニング)→
足IK・つま先IKの接地安定化 → 共通機構による疎化 → VMD書き。ボーン選択は持たず、一般ノイズ軽減と
疎化は全ボーン、足IK安定化は分類 foot_ik / toe_ik のボーンに適用する。対象外セクション(モーフ・
カメラ・照明・セルフ影)は無加工で透過する。既定では疎なキーとベジェ補間を出力し、--no-reduce 時のみ
クリーニング後の密キー(線形補間)を出力する。

--machine 指定時は標準出力を JSON Lines のイベントストリーム(progress / warning / result / error)に
切り替える。既定(非機械)の人間向け表示・出力ファイル・終了コードは変えない。
--describe は VMD を読まずにオプション定義とプリセット一覧の result を出す独立メタ操作。

終了コード: 0 正常 / 1 入力不正(VMDでない・値が非有限・PMX形式不正・モデルプロファイル不正)/
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
    p.add_argument("--describe", action="store_true",
                   help="オプション定義とプリセット一覧を JSON Lines の result で出力して終了する"
                        "(VMD を読まない・入力不要の自己記述)")
    p.add_argument("input", nargs="?", help="入力VMDファイル")
    p.add_argument("-o", "--output", help="出力先(既定: <入力名>_mocap.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
    p.add_argument("--preset", choices=presets.PRESET_NAMES, default="medium",
                   help="キーフレーム圧縮の許容誤差プリセット(速度観点): slower/slow/medium/fast/faster。種別スケールの基準値")
    p.add_argument("--clean-strength", dest="clean_strength", type=float, default=1.0,
                   help="ノイズ軽減の効き量の倍率(ブレンド率に掛ける。0で無加工相当、上げるほど強く平準化する。窓幅は据え置き)")
    p.add_argument("--denoise", dest="denoise", action="store_true", default=True,
                   help="ノイズ軽減を有効化(既定 on)")
    p.add_argument("--no-denoise", dest="denoise", action="store_false",
                   help="ノイズ軽減を無効化")
    p.add_argument("--denoise-mode", dest="denoise_mode", choices=("bone", "pose"), default="bone",
                   help="ノイズ軽減の方式: bone(ボーン単位)/ pose(MMDで実際に見える動き=表現空間で評価)")
    p.add_argument("--pmx", dest="pmx", default=None,
                   help="pose 方式が参照するモデルPMX(未指定時は内蔵の既定モデルプロファイル)")
    p.add_argument("--foot-ik-stabilize", dest="foot_ik_stabilize", action="store_true", default=True,
                   help="足IK接地安定化を有効化(既定 on)")
    p.add_argument("--no-foot-ik-stabilize", dest="foot_ik_stabilize", action="store_false",
                   help="足IK接地安定化を無効化")
    p.add_argument("--foot-slide-suppression", dest="foot_slide_suppression", type=float, default=1.0,
                   help="足IK接地中の横滑り抑制の強さ(0〜1)。既定1.0は接地中の横滑りを除去する")
    p.add_argument("--reduce-error-bone-pos", dest="reduce_error_bone_pos", type=float, default=None,
                   help="キーフレーム圧縮の位置許容誤差の基準値(MMD単位)。明示値はプリセットに優先。種別スケールを掛ける")
    p.add_argument("--reduce-error-bone-rot", dest="reduce_error_bone_rot", type=float, default=None,
                   help="キーフレーム圧縮の回転許容誤差の基準値(度)。明示値はプリセットに優先。種別スケールを掛ける")
    p.add_argument("--curve-mode", dest="curve_mode", choices=("bezier", "linear"), default="bezier",
                   help="出力補間曲線: bezier / linear")
    p.add_argument("--reduce", dest="reduce", action="store_true", default=True,
                   help="キーフレーム圧縮を有効化(既定 on)。少ないキー+ベジェ補間で出力する")
    p.add_argument("--no-reduce", dest="reduce", action="store_false",
                   help="キーフレームを圧縮せず、圧縮前の密キー(線形補間)を出力する(診断・比較用)")
    p.add_argument("--list-bones", dest="list_bones", action="store_true",
                   help="ボーン一覧と分類結果を表示して終了する")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力せず処理計画と診断を表示する(引数検証は実施する)")
    p.add_argument("--quiet", dest="quiet", action="store_true",
                   help="進捗表示を抑制する(警告・診断・終了コードは抑制しない)")
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                   help="通常実行でも 4.4 のレポートを標準出力へ表示する(出力VMDは書く)")
    return p


# --describe の型/制約表。dest → (type, constraint)。help/default は parser の各 action から引く。
# メタ/モード操作(describe/version/help/machine)は _D_TYPE に無いので describe の options から除外される。
# type 関数と1対1で対応するので制約の形は明示表で持つ。順序は parser の add_argument 順に従う。
_D_NONNEG = {"min": 0, "max": None, "exclusive_min": False}
_D_01 = {"min": 0, "max": 1, "exclusive_min": False}
_D_TYPE = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "preset": ("enum", {"choices": list(presets.PRESET_NAMES)}),
    "clean_strength": ("float", _D_NONNEG),
    "denoise": ("flag", None),
    "denoise_mode": ("enum", {"choices": ["bone", "pose"]}),
    "pmx": ("str", None),
    "foot_ik_stabilize": ("flag", None),
    "foot_slide_suppression": ("float", _D_01),
    "reduce_error_bone_pos": ("float", _D_NONNEG),
    "reduce_error_bone_rot": ("float", _D_NONNEG),
    "curve_mode": ("enum", {"choices": ["bezier", "linear"]}),
    "reduce": ("flag", None),
    "list_bones": ("flag", None),
    "dry_run": ("flag", None),
    "quiet": ("flag", None),
    "verbose": ("flag", None),
}


def _describe_options(parser):
    """--describe の options を parser 定義から機械導出する。順序は add_argument 順。

    メタ/モード操作(--describe/--version/--help/--machine)は _D_TYPE に無いので除外される。真偽フラグの
    否定形(--no-* だけの action)は肯定形の長形式で既に載るのでスキップする(重複列挙しない)。
    type/constraint は _D_TYPE、help/default は各 action から引く。
    """
    options = []
    for action in parser._actions:
        dest = action.dest
        if dest not in _D_TYPE:
            continue
        type_, constraint = _D_TYPE[dest]
        if dest == "input":
            name = "input"
        else:
            # 肯定形の長形式を採る。--no-* だけの否定形 action はスキップ(肯定形で既に載る)。
            pos = [s for s in action.option_strings if s.startswith("--") and not s.startswith("--no-")]
            if not pos:
                continue
            name = pos[0]
        options.append({
            "name": name,
            "type": type_,
            "constraint": constraint,
            "default": action.default,
            "help": action.help,
        })
    return options


def _describe_presets():
    """--describe の presets を presets モジュールから導出する。

    各要素は {name, values}。values は基準位置許容・基準回転許容のみ(種別スケール・クリーニング基準・
    接地ロック係数は CLI 非公開の内蔵パラメータなので載せない)。
    """
    out = []
    for name in presets.PRESET_NAMES:
        pos, rot = presets.reduction_base(name)
        out.append({"name": name,
                    "values": {"reduce_error_bone_pos": pos, "reduce_error_bone_rot": rot}})
    return out


def _default_output(input_path):
    base, _ = os.path.splitext(input_path)
    return base + "_mocap.vmd"


def _surface_warnings(read_warnings, machine, emitter):
    """読み込み警告(デコード不能な名前フィールド等)を surface する。

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
            print(f"warning: {w.code}: {w.message}{where}", file=sys.stderr)


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
    """各ボーンの名前と分類を初出順・名前ごとに1行で返す(--list-bones)。"""
    return "\n".join(f"{name} [{classify.classify(name)}]" for name in _bone_order(bone_keys))


def _validate_bones(bone_keys):
    """全ボーンキーの値の健全性(非有限・ゼロノルム quaternion)を検証する(入力不正検出)。

    クリーニング/疎化の有無に依らず入力不正を弾くため、パイプライン前に全キーの位置・回転を検証する。
    不正があれば ValueError を送出する。
    """
    denoise.validate_bone_values([k.position for k in bone_keys], [k.rotation for k in bone_keys])


def _clean_bones(bone_keys, clean_strength):
    """全ボーンを種別別パラメータで一般ノイズ軽減し、密キー(線形補間)で返す。

    各トラックを名前ごとに時系列順へまとめ、クリーニング後の密サンプルを線形補間キーとして組み直す
    (クリーニング後の密キー形式)。クリーニング強度の倍率 clean_strength を種別別の基準ブレンド率へ
    掛ける。キー1個以下のトラックは平滑化できないため逐語保持する。値の健全性は呼び出し前に
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
    """分類 foot_ik / toe_ik の密トラックに接地安定化を適用し、密キー(線形補間)で返す。

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
    """--machine --dry-run の inspect result ペイロードを組む。VMD は書かない。

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
    """CLI エントリポイント。終了コードを返す(0/1/2/3、中断 130)。"""
    # 人間向け標準エラーはロケール符号化(cp932 等)で表せない文字を含んでも UnicodeEncodeError で
    # プロセスを落とさない。エラーハンドラを緩め、表せない文字は退避表記へ置換して出す。
    # argparse 使用法エラー・fail() の error 行・警告ループの警告行の人間向け stderr を一様に覆う
    # (機械モードの stdout はバイナリ + UTF-8 の別経路 cli_events なので影響しない)。
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    if argv is None:
        argv = sys.argv[1:]

    # 構造化出力モード判定。解析前に argv で先取りする: 引数エラー時も出力チャネルを決めるため。
    # --describe は --machine を要さない独立メタ操作。どちらかがあれば emitter を用意し、
    # MachineArgumentParser で使用法エラーも error イベントへ振り替える。emitter はバイナリ stdout へ
    # UTF-8 で書く(ロケール符号化非依存)。どちらも無ければ None(従来の人間向け経路)。
    machine = "--machine" in argv
    describe = "--describe" in argv
    emitter = EventEmitter(sys.stdout.buffer) if (machine or describe) else None

    def fail(code, message, exit_code, *, field=None, path=None):
        """失敗を報告して終了コードを返す。構造化出力モードは error イベントでストリームを終端し、
        それ以外は理由を標準エラーへ 1 行出す(トレースバックは出さない)。"""
        if emitter is not None:
            emitter.error(**error_event(
                code=code, message=message, exit_code=exit_code, field=field, path=path))
        else:
            print(f"error: {message}", file=sys.stderr)
        return exit_code

    parser = _build_parser(machine or describe)
    try:
        args = parser.parse_args(argv)
    except ArgumentParseError as e:
        # 構造化出力モードの MachineArgumentParser は使用法エラーで例外を送出する(SystemExit の代わり)。
        return fail("bad_argument", e.message, 2, field=argparse_error_field(e.message))
    except SystemExit as e:
        # 非機械の使用法エラー(argparse が stderr へ出力済み・code 2)と、両モードの --help/--version
        # (メタ操作・code 0)。例外を握って終了コードへ変換する。
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)

    # 自己記述。VMD を読まず options/presets の result を出して終了する独立メタ操作。
    if args.describe:
        emitter.result(mode="describe", options=_describe_options(parser), presets=_describe_presets())
        return 0
    # input は nargs="?"(--describe を入力無しで成立させるため)。非 describe 実行では必須。
    if args.input is None:
        return fail("bad_argument", "入力VMDファイル(input)が必要", 2, field="input")

    # 引数解析後の本体を畳む。KeyboardInterrupt は中断として cancelled へ、それ以外の想定外例外は
    # internal_error へ。どちらもトレースバックを漏らさない。
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
    # 入力パス検証。不在・非通常ファイルは入力不正。
    if not os.path.isfile(args.input):
        return fail("input_not_file", f"入力が存在しないか通常ファイルでない: {args.input}", 1, field="input")

    # --list-bones は書き込み・疎化をしない診断モード。出力先・上書きガードや疎化許容値・ボーン値検証
    # (処理・書き込み固有)を行わず、読み込んでボーン一覧と分類を出して終了する。
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

    # 出力先・上書きガード。出力先に既存ファイルがあれば --overwrite が必要。
    output = args.output if args.output is not None else _default_output(args.input)
    if not args.overwrite and os.path.exists(output):
        return fail("output_exists",
                    f"出力先に既存ファイルがあります。上書きには --overwrite が必要: {output}",
                    2, field="--output")

    # 疎化の許容誤差 override は非有限・負を引数エラー(bad_argument)とする。
    for name, v in (("--reduce-error-bone-pos", args.reduce_error_bone_pos),
                    ("--reduce-error-bone-rot", args.reduce_error_bone_rot)):
        if v is not None and (not math.isfinite(v) or v < 0.0):
            return fail("bad_argument", f"{name} は有限の非負値が必要: {v}", 2, field=name)

    # クリーニング強度の倍率は非有限・負を引数エラー。
    if not math.isfinite(args.clean_strength) or args.clean_strength < 0.0:
        return fail("bad_argument", f"--clean-strength は有限の非負値が必要: {args.clean_strength}",
                    2, field="--clean-strength")

    # 横滑り抑制は 0〜1 の有限値のみ許容(範囲外・非有限は引数エラー)。
    s = args.foot_slide_suppression
    if not math.isfinite(s) or not 0.0 <= s <= 1.0:
        return fail("bad_argument", f"--foot-slide-suppression は 0〜1 の有限値が必要: {s}",
                    2, field="--foot-slide-suppression")

    # pose モードで --pmx 指定時は、パスの存在・通常ファイルを入力不正(pmx_not_file)で検証する。
    if args.denoise and args.denoise_mode == "pose" and args.pmx is not None:
        if not os.path.isfile(args.pmx):
            return fail("pmx_not_file", f"--pmx が存在しないか通常ファイルでない: {args.pmx}",
                        1, field="--pmx")

    # 入力読み込み(VMDでない等 → 入力不正)。
    try:
        doc, read_warnings = io.read(args.input)
    except Exception as e:
        return fail("not_vmd", f"入力を VMD として読めない: {type(e).__name__}: {e}", 1, field="input")
    _surface_warnings(read_warnings, machine, emitter)

    # 入力ボーン値の健全性(非有限・ゼロノルム quaternion)は処理経路(クリーニング/疎化の有無・dry-run か)
    # に依らず、パイプライン前に全キーを検証する。dry-run でも疎化レポートのため
    # 疎化を実行するので、未検証の不正値が疎化へ流れて逐語透過・例外化するのを防ぐ。
    try:
        _validate_bones(doc.bone)
    except ValueError as e:
        return fail("invalid_bone_values",
                    f"ボーン値が不正(非有限・ゼロノルム quaternion): {e}", 1, field="input")

    # クリーニング → 足IK安定化 → 疎化のパイプライン。dry-run でも疎化レポートの素データを得るため
    # 実行し、出力の書き出しだけを dry-run で省く。進捗は機械モードで progress イベント(端末非依存)、
    # 非機械は端末時のライブ表示(--quiet で無効)。段ラベルは利用者向けの工程名を使う。
    # 例外時もハートビートを止め行を消すため try/finally で囲む。
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
                reporter.stage("ノイズ軽減")
            if args.denoise_mode == "pose":
                # 表現空間ノイズ除去。PMX形式不正・モデルプロファイル不正は入力不正。
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
                reporter.stage("足IK接地安定化")
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
