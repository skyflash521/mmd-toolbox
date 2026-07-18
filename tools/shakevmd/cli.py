"""shakevmd CLI。

コアの薄いラッパー: 引数解析 → VMD読み(vmd.io)→ bake() → VMD書き。
詳細パラメーター(オクターブ構成・persistence・プロファイル・手動カット等)は公開しない。

終了コード: 0 正常 / 1 入力不正(欠落・非VMD・カメラキーなし)/
2 引数エラー(範囲不正・逆順・重複・上書きガード・未知オプション等)/ 3 出力書き込み失敗。
"""

import argparse
import math
import os
import sys
import time

from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    error_event,
)
from vmd import interp, io
from vmd.reduce import Tolerances, reduce_camera_track
from shakevmd import __version__, cuts, presets, progress
from shakevmd.bake import RangeOverlapError, bake
from shakevmd.warn import ShakeWarning

# 公開引数の hard-default。プリセット/個別引数が未指定の項目に使う。
# fade はプリセット対象外(プリセットは7引数)だが、None センチネル解決のため hard-default を持つ。
_HARD_DEFAULTS = {
    "amp_rot": 0.8, "amp_pos": 0.05, "rot_weights": (1.0, 1.0, 0.3),
    "freq": 1.2, "motion_damp": 1.0, "settle": 0.0, "cut_threshold": (5.0, 20.0),
    "fade": 0.7,
}

# --smooth でベイク後にプロセス内で疎ベジェへ削減する固定設定。
# 許容はカメラ4項目の aggressive 値。bone はこの経路で不参照のダミー値。
_SMOOTH_TOLERANCES = Tolerances(
    bone_pos=1.0, bone_rot=30.0,
    camera_pos=0.10, camera_rot=0.25, camera_distance=0.10, camera_fov=1.00,
)
# reduce へ渡すカット距離閾値。検出は no_cut_detect=True で無効化するが cut_thresholds は
# 必須引数で 3 要素を要するため、pos と同程度の値を添える。
_SMOOTH_CUT_DIST = 5.0
_SMOOTH_MAX_SEG = 5


def _finite_float(text):
    """有限な float に変換する。inf/nan は引数エラー(下流の OverflowError 等を防ぐ)。"""
    v = float(text)  # 非数値は ValueError → argparse が exit 2 にする
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"有限な数値が必要: {text!r}")
    return v


def _nonneg_float(text):
    """有限かつ非負(≥0)の float。振幅・秒数・係数など負が無意味な量に使う。"""
    v = _finite_float(text)
    if v < 0:
        raise argparse.ArgumentTypeError(f"非負の数値が必要: {text!r}")
    return v


def _positive_float(text):
    """有限かつ正(>0)の float。周波数など 0/負が無意味な量に使う。"""
    v = _finite_float(text)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"正の数値が必要: {text!r}")
    return v


def _parse_range(text):
    """`START:END` を (start|None, end|None) に解析する。

    各辺は省略可(空=先頭/末尾)。両方省略(`:`)は全範囲。整数のみ。`START>END` は不正。
    端のスナップ・重複検出は bake() が行うため、ここでは書式と逆順のみ検証する。
    """
    if text.count(":") != 1:
        raise argparse.ArgumentTypeError(f"範囲は START:END 形式: {text!r}")
    s_str, e_str = text.split(":")

    def part(v):
        if v == "":
            return None
        try:
            n = int(v)
        except ValueError:
            raise argparse.ArgumentTypeError(f"範囲端は整数: {v!r}")
        if n < 0:
            raise argparse.ArgumentTypeError(f"フレーム番号は非負: {v!r}")
        return n

    s, e = part(s_str), part(e_str)
    if s is not None and e is not None and s > e:
        raise argparse.ArgumentTypeError(f"範囲は START<=END: {text!r}")
    return (s, e)


def _parse_rot_weights(text):
    """`P,Y,R` を (float, float, float) に解析する。"""
    parts = text.split(",")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--rot-weights は P,Y,R の3要素: {text!r}")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--rot-weights は数値3要素: {text!r}")
    if not all(math.isfinite(v) for v in vals):
        raise argparse.ArgumentTypeError(f"--rot-weights は有限値: {text!r}")
    return vals


def _parse_cut_threshold(text):
    """`位置,角度` を (float, float) に解析する。"""
    parts = text.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"--cut-threshold は 位置,角度 の2要素: {text!r}")
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--cut-threshold は数値2要素: {text!r}")
    if not all(math.isfinite(v) for v in vals):
        raise argparse.ArgumentTypeError(f"--cut-threshold は有限値: {text!r}")
    if any(v < 0 for v in vals):
        raise argparse.ArgumentTypeError(f"--cut-threshold は非負(感度閾値): {text!r}")
    return vals


def _parse_impulse(text):
    """`F:S:D` を (int, float, float) に解析する。F=フレーム, S=強さ度, D=減衰秒。"""
    parts = text.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(f"--impulse は F:S:D の3要素: {text!r}")
    f_str, s_str, d_str = parts
    try:
        f, s, d = int(f_str), float(s_str), float(d_str)
    except ValueError:
        raise argparse.ArgumentTypeError(f"--impulse は F:S:D(F=整数, S/D=数値): {text!r}")
    if f < 0:
        raise argparse.ArgumentTypeError(f"--impulse の F(フレーム)は非負: {text!r}")
    if not (math.isfinite(s) and math.isfinite(d)):
        raise argparse.ArgumentTypeError(f"--impulse の S/D は有限値: {text!r}")
    if s < 0:
        raise argparse.ArgumentTypeError(f"--impulse の S(強さ)は非負: {text!r}")
    if d <= 0:
        raise argparse.ArgumentTypeError(f"--impulse の D(減衰秒)は正: {text!r}")
    return (f, s, d)


def _build_parser(machine: bool = False) -> argparse.ArgumentParser:
    # allow_abbrev=False: 仕様外の前置き省略形(--over→--overwrite 等)を受理しない。
    # 未知/省略形は exit 2(非公開・繰延フラグ拒否とも整合)。
    # 機械モードは使用法エラーを error イベントへ振り替えるため、SystemExit の代わりに
    # ArgumentParseError を送出する MachineArgumentParser を使う(--help/--version は error() を
    # 経由しないので影響を受けず、従来どおり SystemExit で短絡する)。
    cls = MachineArgumentParser if machine else argparse.ArgumentParser
    p = cls(prog="shakevmd", allow_abbrev=False)
    p.add_argument("--version", action="version", version=f"shakevmd {__version__}",
                   help="バージョンを表示して終了する")
    # 機械モード。出力を JSON Lines のイベントストリームにし、stdout をイベント専用へ固定する。
    p.add_argument("--machine", action="store_true",
                   help="出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・"
                        "標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない")
    # 自己記述。VMD を読まず入力も要求しない独立メタ操作。input を nargs="?" にして
    # `shakevmd --describe` 単独で成立させ、非 describe 実行では main() が input の欠落を検査する。
    p.add_argument("--describe", action="store_true",
                   help="オプション定義とプリセット一覧を JSON Lines の result で出力して終了する"
                        "(VMD を読まない・入力不要の自己記述)")
    p.add_argument("input", nargs="?", help="入力カメラ VMD ファイル")
    p.add_argument("-o", "--output", help="出力先(既定: <入力名>_shake.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
    p.add_argument("--range", dest="ranges", action="append", type=_parse_range,
                   metavar="START:END",
                   help="揺れ適用範囲。START/END は各々省略可。複数指定可(既定は全範囲)")
    # 公開揺れパラメーターは default=None(未指定センチネル)。--preset と個別引数の優先を
    # main() で解決する(明示 > preset > hard-default)。型検証は明示値にのみ適用される。
    p.add_argument("--amp-rot", type=_nonneg_float, help="回転振幅の基準値(度)")        # 振幅 ≥0
    p.add_argument("--amp-pos", type=_nonneg_float, help="位置振幅(MMD距離単位)")       # 振幅 ≥0
    p.add_argument("--rot-weights", type=_parse_rot_weights, metavar="P,Y,R",
                   help="Pitch/Yaw/Roll 個別重み")
    p.add_argument("--freq", type=_positive_float, help="ノイズ基本周波数(Hz)")         # 周波数 >0
    p.add_argument("--seed", type=int, default=1, help="ノイズシード(既定 1・再現性)")
    p.add_argument("--fade", type=_nonneg_float, help="範囲端の自動フェード時間(秒)")   # 秒 ≥0
    p.add_argument("--motion-damp", type=_nonneg_float,
                   help="元モーション速度に応じた振幅減衰係数(0 で無効)")               # 係数 ≥0
    p.add_argument("--settle", type=_nonneg_float,
                   help="停止検出時の減衰振動の初期振幅(度・0 で無効)")                 # 度 ≥0
    p.add_argument("--cut-threshold", type=_parse_cut_threshold, metavar="位置,角度",
                   help="カット自動検出の感度(位置ジャンプ,角度ジャンプ)")
    p.add_argument("--impulse", dest="impulses", action="append", type=_parse_impulse,
                   metavar="F:S:D",
                   help="フレーム F に強さ S・減衰 D 秒の衝撃を加算(複数指定可)")
    # 運用/プリセット系。
    p.add_argument("--preset", choices=presets.PRESET_NAMES,   # 未知名は argparse が exit 2
                   help="公開引数を一括設定するプリセット(個別引数の明示指定が優先)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力せず統計を表示する(引数検証は実施する)")
    p.add_argument("-v", "--verbose", action="store_true", help="詳細ログを出す")
    # 既定 on: ベイク後にプロセス内で疎ベジェへ削減し、30fps 超再生のカクつきを低減する。
    # --no-smooth で無効化(密キー＋線形のまま出力する)。
    p.add_argument("--smooth", default=True, action=argparse.BooleanOptionalAction,
                   help="ベイク後の密なキーをベジェ補間でなめらかに整理する(スムージング。既定 on。--no-smooth で密キー+線形)")
    # 進捗表示の抑制。抑制するのは進捗表示だけで、警告・統計・終了コードは変えない。
    p.add_argument("--quiet", dest="quiet", action="store_true",
                   help="進捗表示を抑制する(警告・統計・終了コードは抑制しない)")
    return p


# --describe の型/制約表。dest → (type, constraint)。help と default は parser・定数から引く。
# type 関数と1対1で対応するので制約の形は明示表で持ち、cli.py の引数定義と乖離しないよう順序は
# parser の add_argument 順に従う(_describe_options が parser を走査する)。
_D_NONNEG = {"min": 0, "max": None, "exclusive_min": False}     # _nonneg_float
_D_POSITIVE = {"min": 0, "max": None, "exclusive_min": True}    # _positive_float


def _cfield(name, type_, mn, mx, ex):
    return {"name": name, "type": type_, "min": mn, "max": mx, "exclusive_min": ex}


_D_COMPOUND = {
    "ranges": {"format": "START:END", "fields": [
        _cfield("START", "int", 0, None, False), _cfield("END", "int", 0, None, False)]},
    "rot_weights": {"format": "P,Y,R", "fields": [
        _cfield("P", "float", None, None, False), _cfield("Y", "float", None, None, False),
        _cfield("R", "float", None, None, False)]},
    "cut_threshold": {"format": "位置,角度", "fields": [
        _cfield("位置", "float", 0, None, False), _cfield("角度", "float", 0, None, False)]},
    "impulses": {"format": "F:S:D", "fields": [
        _cfield("F", "int", 0, None, False), _cfield("S", "float", 0, None, False),
        _cfield("D", "float", 0, None, True)]},
}

_D_TYPE = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "ranges": ("compound", _D_COMPOUND["ranges"]),
    "amp_rot": ("float", _D_NONNEG),
    "amp_pos": ("float", _D_NONNEG),
    "rot_weights": ("compound", _D_COMPOUND["rot_weights"]),
    "freq": ("float", _D_POSITIVE),
    "seed": ("int", None),          # 範囲制約の無い裸の int は constraint:null
    "fade": ("float", _D_NONNEG),
    "motion_damp": ("float", _D_NONNEG),
    "settle": ("float", _D_NONNEG),
    "cut_threshold": ("compound", _D_COMPOUND["cut_threshold"]),
    "impulses": ("compound", _D_COMPOUND["impulses"]),
    "preset": ("enum", {"choices": list(presets.PRESET_NAMES)}),
    "dry_run": ("flag", None),
    "verbose": ("flag", None),
    "smooth": ("flag", None),
    "quiet": ("flag", None),
}


def _describe_options(parser):
    """--describe の options を parser 定義から機械導出する。順序は add_argument 順。

    メタ/モード操作(--describe/--version/--help/--machine)は _D_TYPE に無いので除外される。
    type/constraint は _D_TYPE(型関数と対応)、help は各 action、default は揺れパラメーターのみ
    未指定センチネルを解決後の hard-default に置き換え、それ以外は action の既定をそのまま出す。
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
            # BooleanOptionalAction は ["--smooth","--no-smooth"] を持つので否定形を除いた長形式を採る。
            name = next(s for s in action.option_strings
                        if s.startswith("--") and not s.startswith("--no-"))
        if dest in _HARD_DEFAULTS:
            v = _HARD_DEFAULTS[dest]
            default = list(v) if isinstance(v, tuple) else v
        else:
            default = action.default
        options.append({
            "name": name, "type": type_, "constraint": constraint,
            "default": default, "help": action.help,
        })
    return options


def _describe_presets():
    """--describe の presets を presets モジュールから導出する。内蔵パラメーターは除外する。"""
    out = []
    for name in presets.PRESET_NAMES:
        values = {
            k: (list(v) if isinstance(v, tuple) else v)
            for k, v in presets.get_preset(name).items()
            if k not in presets.INTERNAL_PARAM_NAMES
        }
        out.append({"name": name, "values": values})
    return out


def _default_output(input_path: str) -> str:
    # 既定出力は `<入力名(拡張子なし)>_shake.vmd`。元の拡張子に依らず常に .vmd。
    base, _ = os.path.splitext(input_path)
    return base + "_shake.vmd"


def _all_finite(keys) -> bool:
    """全カメラキーの数値(距離・位置・回転)が有限か。

    有限な引数でも bake 内の乗算(例 --amp-pos 1e308 を減衰なしで焼く)で inf に
    なりうる。float32 は inf を例外なく pack するため、書き出し前にここで検査する。
    """
    for k in keys:
        if not math.isfinite(k.distance):
            return False
        if not all(math.isfinite(c) for c in k.position):
            return False
        if not all(math.isfinite(r) for r in k.rotation):
            return False
    return True


def _working_view(camera):
    """正規化作業ビュー: フレームでソートし重複は後勝ち。bake の正規化と同じ規則。"""
    by_frame = {}
    for k in camera:
        by_frame[k.frame] = k          # 同一フレームは後勝ち
    return [by_frame[f] for f in sorted(by_frame)]


def _snap(x, frames):
    """x を最近接の既存キーフレームへスナップ(同距離は小さい方)。bake の _snap と同規則。"""
    return min(frames, key=lambda f: (abs(f - x), f))


def _resolve_param(name, args, preset):
    """公開引数を 明示 > preset > hard-default の優先で解決する。"""
    v = getattr(args, name)
    if v is not None:
        return v
    if name in preset:
        return preset[name]
    return _HARD_DEFAULTS[name]


def _shake_stats(orig_camera, baked_keys, applied, cut_pos, cut_rot):
    """dry-run/verbose 用の統計を算出する。

    戻り値: (max_amp, detected_cuts)。
    最大振幅は ベイク値 − 元サンプリング(適用範囲内の各ベイクフレーム) から算出する。
    bake は不変のまま、出力と cuts/interp から算出する。
    """
    wv = _working_view(orig_camera)
    detected_cuts = cuts.detect_cuts(wv, cut_pos, cut_rot)
    applied_frames = set()
    for a, b in applied:
        applied_frames.update(range(a, b + 1))
    max_amp = 0.0
    for k in sorted(baked_keys, key=lambda x: x.frame):
        if k.frame not in applied_frames:
            continue
        s = interp.sample_camera(wv, k.frame)
        dr = [k.rotation[i] - s["rotation"][i] for i in range(3)]
        dp = [k.position[i] - s["position"][i] for i in range(3)]
        for v in dr + dp:
            max_amp = max(max_amp, abs(v))
    return max_amp, detected_cuts


def _argparse_field(message: str):
    """argparse の使用法エラーメッセージから対象引数名を取り出す(bad_argument の field)。

    argparse は起因引数を構造化して渡さないので、標準の文言形からベストエフォートで抽出する。
    文言に依存するため未知の形は None(field なし)へ退避し、詳細は message 側に残す。
    """
    if message.startswith("argument "):
        name = message[len("argument "):].split(":", 1)[0].strip()
        # 複数のオプション文字列は "-o/--output" のように連結される。長形式(最後)を採る。
        return name.split("/")[-1] if name.startswith("-") else name
    if message.startswith("unrecognized arguments:"):
        rest = message[len("unrecognized arguments:"):].split()
        return rest[0] if rest else None
    if message.startswith("the following arguments are required:"):
        rest = message[len("the following arguments are required:"):].strip()
        return rest.split(",")[0].strip() or None
    return None


def main(argv=None) -> int:
    """CLI エントリポイント。終了コードを返す(0/1/2/3、中断 130)。"""
    # 人間向け標準エラーはロケール符号化(cp932 等)で表せない文字を含んでも UnicodeEncodeError で
    # プロセスを落とさない。エラーハンドラを緩め、表せない文字は退避表記へ置換して出す。
    # argparse の使用法エラー・fail() の error 行・警告ループの warning 行の人間向け stderr を一様に覆う
    # (機械モードの stdout はバイナリ + UTF-8 の別経路 cli_events なので影響しない)。
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    if argv is None:
        argv = sys.argv[1:]

    # 機械モード判定。解析前に argv で先取りする: 引数エラー時も出力チャネルを決めるため。
    # emitter はバイナリ stdout へ UTF-8 で書く(ロケール符号化非依存)。--describe は --machine を
    # 要さず構造化出力を起動するので、describe でも emitter を用意する。両方無ければ None(従来経路)。
    machine = "--machine" in argv
    describe = "--describe" in argv
    emitter = EventEmitter(sys.stdout.buffer) if (machine or describe) else None

    def fail(code, message, exit_code, *, field=None, path=None):
        """失敗を報告して終了コードを返す。構造化出力モード(機械モード・自己記述)は error
        イベントでストリームを終端し、それ以外は理由を標準エラーへ1行出す(トレースバックは出さない)。
        emitter の有無(= machine or describe)で分岐する。"""
        if emitter is not None:
            emitter.error(**error_event(
                code=code, message=message, exit_code=exit_code, field=field, path=path))
        else:
            print(f"error: {message}", file=sys.stderr)
        return exit_code

    # 構造化出力モード(--machine・--describe)は使用法エラーも error イベントへ振り替えるため
    # MachineArgumentParser を使う。どちらでもなければ従来の argparse(SystemExit→終了コード)。
    parser = _build_parser(machine or describe)
    try:
        args = parser.parse_args(argv)
    except ArgumentParseError as e:
        # 機械モードの MachineArgumentParser は使用法エラーで例外を送出する(SystemExit の代わり)。
        return fail("bad_argument", e.message, 2, field=_argparse_field(e.message))
    except SystemExit as e:
        # 非機械の使用法エラー(argparse が stderr へ出力済み・code 2)と、両モードの --help/--version
        # (メタ操作・code 0)。例外を握って終了コードへ変換する。
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)

    # 自己記述。VMD を読まず options/presets の result を出して終了する独立メタ操作。
    if args.describe:
        emitter.result(
            mode="describe",
            options=_describe_options(parser),
            presets=_describe_presets(),
        )
        return 0
    # input は nargs="?"(--describe を入力無しで成立させるため)。非 describe 実行では必須。
    if args.input is None:
        return fail("bad_argument", "入力カメラ VMD ファイル(input)が必要", 2, field="input")

    # 引数解析後の本体を畳む。KeyboardInterrupt(BaseException)は中断として cancelled へ、
    # それ以外の想定外例外は internal_error へ。どちらもトレースバックは漏らさない。
    try:
        return _run(args, machine, emitter, fail)
    except KeyboardInterrupt:
        # 協調的な中断(Ctrl-C / 親プロセスの中断)。書き込みは全計算後に1回だけで原子的なので、
        # ここに来た時点で出力は未書き込みか原子置換済みのいずれかで、中途半端な出力は残らない。
        return fail("cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return fail("internal_error", f"{type(e).__name__}: {e}", 1)


def _run(args, machine, emitter, fail) -> int:
    """引数解析済みの本体処理(ベイク→スムージング→書き込み)。失敗は fail() 経由で終了コードを返す。"""
    output = args.output if args.output is not None else _default_output(args.input)

    # 上書きガード: 出力先に既存ファイルがあれば --overwrite 必須。未許可なら書かずにエラー。
    if not args.overwrite and os.path.exists(output):
        return fail(
            "output_exists",
            f"出力先に既存ファイルがあります。上書きには --overwrite が必要: {output}",
            2, field="--output",
        )

    # 入力読み込み(欠落・非VMD・カメラキーなし → 入力不正でコード1)。
    # io.read は継続可能な問題(名前のデコード不可・トレーリングデータ等)を警告で返す。
    # VMD I/O は vmd へ委譲する設計なので、その警告もユーザーへ伝播する。
    try:
        doc, read_warnings = io.read(args.input)
    except Exception as e:
        return fail("not_vmd", f"入力を VMD として読めない: {type(e).__name__}: {e}",
                    1, field="input")
    if not doc.camera:
        return fail("no_camera_keys", "入力にカメラキーがない", 1, field="input")

    # 範囲の省略側を先頭/末尾キーへ解決(端のスナップ・重複検出は bake() が担う)。
    ranges = None
    if args.ranges:
        frames = [k.frame for k in doc.camera]
        first, last = min(frames), max(frames)
        ranges = [
            (first if s is None else s, last if e is None else e)
            for (s, e) in args.ranges
        ]
        # 省略端を先頭/末尾へ解決した「後」にも逆順を検査する(START>END は引数エラー)。
        # 例: `999:` は END=末尾60 に解決され 999>60。bake は端をスナップ後に swap するため
        # ここで弾かないと [60,60] として黙って焼かれてしまう。
        if any(s > e for (s, e) in ranges):
            return fail("range_reversed",
                        "範囲の開始が終了より後(省略端の解決後に START>END)", 2, field="--range")

    # 公開揺れパラメーターを解決(明示 > --preset > hard-default)。
    preset = presets.get_preset(args.preset) if args.preset else {}
    amp_rot = _resolve_param("amp_rot", args, preset)
    amp_pos = _resolve_param("amp_pos", args, preset)
    rot_weights = _resolve_param("rot_weights", args, preset)
    freq = _resolve_param("freq", args, preset)
    motion_damp = _resolve_param("motion_damp", args, preset)
    settle = _resolve_param("settle", args, preset)
    cut_threshold = _resolve_param("cut_threshold", args, preset)
    fade = _resolve_param("fade", args, preset)
    # 内蔵パラメーター(CLI 非公開、プリセットのみ)を bake へ転送する。
    # 例: walking の歩調成分 gait_freq/gait_amp。bake 既定(無効)を上書きする。
    internal = {k: preset[k] for k in presets.INTERNAL_PARAM_NAMES if k in preset}

    # 進捗のライブ表示。重いベイク・スムージングの進行を端末へ出す(--quiet で無効、既定は
    # stderr が端末のときだけ)。機械モードでは進捗をイベントで出すのでライブ行は無効化する。副作用専用=
    # 出力VMD・終了コード・統計・警告を変えない。最初の stage 以降は捕捉例外で終了コードを返す経路・
    # dry-run の早期 return・想定外例外のいずれでも heartbeat を止め行を消すため try/finally で囲む。
    # close は二重呼び出しに耐えるので明示 close と finally が重なって安全。
    reporter = progress.ProgressReporter(
        sys.stderr, enabled=False if (args.quiet or machine) else None
    )
    try:
        # ベイク。引数由来の異常は下の except で code 別に分ける(range_overlap / value_overflow)。
        # 機械モードはベイク進捗をイベントで出す(TTY 非依存)。段開始で done=0,total=null を1本、
        # 以降は bake() のフレーム進捗コールバックで done/total を出す。非機械は従来どおりライブ行の段開始のみ。
        bake_cb = None
        if machine:
            bake_start = time.monotonic()
            emitter.progress(stage="bake", done=0, total=None, note="", elapsed=0.0)
            bake_cb = lambda done, total: emitter.progress(
                stage="bake", done=done, total=total, note="",
                elapsed=time.monotonic() - bake_start,
            )
        else:
            reporter.stage("ベイク")
        try:
            result = bake(
                doc.camera,
                ranges=ranges,
                seed=args.seed,
                amp_rot=amp_rot,
                amp_pos=amp_pos,
                rot_weights=rot_weights,
                freq=freq,
                motion_damp=motion_damp,
                settle=settle,
                fade_sec=fade,
                cut_pos_threshold=cut_threshold[0],
                cut_rot_threshold=cut_threshold[1],
                impulses=tuple(args.impulses or ()),
                progress=bake_cb,
                **internal,
            )
        except RangeOverlapError as e:
            # 範囲の重複/接触。意図的な引数エラーなので range_overlap に対応付ける。
            return fail("range_overlap", str(e), 2, field="--range")
        except (ValueError, OverflowError) as e:
            # 過大値でベイクが破綻(例 --fade 1e308 → int(inf) の OverflowError、inf 回転による
            # math domain error の ValueError)。単一引数へ帰属させられないので field は null。
            return fail("value_overflow", f"値が過大でベイクが破綻した: {type(e).__name__}: {e}", 2)
        # 進捗行を解放してから warning(stderr)・統計を出す(行の混線を防ぐ)。
        reporter.close()

        # 焼き出力が非有限(inf/nan)なら引数起因の異常として弾く(non_finite_output)。
        # 通常の有限引数では過大値でベイクが例外側に倒れるため到達しにくいが、float32 は inf/nan を
        # 例外なく素通しするので、書き出し前の防御的検査として残す。
        if not _all_finite(result.camera_keys):
            return fail("non_finite_output",
                        "焼き出力が非有限(inf/nan)になった。振幅・重みが過大", 2)

        # 警告: io.read のライブラリ警告(コードはハイフン形式のまま透過)+ bake の
        # 構造化警告 + 非カメラセクション透過。機械モードは stdout へ warning イベント、非機械は stderr へ1行。
        warnings = [
            ShakeWarning(w.code, w.message, (w.section,) if w.section else None)
            for w in read_warnings
        ]
        warnings += list(result.warnings)
        sections = [
            name for name, present in (
                ("bone", doc.bone), ("morph", doc.morph), ("light", doc.light),
                ("self_shadow", doc.self_shadow), ("ik_property", doc.ik_property),
            ) if present
        ]
        if sections:
            warnings.append(ShakeWarning(
                "non_camera_sections_passthrough",
                "カメラ以外のセクションは無加工で透過した",
                tuple(sections),
            ))
        for w in warnings:
            if machine:
                emitter.warning(
                    code=w.code, message=w.message,
                    section=list(w.section) if w.section is not None else None,
                )
            else:
                print(f"warning: {w.code}: {w.message}", file=sys.stderr)

        # 適用範囲(スナップ後)・統計を算出(dry-run/verbose 用)。bake は不変のまま、
        # 出力と cuts/interp から求める。範囲端は最近接キーへスナップ。
        wv_frames = [k.frame for k in _working_view(doc.camera)]
        if ranges is None:
            applied = [(wv_frames[0], wv_frames[-1])]
        else:
            applied = sorted((_snap(s, wv_frames), _snap(e, wv_frames)) for (s, e) in ranges)
        max_amp, detected_cuts = _shake_stats(
            doc.camera, result.camera_keys, applied, cut_threshold[0], cut_threshold[1])

        # 詳細統計は --dry-run と --verbose のみで表示(通常実行は出さない)。
        # 機械モードは stdout をイベント専用にするので人間向け統計 print を抑止する(統計は result イベントへ)。
        if not machine and (args.dry_run or args.verbose):
            print(f"range: {applied}")
            print(f"keys: {len(result.camera_keys)}")
            print(f"max amplitude: {max_amp:.6g}")
            print(f"cuts: {detected_cuts}")

        # --dry-run は VMD を書かない。統計表示のみ。進捗行は finally の close で消える。
        # 機械モードでは入力検査(mode:"inspect")として、書かずに入力メタ情報 + 揺れ
        # プレビュー統計を result で出してストリームを終端する。keys は入力カメラキー数(正規化作業
        # ビューの件数=ベイク後の密キー数ではない)、frame_range は入力キーの [最小, 最大]、
        # duration_sec は最大フレーム÷30(ベイクは 30fps 固定)、sections は camera と混在する
        # 全セクション名。applied_ranges/max_amplitude/detected_cuts は bake の統計と同義。
        if args.dry_run:
            if machine:
                emitter.result(
                    mode="inspect",
                    output=None,
                    input_kind="camera",
                    keys=len(wv_frames),
                    frame_range=[int(wv_frames[0]), int(wv_frames[-1])],
                    duration_sec=wv_frames[-1] / 30.0,
                    sections=["camera"] + list(sections),
                    applied_ranges=[[int(a), int(b)] for a, b in applied],
                    max_amplitude=float(max_amp),
                    detected_cuts=[int(c) for c in detected_cuts],
                )
            return 0

        # --smooth: ベイクした密キーをプロセス内でそのまま reduce へ渡し、疎ベジェへ変換する
        # (中間VMDは作らない)。範囲はベイクと同じスナップ後範囲 result.resolved を使い、CLI
        # パイプラインと同値にする。
        # カット境界: bake は原本の隣接キーで、reduce はベイク後の毎フレーム値でカットを検出するため
        # 検出ドメインが異なる。reduce 側の再検出は無効化し(no_cut_detect=True)、ベイクが確定した
        # カット detected_cuts だけを境界とする。カット F は F-1→F の不連続なので、両側を必須キーに
        # して境界をまたぐ曲線を作らないよう F-1 と F の対を keep_frames に渡す(perspective 切替は
        # reduce が常に境界化する)。再検出を切ることで、振幅の大きい手ぶれを区間内の偽カットと
        # 誤判定する余地も無くす。
        camera_out = result.camera_keys
        if args.smooth:
            # 機械モードはスムージング進捗をイベントで出す。段開始で done=0,total=null を1本、以降は
            # reduce_camera_track の progress(done, total, note) をそのままイベント化する。非機械は
            # 従来どおりライブ行の段開始 + reporter.update を渡す。
            if machine:
                smooth_start = time.monotonic()
                emitter.progress(stage="smooth", done=0, total=None, note="", elapsed=0.0)
                smooth_cb = lambda done, total, note="": emitter.progress(
                    stage="smooth", done=done, total=total, note=note,
                    elapsed=time.monotonic() - smooth_start,
                )
            else:
                reporter.stage("スムージング")
                smooth_cb = reporter.update
            # 機械的 grid: 各範囲を max_seg 間隔のキーで区切る。sliding max_seg は「線形でも許容内に収まる」
            # 長区間を作り、手ぶれを疎キー＋線形補間=キー境界のコーナーで返すため 30fps 超でカクつき、
            # サブフレームでは揺れを取りこぼす。grid を keep に与えて区間長を max_seg 以下に抑えると、各区間は
            # 手ぶれが線形許容に収まらずベジェ曲線でフィットされ、曲線で滑らかに(judder低減)かつサブフレーム
            # でも揺れを許容内に保つ。区間が短く bounded なので least_squares 回数も抑えられ高速。
            grid = {f for f0, f1 in result.resolved for f in range(int(f0), int(f1) + 1, _SMOOTH_MAX_SEG)}
            keep = sorted({f for c in detected_cuts for f in (c - 1, c) if f >= 0} | grid)
            # progress=smooth_cb: reduce は progress(処理済みフレーム, 全フレーム総数, note) を 3 引数で
            # 呼び、出力後検証区間では note="出力後検証" を添える。非機械では smooth_cb=reporter.update で
            # シグネチャが一致し、機械では note を載せる smooth イベントに中継する。
            camera_out = reduce_camera_track(
                result.camera_keys,
                result.resolved,
                _SMOOTH_TOLERANCES,
                cut_thresholds=(cut_threshold[0], cut_threshold[1], _SMOOTH_CUT_DIST),
                keep_frames=tuple(keep),
                no_cut_detect=True,
                min_seg=1,
                max_seg=_SMOOTH_MAX_SEG,
                strict=False,
                curve_mode="bezier",
                force_bezier=True,
                progress=smooth_cb,
            )
            reporter.close()

        doc.camera = camera_out

        # 出力書き込み。失敗の原因で終了コードを分ける:
        # - OverflowError: 過大な値が float32 シリアライズで溢れた=引数起因 → コード2。
        # - その他の例外: 実際の I/O 失敗(権限・不正パス・ディスク等)→ コード3。
        try:
            io.write_file(doc, output)
        except OverflowError:
            return fail("value_overflow", "値が大きすぎて書き込み時に float32 で溢れた", 2)
        except Exception as e:
            return fail("write_failed", f"出力の書き込みに失敗: {type(e).__name__}: {e}",
                        3, field="--output", path=output)

        # 書き込み成功後に終端イベント/完了行を1回出す(進捗行は close で消えている)。
        # 機械モードは result(mode:"bake")でストリームを終端する。applied_ranges/detected_cuts は
        # numpy int が混じると json.dumps が失敗するため素の int/float へ変換する。
        if machine:
            emitter.result(
                mode="bake",
                output=output,
                keys=len(camera_out),
                applied_ranges=[[int(a), int(b)] for a, b in applied],
                max_amplitude=float(max_amp),
                detected_cuts=[int(c) for c in detected_cuts],
            )
        else:
            reporter.summary(f"完了 {output}")
        return 0
    finally:
        reporter.close()
