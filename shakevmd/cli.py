"""shakevmd CLI(shakevmd.md §2, §9)。

コアの薄いラッパー: 引数解析 → VMD読み(mmd_toolbox.vmd.io)→ bake() → VMD書き。
詳細パラメーター(オクターブ構成・persistence・プロファイル・手動カット等)は公開しない(§8)。

終了コード(§9): 0 正常 / 1 入力不正(欠落・非VMD・カメラキーなし)/
2 引数エラー(範囲不正・逆順・重複・上書きガード・未知オプション等)/ 3 出力書き込み失敗。
"""

import argparse
import math
import os
import sys

from mmd_toolbox.vmd import interp, io
from mmd_toolbox.vmd.reduce import Tolerances, reduce_camera_track
from shakevmd import cuts, presets
from shakevmd.bake import bake

# 公開引数の hard-default(§2.3-2.6)。プリセット/個別引数が未指定の項目に使う。
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
    """`START:END` を (start|None, end|None) に解析する(§2.2, §5.2)。

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
    """`P,Y,R` を (float, float, float) に解析する(§2.3)。"""
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
    """`位置,角度` を (float, float) に解析する(§2.6)。"""
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
    """`F:S:D` を (int, float, float) に解析する(§2.4)。F=フレーム, S=強さ度, D=減衰秒。"""
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


def _build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: 仕様外の前置き省略形(--over→--overwrite 等)を受理しない。
    # 未知/省略形は exit 2(§8/§2.7 の非公開・繰延フラグ拒否とも整合)。
    p = argparse.ArgumentParser(prog="shakevmd", allow_abbrev=False)
    p.add_argument("input")
    p.add_argument("-o", "--output")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--range", dest="ranges", action="append", type=_parse_range)
    # 公開揺れパラメーターは default=None(未指定センチネル)。--preset と個別引数の優先を
    # main() で解決する(明示 > preset > hard-default)。型検証は明示値にのみ適用される。
    p.add_argument("--amp-rot", type=_nonneg_float)        # 振幅 ≥0
    p.add_argument("--amp-pos", type=_nonneg_float)        # 振幅 ≥0
    p.add_argument("--rot-weights", type=_parse_rot_weights)
    p.add_argument("--freq", type=_positive_float)         # 周波数 >0
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fade", type=_nonneg_float)           # 秒 ≥0
    p.add_argument("--motion-damp", type=_nonneg_float)    # 減衰係数 ≥0(0で無効)
    p.add_argument("--settle", type=_nonneg_float)         # 度(振幅)≥0(0で無効)
    p.add_argument("--cut-threshold", type=_parse_cut_threshold)
    p.add_argument("--impulse", dest="impulses", action="append", type=_parse_impulse)
    # §2.7 運用/プリセット系。
    p.add_argument("--preset", choices=presets.PRESET_NAMES)   # 未知名は argparse が exit 2
    p.add_argument("--dry-run", dest="dry_run", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    # 既定 on: ベイク後にプロセス内で疎ベジェへ削減し、30fps 超再生のカクつきを低減する。
    # --no-smooth で無効化(密キー＋線形のまま出力する)。
    p.add_argument("--smooth", default=True, action=argparse.BooleanOptionalAction)
    return p


def _default_output(input_path: str) -> str:
    # §2.2: 既定出力は `<入力名(拡張子なし)>_shake.vmd`。元の拡張子に依らず常に .vmd。
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


def _same_path(a: str, b: str) -> bool:
    # 実ファイルが同一かを優先(symlink・大小無視 FS でも inode で一致判定)。
    # 出力先が未存在なら samefile が立たないので、symlink 解決した realpath で比較する。
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


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
    """公開引数を 明示 > preset > hard-default の優先で解決する(§2.7)。"""
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


def main(argv=None) -> int:
    """CLI エントリポイント。終了コードを返す(§9: 0/1/2/3)。"""
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

    # 上書きガード: 入力と同一パスへの出力は --overwrite 必須(§2.2)。未許可なら書かずにエラー。
    if not args.overwrite and _same_path(output, args.input):
        return 2

    # 入力読み込み(欠落・非VMD・カメラキーなし → 入力不正 §9 コード1)。
    # io.read は継続可能な問題(名前のデコード不可・トレーリングデータ等)を警告で返す。
    # VMD I/O は mmd_toolbox へ委譲する設計なので、その警告もユーザーへ伝播する。
    try:
        doc, read_warnings = io.read(args.input)
    except Exception:
        return 1
    if not doc.camera:
        return 1

    # 範囲の省略側を先頭/末尾キーへ解決(端のスナップ・重複検出は bake() が担う)。
    ranges = None
    if args.ranges:
        frames = [k.frame for k in doc.camera]
        first, last = min(frames), max(frames)
        ranges = [
            (first if s is None else s, last if e is None else e)
            for (s, e) in args.ranges
        ]
        # 省略端を先頭/末尾へ解決した「後」にも逆順を検査する(§2.2 START>END は引数エラー)。
        # 例: `999:` は END=末尾60 に解決され 999>60。bake は端をスナップ後に swap するため
        # ここで弾かないと [60,60] として黙って焼かれてしまう。
        if any(s > e for (s, e) in ranges):
            return 2

    # 公開揺れパラメーターを解決(明示 > --preset > hard-default、§2.7)。
    preset = presets.get_preset(args.preset) if args.preset else {}
    amp_rot = _resolve_param("amp_rot", args, preset)
    amp_pos = _resolve_param("amp_pos", args, preset)
    rot_weights = _resolve_param("rot_weights", args, preset)
    freq = _resolve_param("freq", args, preset)
    motion_damp = _resolve_param("motion_damp", args, preset)
    settle = _resolve_param("settle", args, preset)
    cut_threshold = _resolve_param("cut_threshold", args, preset)
    fade = _resolve_param("fade", args, preset)
    # 内蔵パラメーター(CLI 非公開、プリセットのみ。§8)を bake へ転送する。
    # 例: walking の歩調成分 gait_freq/gait_amp。bake 既定(無効)を上書きする。
    internal = {k: preset[k] for k in presets.INTERNAL_PARAM_NAMES if k in preset}

    # ベイク。引数由来の異常は §9 コード2 に集約する:
    # - ValueError: 範囲の重複/接触(空カメラは上で弾き済み)。
    # - OverflowError: 有限だが過大な値(例 --fade 1e308 → int(round(fade*FPS)) が inf 変換で失敗)。
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
            **internal,
        )
    except (ValueError, OverflowError):
        return 2

    # 焼き出力が非有限(inf/nan)なら引数起因の異常 → 引数エラー(§9 コード2)。
    # bake 内の乗算で有限引数が inf 化しても float32 は inf を素通しするため明示検査する。
    if not _all_finite(result.camera_keys):
        return 2

    # 警告表示(§3.1): io.read の警告 + bake の警告 + 非カメラセクション透過。常に表示する。
    warnings = [f"{w.code}: {w.message}" for w in read_warnings]
    warnings += list(result.warnings)
    if doc.bone or doc.morph or doc.light or doc.self_shadow or doc.ik_property:
        warnings.append("カメラ以外のセクションは無加工で透過した(§3.1)")
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    # 適用範囲(スナップ後)・統計を算出(dry-run/verbose 用)。bake は不変のまま、
    # 出力と cuts/interp から求める。範囲端は最近接キーへスナップ(§5.2)。
    wv_frames = [k.frame for k in _working_view(doc.camera)]
    if ranges is None:
        applied = [(wv_frames[0], wv_frames[-1])]
    else:
        applied = sorted((_snap(s, wv_frames), _snap(e, wv_frames)) for (s, e) in ranges)
    max_amp, detected_cuts = _shake_stats(
        doc.camera, result.camera_keys, applied, cut_threshold[0], cut_threshold[1])

    # 詳細統計は --dry-run と --verbose のみで表示(通常実行は出さない、§5.2/§5.3)。
    if args.dry_run or args.verbose:
        print(f"range: {applied}")
        print(f"keys: {len(result.camera_keys)}")
        print(f"max amplitude: {max_amp:.6g}")
        print(f"cuts: {detected_cuts}")

    # --dry-run は VMD を書かない。統計表示のみ(§2.7)。
    if args.dry_run:
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
        # 機械的 grid: 各範囲を max_seg 間隔のキーで区切る。sliding max_seg は「線形でも許容内に収まる」
        # 長区間を作り、手ぶれを疎キー＋線形補間=キー境界のコーナーで返すため 30fps 超でカクつき、
        # サブフレームでは揺れを取りこぼす。grid を keep に与えて区間長を max_seg 以下に抑えると、各区間は
        # 手ぶれが線形許容に収まらずベジェ曲線でフィットされ、曲線で滑らかに(judder低減)かつサブフレーム
        # でも揺れを許容内に保つ。区間が短く bounded なので least_squares 回数も抑えられ高速。
        grid = {f for f0, f1 in result.resolved for f in range(int(f0), int(f1) + 1, _SMOOTH_MAX_SEG)}
        keep = sorted({f for c in detected_cuts for f in (c - 1, c) if f >= 0} | grid)
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
        )

    doc.camera = camera_out

    # 出力書き込み。失敗の原因で終了コードを分ける(§9):
    # - OverflowError: 過大な値が float32 シリアライズで溢れた=引数起因 → コード2。
    # - その他の例外: 実際の I/O 失敗(権限・不正パス・ディスク等)→ コード3。
    try:
        io.write_file(doc, output)
    except OverflowError:
        return 2
    except Exception:
        return 3

    return 0
