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

from mmd_toolbox.vmd import io
from shakevmd.bake import bake


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
    p.add_argument("--amp-rot", type=_nonneg_float, default=0.8)       # 振幅 ≥0
    p.add_argument("--amp-pos", type=_nonneg_float, default=0.05)      # 振幅 ≥0
    p.add_argument("--rot-weights", type=_parse_rot_weights, default=(1.0, 1.0, 0.3))
    p.add_argument("--freq", type=_positive_float, default=1.2)        # 周波数 >0
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--fade", type=_nonneg_float, default=0.7)          # 秒 ≥0
    p.add_argument("--motion-scale", type=_nonneg_float, default=0.5)  # 係数 ≥0(0で無効)
    p.add_argument("--settle", type=_nonneg_float, default=0.3)        # 度(振幅)≥0(0で無効)
    p.add_argument("--cut-threshold", type=_parse_cut_threshold, default=(5.0, 20.0))
    p.add_argument("--impulse", dest="impulses", action="append", type=_parse_impulse)
    return p


def _default_output(input_path: str) -> str:
    # §2.2: 既定出力は `<入力名(拡張子なし)>_shake.vmd`。元の拡張子に依らず常に .vmd。
    base, _ = os.path.splitext(input_path)
    return base + "_shake.vmd"


def _all_finite(keys) -> bool:
    """全カメラキーの数値(距離・位置・回転)が有限か。

    有限な引数でも bake 内の乗算(例 --amp-pos 1e308 × --motion-scale 1e308)で inf に
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

    # ベイク。引数由来の異常は §9 コード2 に集約する:
    # - ValueError: 範囲の重複/接触(空カメラは上で弾き済み)。
    # - OverflowError: 有限だが過大な値(例 --fade 1e308 → int(round(fade*FPS)) が inf 変換で失敗)。
    try:
        result = bake(
            doc.camera,
            ranges=ranges,
            seed=args.seed,
            amp_rot=args.amp_rot,
            amp_pos=args.amp_pos,
            rot_weights=args.rot_weights,
            freq=args.freq,
            motion_scale=args.motion_scale,
            settle=args.settle,
            fade_sec=args.fade,
            cut_pos_threshold=args.cut_threshold[0],
            cut_rot_threshold=args.cut_threshold[1],
            impulses=tuple(args.impulses or ()),
        )
    except (ValueError, OverflowError):
        return 2

    # 焼き出力が非有限(inf/nan)なら引数起因の異常 → 引数エラー(§9 コード2)。
    # bake 内の乗算で有限引数が inf 化しても float32 は inf を素通しするため明示検査する。
    if not _all_finite(result.camera_keys):
        return 2

    doc.camera = result.camera_keys

    # 警告表示(§3.1): io.read の警告(名前デコード不可等)+ bake の警告(正規化・帯域制限
    # クランプ等)+ 非カメラセクション透過。安定マーカー "warning:" を前置して提示する。
    warnings = [f"{w.code}: {w.message}" for w in read_warnings]
    warnings += list(result.warnings)
    if doc.bone or doc.morph or doc.light or doc.self_shadow or doc.ik_property:
        warnings.append("カメラ以外のセクションは無加工で透過した(§3.1)")
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)

    # 出力書き込み。失敗の原因で終了コードを分ける(§9):
    # - OverflowError: 過大な値が float32 シリアライズで溢れた=引数起因 → コード2。
    #   (例 --amp-rot 1e308 → 回転 ~1e306 → struct.pack("<f") が溢れる)
    # - その他の例外: 実際の I/O 失敗(権限・不正パス・ディスク等)→ コード3。
    try:
        io.write_file(doc, output)
    except OverflowError:
        return 2
    except Exception:
        return 3

    return 0
