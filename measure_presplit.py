"""flat-span-presplit §5 受入計測(一時スクリプト・未コミット)。

代表入力 tuning/dense_dance_motion_in.vmd を production と同じ疎化パラメータ
(max_seg=15・min_seg=1・cut=(1.0,30.0)・bezier・medium 許容)でボーン疎化し、
wall-time・総出力キー数・定数(do-nothing/静止)トラックのキー数を測る。

ウォール時間上限はこのスクリプト内に持つ(ボーン間で経過を見て打ち切り)。Pool は使わず
シリアルに reduce_bone_track を直呼びする(オーファン化を防ぐ)。before/after は、それぞれの
コードを持つ作業ツリーから本スクリプトを走らせて比較する(同一の絶対 VMD パスを使う)。

使い方: python measure_presplit.py <vmd_abs_path> <label> [wall_cap_secs]
"""

import sys
import time
from collections import OrderedDict
from pathlib import Path

# worktree を切り替えた before/after 比較で、必ずこのスクリプトと同じツリーの実装(libs/vmd /
# tools/mocapvmd)を読むよう、同じツリーの libs/・tools/ を import より先に sys.path 先頭へ置く
# (editable install や cwd 依存だと別ツリーを指しうる)。
_TREE_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_TREE_ROOT / "tools"))
sys.path.insert(0, str(_TREE_ROOT / "libs"))

from vmd import io  # noqa: E402
from vmd.reduce import build_bone_tolerances, reduce_bone_track  # noqa: E402
from mocapvmd import classify, presets  # noqa: E402

CUT_THRESHOLDS = (1.0, 30.0)
MIN_SEG = 1
MAX_SEG = 15
PRESET = "medium"


def _reduce_one(keys, tol_pos, tol_rot):
    tols = build_bone_tolerances(tol_pos, tol_rot)
    return list(
        reduce_bone_track(
            keys,
            [(keys[0].frame, keys[-1].frame)],
            tols,
            cut_thresholds=CUT_THRESHOLDS,
            keep_frames=[],
            no_cut_detect=False,
            min_seg=MIN_SEG,
            max_seg=MAX_SEG,
            strict=False,
            curve_mode="bezier",
        )
    )


def _is_constant_track(keys):
    p0 = keys[0].position
    r0 = keys[0].rotation
    for k in keys:
        if k.position != p0 or k.rotation != r0:
            return False
    return True


def _worst_case_overhead():
    """最悪ケース回帰: 長大単一 span が「末尾でだけ逸脱」する合成トラック。is_constant が span 全体を
    走査してから False を返す純粋オーバーヘッドを測る(逸脱はカット閾値未満・定数しきい値超で隔離)。"""
    from vmd.types import BoneKey

    name_raw = b"worst\x00".ljust(15, b"\x00")
    n = 2000
    keys = []
    for f in range(n + 1):
        pos = (0.0, 0.0, 0.0) if f < n else (0.5, 0.0, 0.0)  # 末尾のみ逸脱(cut<1.0, const>1e-9)
        keys.append(BoneKey(name_raw=name_raw, frame=f, position=pos,
                            rotation=(0.0, 0.0, 0.0, 1.0), interpolation=bytes(64)))
    t = time.monotonic()
    out = _reduce_one(keys, 0.20, 1.50)
    return time.monotonic() - t, len(out)


def main():
    vmd_path = sys.argv[1]
    label = sys.argv[2]
    wall_cap = float(sys.argv[3]) if len(sys.argv) > 3 else 480.0

    doc, _ = io.read(vmd_path)
    tracks = OrderedDict()
    for k in doc.bone:
        tracks.setdefault(k.name_raw, []).append(k)
    for name_raw in tracks:
        tracks[name_raw].sort(key=lambda k: k.frame)

    multi = {nr: ks for nr, ks in tracks.items() if len(ks) > 1}
    total_bones = len(multi)
    # 定数トラックを先に処理して必ず全数測る(高速)。移動トラックは遅いので上限まで。
    const_nrs = [nr for nr, ks in multi.items() if _is_constant_track(ks)]
    moving_nrs = [nr for nr in multi if nr not in set(const_nrs)]
    ordered = const_nrs + moving_nrs
    print(f"[{label}] bones(total)={len(tracks)} multi-key={total_bones} "
          f"constant={len(const_nrs)} cap={wall_cap}s")

    done = 0
    in_keys = out_keys = 0
    const_rows = []
    start = time.monotonic()
    const_elapsed = None
    capped = False
    for nr in ordered:
        if done >= len(const_nrs) and time.monotonic() - start > wall_cap:
            capped = True
            break
        ks = multi[nr]
        name = nr.split(b"\x00", 1)[0].decode("cp932", "replace")
        cat = classify.classify(name)
        tol = presets.resolve_reduction_tolerances(PRESET, cat)
        out = _reduce_one(ks, tol["bone_pos"], tol["bone_rot"])
        done += 1
        in_keys += len(ks)
        out_keys += len(out)
        if _is_constant_track(ks):
            const_rows.append((name, len(ks), len(out), ks[-1].frame - ks[0].frame))
        if done == len(const_nrs):
            const_elapsed = time.monotonic() - start
    elapsed = time.monotonic() - start

    print(f"[{label}] reduced bones={done}/{total_bones} capped={capped} "
          f"elapsed={elapsed:.1f}s in_keys={in_keys} out_keys={out_keys}")
    print(f"[{label}] constant tracks ({len(const_rows)}) reduced in {const_elapsed:.3f}s: "
          f"name in->out (range)")
    for name, ni, no, rng in const_rows:
        print(f"  [{label}] {name!r}: {ni} -> {no} (range={rng})")

    wt, wkeys = _worst_case_overhead()
    print(f"[{label}] worst-case-overhead span=2000 reduce={wt:.3f}s out_keys={wkeys}")


if __name__ == "__main__":
    main()
