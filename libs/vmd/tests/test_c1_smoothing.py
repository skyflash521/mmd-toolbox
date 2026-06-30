"""位置端点速度平滑化(§3.5 位置C1近似)のテスト。

疎化が位置誤差だけでフィットしたベジェは内部キーで速度が不連続になりがち。C1ポストパスは
各内部キーで両隣区間の端速度をソース中心差分速度に近づけ速度連続(C1)に寄せる。制御点を
[0,127] にクランプするため厳密 C1 ではなく実現可能範囲への射影(C1近似)。許容契約は破らない:
strict では C1 が許容を破る範囲では適用せず元フィットを残す(StrictError を C1 で誘発しない)。
"""

import math

import pytest

from vmd import reduce as reducer
from vmd.reduce import (
    BONE_LINEAR_INTERP,
    StrictError,
    build_bone_tolerances,
    reduce_bone_track,
    verify_bone_track,
)
from vmd.types import BoneKey

NAME = b"bone".ljust(15, b"\x00")


def _curvy_src(n=41, amp=5.0, w=0.4):
    # 複数の変曲点を持つ正弦状の位置(pos_x = amp*sin(w*f))。単一3次ベジェでは全域を表せず
    # 内部キーが必ず出るので、内部キーに作用する C1 を検証できる。
    return [
        BoneKey(NAME, f, (amp * math.sin(w * f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)
        for f in range(n)
    ]


def _reduce(src, tols, *, strict, c1, monkeypatch):
    monkeypatch.setattr(reducer, "_C1_SMOOTHING", c1)
    return reduce_bone_track(
        src,
        [(0, src[-1].frame)],
        tols,
        cut_thresholds=(5.0, 20.0),
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=strict,
        curve_mode="bezier",
    )


def test_c1_changes_interior_interpolation(monkeypatch):
    # 曲がる入力では C1 あり/なしで内部キーの補間制御点が変わる(ポストパスが効いている)。
    src = _curvy_src()
    tols = build_bone_tolerances(bone_pos=0.5, bone_rot=5.0)
    off = _reduce(src, tols, strict=False, c1=False, monkeypatch=monkeypatch)
    on = _reduce(src, tols, strict=False, c1=True, monkeypatch=monkeypatch)
    # キー列(フレーム)は一致し、内部キーの補間バイトだけが変わる。
    assert [k.frame for k in off] == [k.frame for k in on]
    assert any(off[i].interpolation != on[i].interpolation for i in range(1, len(on) - 1))


def test_c1_output_within_tolerance_nonstrict(monkeypatch):
    # 非strict: C1 適用後も出力は許容内(検証ループが密化で吸収)。
    src = _curvy_src()
    tols = build_bone_tolerances(bone_pos=0.5, bone_rot=5.0)
    keys = _reduce(src, tols, strict=False, c1=True, monkeypatch=monkeypatch)
    assert verify_bone_track(src, keys, [(0, src[-1].frame)], tols) == []


def test_c1_pos_control_points_within_0_127(monkeypatch):
    # クランプで位置チャンネルの制御点は [0,127] に収まり単調性(=MMD適合)を保つ。
    src = _curvy_src()
    tols = build_bone_tolerances(bone_pos=0.5, bone_rot=5.0)
    keys = _reduce(src, tols, strict=False, c1=True, monkeypatch=monkeypatch)
    for k in keys:
        b = k.interpolation
        for c in range(3):  # pos_x, pos_y, pos_z
            x1, y1, x2, y2 = b[c], b[4 + c], b[8 + c], b[12 + c]
            assert 0 <= x1 <= 127 and 0 <= y1 <= 127
            assert 0 <= x2 <= 127 and 0 <= y2 <= 127
            assert x1 <= x2  # X 単調(x1≤x2)


def test_c1_strict_does_not_regress(monkeypatch):
    # strict 契約: C1 なしで通る入力は C1 ありでも StrictError を出さず、出力は許容内に保たれる。
    # C1 候補が許容を破る範囲では C1 を適用せず元フィットを残すフォールバックでこれを満たす。
    # この入力の tol は、フィットは strict を通るが C1 候補は許容を超える値に選んであり、
    # フォールバック経路を実際に通す。
    src = _curvy_src()
    rng = [(0, src[-1].frame)]
    tols = build_bone_tolerances(bone_pos=0.2, bone_rot=5.0)
    # 前提1: C1 なしフィットは strict を通る(許容内)。
    base = _reduce(src, tols, strict=True, c1=False, monkeypatch=monkeypatch)
    assert verify_bone_track(src, base, rng, tols) == []
    # 前提2: そのフィットに C1 を当てた候補は許容を超える(=フォールバックが必要な入力)。
    candidate = reducer._apply_c1_bone(base, src)
    assert verify_bone_track(src, candidate, rng, tols) != []
    # 結果: strict で C1 ありでも例外を出さず、出力は許容内。
    keys = _reduce(src, tols, strict=True, c1=True, monkeypatch=monkeypatch)
    assert verify_bone_track(src, keys, rng, tols) == []
    # フォールバックで C1 を捨て元フィットを残すので、キー列も補間も base と一致する。
    assert [k.frame for k in keys] == [k.frame for k in base]
    assert [k.interpolation for k in keys] == [k.interpolation for k in base]
