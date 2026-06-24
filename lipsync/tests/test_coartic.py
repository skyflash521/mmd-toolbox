"""L-4 協調調音のテスト(implementation-plan.md §4.3, lipsync.md §3/§4)。

両唇閉鎖・無音を挟まず直接隣接する異母音グループの境界で、閉口を挟まず中間口形へ短く遷移すること、
遷移長が口形差と区間長で決まること、両唇閉鎖を挟む境界では協調調音を作らないことを既知値で検証する。
口形差・遷移長は純粋ヘルパとして精密に、境界のキー配置は遷移長が偶数(T=2)になる
coartic_overlap_max=4 の明快なフィクスチャで検証する(既定 overlap_max=2 では T=1 が量子化で潰れ
衝突解決が §4.5/L-9 管轄になるため)。協調調音の新挙動・未実装ヘルパは xfail で印を付け、両唇閉鎖を
挟む非協調の回帰ガードは現行実装でも成立するため印を付けない。
"""

import math

import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape
from lipsync import generate

_L4 = pytest.mark.xfail(reason="impl pending: L-4 協調調音")

# T=2 を得るための overlap_max=4(直交対 diff=1 で T=round(4*(1-0.5))=2、境界 b±1 の整数窓)。
_WIDE = GenerationParams(coartic_overlap_max=4)


def _envelope(events, params):
    by_morph: dict[str, list[tuple[int, float]]] = {}
    for k in lipsync.generate_morph_keys(events, params):
        by_morph.setdefault(k.name, []).append((k.frame, k.weight))
    for name in by_morph:
        by_morph[name].sort()
    return by_morph


def _approx_envelope(actual, expected):
    assert [f for f, _ in actual] == [f for f, _ in expected]
    for (_, aw), (_, ew) in zip(actual, expected):
        assert aw == pytest.approx(ew)


# --- 口形差 _shape_diff(純粋ヘルパ。端点と算出可能な部分値) ---

@_L4
def test_shape_diff_identical_is_zero():
    assert generate._shape_diff(MouthShape.A, MouthShape.A, GenerationParams()) == pytest.approx(0.0)


@_L4
def test_shape_diff_disjoint_is_one():
    # あ={あ}・う={う,お} はモーフ集合が重ならず直交 → 正規化距離/√2 = 1.0。
    assert generate._shape_diff(MouthShape.A, MouthShape.U, GenerationParams()) == pytest.approx(1.0)


@_L4
def test_shape_diff_partial_known_value():
    # う={う:1.0,お:0.2}・お={う:0.2,お:1.0}。L2正規化後の距離/√2。
    dot = (1.0 * 0.2 + 0.2 * 1.0) / 1.04
    expected = math.sqrt(2.0 - 2.0 * dot) / math.sqrt(2.0)
    assert generate._shape_diff(MouthShape.U, MouthShape.O, GenerationParams()) == pytest.approx(expected)


# --- 遷移長 _transition_frames(純粋ヘルパ。合成 diff で精密検証) ---

@_L4
@pytest.mark.parametrize(
    "diff,shorter_len,overlap_max,expected",
    [
        (1.0, 10.0, 4, 2),   # 4*(1-0.5)=2、cap=min(4,5)=4 → 2
        (0.5, 10.0, 4, 3),   # 4*0.75=3 → 3(diff 大ほど短い: 1.0→2 < 0.5→3)
        (0.0, 10.0, 4, 4),   # 4*1=4 → 4
        (0.0, 4.0, 4, 2),    # 値4だが短い側1/2=2 で頭打ち(自動短縮)
        (1.0, 10.0, 2, 1),   # 既定 overlap_max=2: 2*0.5=1.0(ちょうど下限上)
        (1.0, 10.0, 1, 1),   # overlap_max=1: 1*0.5=0.5 < 1 を clamp 下限1へ引き上げ
    ],
)
def test_transition_frames_formula(diff, shorter_len, overlap_max, expected):
    p = GenerationParams(coartic_overlap_max=overlap_max)
    assert generate._transition_frames(diff, shorter_len, p) == expected


# --- 境界のキー配置(統合。overlap_max=4 で T=2) ---

@_L4
def test_no_close_at_coartic_boundary():
    # あ[0,10]・う[10,20] は直接隣接の異母音 → 境界10で閉口せず中間口形へ。
    # diff=1 で T=2、窓[9,11]、境界10は中間口形 あ:0.25・う:0.25・お:0.05(いずれも非ゼロ)。
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)],
        _WIDE,
    )
    at10 = {name: w for name, keys in env.items() for f, w in keys if f == 10}
    assert at10 == pytest.approx({"あ": 0.25, "う": 0.25, "お": 0.05})


@_L4
def test_coartic_full_envelopes():
    # あ→う、overlap_max=4 で T=2、窓[9,11]。前母音は先頭アタックのみ、次母音は末尾リリースのみ。
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)],
        _WIDE,
    )
    assert set(env) == {"あ", "う", "お"}
    # あ: 0からアタックで0.5、保持、境界で0.25へ、遷移終端11で0(うへ明け渡す)。
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (9, 0.5), (10, 0.25), (11, 0.0)])
    # う: 遷移始端9で0、境界0.25、11で0.5に達し保持、末尾リリースで0。
    _approx_envelope(env["う"], [(9, 0.0), (10, 0.25), (11, 0.5), (18, 0.5), (20, 0.0)])
    # お(うの補助 0.1): 同じ窓で 0→0.05→0.1、保持、リリースで0。
    _approx_envelope(env["お"], [(9, 0.0), (10, 0.05), (11, 0.1), (18, 0.1), (20, 0.0)])


def test_bilabial_between_no_coartic():
    # あ[0,10]・両唇閉鎖[10,14]・う[14,24]: 両唇閉鎖を挟むので終端10≠次始端14 → 協調調音を作らない。
    # あ は境界10で閉口0へ戻り、う は14で0から開く(各々 §4.9 の単一区間エンベロープ)。
    # 両唇閉鎖区間の閉口キーは L-8 で置くため、この段階では出力されない。
    # 既定 overlap_max(協調調音が起きうる設定)でも、両唇閉鎖を挟むと非協調になることを示す。
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.BILABIAL, 10.0, 14.0),
            MouthEvent(MouthShape.U, 14.0, 24.0, 0.5),
        ],
        GenerationParams(),
    )
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])
    _approx_envelope(env["う"], [(14, 0.0), (16, 0.5), (22, 0.5), (24, 0.0)])
