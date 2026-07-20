"""協調調音のテスト。

両唇閉鎖・無音を挟まず直接隣接する異母音グループの境界で、閉口を挟まず中間口形へ遷移すること、
遷移長が基準長と区間長で決まる(口形差では短縮しない)こと、両唇閉鎖を挟む境界では協調調音を作らないことを
既知値で検証する。境界のキー配置は遷移長が偶数(T=4)になる coartic_overlap_max=4 の明快なフィクスチャで
検証する(短い側区間長10/2=5 で頭打ちされず基準長4が効く)。
"""

import math

import pytest

import lipsync
from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape
from lipsync import generate

# T=4 を得るための overlap_max=4(T=clamp(4, 1, 短い側10/2=5)=4、境界 b±2 の整数窓[8,12])。
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

_NONE = ConsonantClass.NONE


def test_shape_diff_identical_is_zero():
    assert generate._shape_diff(
        MouthShape.A, _NONE, MouthShape.A, _NONE, GenerationParams()
    ) == pytest.approx(0.0)


def test_shape_diff_disjoint_is_one():
    # 純母音 あ={あ}・う={う} はモーフ集合が重ならず直交 → 正規化距離/√2 = 1.0。
    assert generate._shape_diff(
        MouthShape.A, _NONE, MouthShape.U, _NONE, GenerationParams()
    ) == pytest.approx(1.0)


def test_shape_diff_partial_known_value_with_consonant():
    # 純母音どうしは直交だが、子音変調を入れると部分重複が生じる。あ(子音なし)={あ:1.0}・
    # あ(ROUNDED)={あ:1.0, う:0.3}。L2正規化後の距離/√2。子音変調が口形差に効くことを既知値で固定。
    dot = 1.0 / math.sqrt(1.09)  # unit_a·unit_b = 1/√(1+0.3²)
    expected = math.sqrt(2.0 - 2.0 * dot) / math.sqrt(2.0)
    assert generate._shape_diff(
        MouthShape.A, _NONE, MouthShape.A, ConsonantClass.ROUNDED, GenerationParams()
    ) == pytest.approx(expected)


# --- 遷移長 _transition_frames(純粋ヘルパ。合成 diff で精密検証) ---

@pytest.mark.parametrize(
    "shorter_len,overlap_max,expected",
    [
        (10.0, 4, 4),   # min(4, 5)=4(基準長まで広く取る)
        (4.0, 4, 2),    # 短い側1/2=2 で頭打ち(自動短縮)
        (10.0, 2, 2),   # min(2, 5)=2
        (10.0, 1, 1),   # min(1, 5)=1
        (1.0, 4, 1),    # 短い側1/2=0.5 < 1 を clamp 下限1へ引き上げ
    ],
)
def test_transition_frames_clamped_to_base_and_half(shorter_len, overlap_max, expected):
    # 遷移長 = clamp(基準長, 1, 短い側区間長/2)。口形差で短縮しない(差に依らず同じ)。
    p = GenerationParams(coartic_overlap_max=overlap_max)
    for diff in (0.0, 0.5, 1.0):
        assert generate._transition_frames(diff, shorter_len, p) == expected


# --- 境界のキー配置(統合。overlap_max=4 で T=4) ---

def test_no_close_at_coartic_boundary():
    # あ[0,10]・う[10,20] は直接隣接の異母音 → 境界10で閉口せず中間口形へ。
    # overlap_max=4 で T=min(4, 短い側10/2=5)=4、窓[8,12]、境界10は中間口形 あ:0.25・う:0.25・お:0.05。
    # 中間口形値 (w_a+w_b)/2 は T に依らない(窓幅だけ変わる)。
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)],
        _WIDE,
    )
    at10 = {name: w for name, keys in env.items() for f, w in keys if f == 10}
    assert at10 == pytest.approx({"あ": 0.25, "う": 0.25})


def test_coartic_full_envelopes():
    # あ→う(純母音)、overlap_max=4 で T=min(4, 5)=4(口形差で短縮しない)、窓[8,12]。
    # 前母音は先頭アタックのみ、次母音は末尾リリースのみ。純母音なので補助モーフは出ない。
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)],
        _WIDE,
    )
    assert set(env) == {"あ", "う"}
    # あ: 0からアタックで0.5、保持、遷移始端8で0.5、境界10で0.25、遷移終端12で0(うへ明け渡す)。
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.25), (12, 0.0)])
    # う: 遷移始端8で0、境界0.25、12で0.5に達し保持、末尾リリースで0。
    _approx_envelope(env["う"], [(8, 0.0), (10, 0.25), (12, 0.5), (18, 0.5), (20, 0.0)])


@pytest.mark.xfail(reason="impl pending: 両唇閉鎖隣接時の先行/後行残し", strict=True)
def test_bilabial_between_no_coartic():
    # あ[0,10]・両唇閉鎖[10,14]・う[14,24]: 両唇閉鎖を挟むので終端10≠次始端14 → 協調調音を作らない
    # (各々 単一区間エンベロープ)。両唇閉鎖区間は専用の閉口キーを持たず、キーは出力されない
    # (閉口はキー不在=0で表す)。既定 overlap_max(協調調音が起きうる設定)でも、両唇閉鎖を挟むと
    # 非協調になることを示す。
    # 両唇閉鎖[10,14]は先行準備・後行残しの対象(隣接区間長4、開き量比0.5/0.8=0.625)でもある:
    # あ の後行残し R_eff=min(half_up(1×0.625)=1, floor(4/2)=2)=1 で境界10からさらに1フレーム
    # (11)まで緩やかに閉じる。う の先行準備 A_eff は同じ計算で1、境界14の1フレーム手前(13)から
    # 立ち上がる。
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5),
            MouthEvent(MouthShape.BILABIAL, 10.0, 14.0),
            MouthEvent(MouthShape.U, 14.0, 24.0, 0.5),
        ],
        GenerationParams(),
    )
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (10, 0.5), (11, 0.0)])
    _approx_envelope(env["う"], [(13, 0.0), (14, 0.5), (22, 0.5), (24, 0.0)])
