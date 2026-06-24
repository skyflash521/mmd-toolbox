"""L-6 最小保持・競合短縮のテスト(implementation-plan.md §4.4, lipsync.md §3/§4)。

§4.2 グループ化後の前処理として、最小保持を満たせない短区間(available=max(0,L-min_hold_frames)<2)を
隣接母音へ吸収/除去し、保持は確保できるがアタック+リリース全量が入らない区間は competition 短縮で
実効アタック/リリースを詰めて保持を最優先で残すことを既知値で検証する。先行・協調調音の干渉を避けるため、
競合短縮は時間軸先頭の単一母音、吸収は同一母音アンカーのフィクスチャで確認する。通常長で不変であることは
回帰ガードで併せて確認する。
"""

import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape


def _envelope(events, params=None):
    params = params or GenerationParams()
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


def test_competition_shortening_proportional():
    # あ[0,7]op0.5、attack=2/release=6/min_hold=3。available=max(0,7-3)=4 < a+r=8。
    # scale=4/8=0.5、a'=max(1,2*0.5)=1、r'=4-1=3(比例 2:6=1:3 を保つ)。保持=7-1-3=3=min_hold。
    p = GenerationParams(release_frames=6)
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 7.0, 0.5)], p)
    _approx_envelope(env["あ"], [(0, 0.0), (1, 0.5), (4, 0.5), (7, 0.0)])


def test_competition_shortening_clamps_to_one():
    # あ[0,5]op0.5、attack=3/release=9/min_hold=3。available=max(0,5-3)=2 < a+r=12。
    # a'=max(1,3*2/12=0.5)=1、r'=2-1=1。各最小1フレーム。保持=5-1-1=3=min_hold。
    p = GenerationParams(attack_frames=3, release_frames=9)
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 5.0, 0.5)], p)
    _approx_envelope(env["あ"], [(0, 0.0), (1, 0.5), (4, 0.5), (5, 0.0)])


def test_short_group_dropped_at_edge():
    # あ[0,4]op0.5、既定。available=max(0,4-3)=1<2 で短区間。母音アンカー無し(単独・時間軸端)→除去。
    # 独立キーを出さず閉口(出力は空)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 4.0, 0.5)])
    assert env == {}


def test_short_vowel_absorbed_between_same_vowels():
    # あ[0,10]op0.8・い[10,14]op0.5(短)・あ[14,24]op0.3、既定。い は available=1<2 で短区間。
    # 吸収先タイブレーク=開き量大: 前あ0.8 > 後あ0.3 → 前へ吸収。前あが span を貫いて後あと同母音連結
    # (§4.2)になり、い の独立キーは消える。結果は あ 一系列のみ・内部に閉口0キー無し。
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.8),
            MouthEvent(MouthShape.I, 10.0, 14.0, 0.5),
            MouthEvent(MouthShape.A, 14.0, 24.0, 0.3),
        ]
    )
    assert set(env) == {"あ"}  # い は吸収され独立キーが残らない
    keys = env["あ"]
    assert keys[0] == (0, pytest.approx(0.0))
    assert keys[-1] == (24, pytest.approx(0.0))
    for frame, weight in keys:
        if 0 < frame < 24:
            assert weight > 0.0  # 内部に再アタック・閉口を挟まず連続保持


def test_normal_groups_not_shortened():
    # 通常長(保持に余裕)のグループは L-6 で変化しない(回帰ガード)。
    # あ[0,10]op0.5: L=10、available=7 >= a+r=4 で短縮なし。§4.9 の4点エンベロープのまま。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)])
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])
