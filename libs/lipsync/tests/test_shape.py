"""形状(アタック・保持・リリース)のテスト。

各母音区間がエンベロープ(各出力モーフに (start,0.0)・(start+attack,w_m)・
(end-release,w_m)・(end,0.0) の4点)を出力し、短いアタックで保持値まで開き・区間の大半を
一定保持・短いリリースで閉口へ戻ることを既知値で検証する。保持中はフレーム単位で揺れない。
"""

import pytest

import lipsync
from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape


def _envelope(events, params=None):
    """イベント列を変換し、モーフ名→(frame, weight) の時間順リストの辞書を返す。"""
    params = params or GenerationParams()
    keys = lipsync.generate_morph_keys(events, params)
    by_morph: dict[str, list[tuple[int, float]]] = {}
    for k in keys:
        by_morph.setdefault(k.name, []).append((k.frame, k.weight))
    for name in by_morph:
        by_morph[name].sort()
    return by_morph


def _approx_envelope(actual, expected):
    """(frame, weight) リストを frame 完全一致・weight 近似で比較する。"""
    assert [f for f, _ in actual] == [f for f, _ in expected]
    for (_, aw), (_, ew) in zip(actual, expected, strict=True):
        assert aw == pytest.approx(ew)


def test_single_vowel_four_key_envelope():
    # あ [0,10] open=0.5、保持値 w=0.5、attack=2/release=2。
    # 4点: (0,0)・(2,0.5)・(8,0.5)・(10,0)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)])
    assert set(env) == {"あ"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])


def test_attack_release_frames_honored():
    # attack=3/release=1、あ [0,12] open=0.5 → (0,0)・(3,0.5)・(11,0.5)・(12,0)。
    p = GenerationParams(attack_frames=3, release_frames=1)
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 12.0, 0.5)], p)
    _approx_envelope(env["あ"], [(0, 0.0), (3, 0.5), (11, 0.5), (12, 0.0)])


def test_each_composed_morph_gets_own_envelope():
    # ROUNDED あ [0,10] open=0.5 → 合成 {あ:0.5, う:0.15}。各モーフが同じ attack/release で4点を持つ。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5, ConsonantClass.ROUNDED)])
    assert set(env) == {"あ", "う"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])
    _approx_envelope(env["う"], [(0, 0.0), (2, 0.15), (8, 0.15), (10, 0.0)])


def test_plateau_is_flat():
    # 保持区間の2キーは同一保持値で、フレーム単位の揺れが無い(プラトー=ピーク値で一定)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)])
    keys = env["あ"]
    _, attack_weight = keys[1]
    _, release_weight = keys[2]
    assert attack_weight == pytest.approx(release_weight)
    assert attack_weight == pytest.approx(0.5)


def test_boundary_zero_keys():
    # 孤立区間は閉口0から開き、閉口0へ閉じる: 先頭・末尾キーは weight 0.0、フレームは start/end。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)])
    keys = env["あ"]
    assert keys[0] == (0, pytest.approx(0.0))
    assert keys[-1] == (10, pytest.approx(0.0))


def test_envelope_respects_interval_start():
    # 非ゼロ開始 [10,20] open=0.5 → (10,0)・(12,0.5)・(18,0.5)・(20,0)。
    env = _envelope([MouthEvent(MouthShape.A, 10.0, 20.0, 0.5)])
    _approx_envelope(env["あ"], [(10, 0.0), (12, 0.5), (18, 0.5), (20, 0.0)])


def test_total_clamp_plateau_value():
    # ROUNDED あ open=1.0 は合成総量が cap 超で比例縮小。プラトー値はその縮小後の合成重みで一定。
    # raw={あ:0.8, う:0.24} 総量1.04>0.8 → factor=0.8/1.04。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 1.0, ConsonantClass.ROUNDED)])
    factor = 0.8 / 1.04
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.8 * factor), (8, 0.8 * factor), (10, 0.0)])
    assert set(env) == {"あ", "う"}
