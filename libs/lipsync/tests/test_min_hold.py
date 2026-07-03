"""最小保持・競合短縮・三角形短区間のテスト(lipsync.md §3/§4.4/§4.9)。

母音グループを長さで3分類することを既知値で検証する: L<triangle_min は吸収/除去、triangle_min≤L<min_hold+2 は
三角形(中央に保持値ピーク1点)で残す、それ以上は通常形状で競合短縮(アタック+リリースが入らない区間は実効
アタック/リリースを詰めて保持を最優先で残す)。三角形は極短母音を吸収せず一瞬開いて見せ、発声中の閉口を防ぐ。
先行・協調調音の干渉を避けるため、競合短縮は時間軸先頭の単一母音、吸収は同一母音アンカーのフィクスチャで確認し、
通常長で不変であることは回帰ガードで併せて確認する。
"""

import pytest

import lipsync
from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape
from lipsync import generate


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
    # あ[0,1.5]op0.5、既定。L=1.5<triangle_min=2 で吸収対象。母音アンカー無し(単独・時間軸端)→除去。
    # 独立キーを出さず閉口(出力は空)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 1.5, 0.5)])
    assert env == {}


def test_short_vowel_triangle_peak():
    # あ[0,4]op0.5、既定。L=4 は triangle_min(2)≤L<min_hold+2(5) で三角形。母音アンカー無しでも吸収せず、
    # 中央4/2=2に保持値ピーク1点(開始0→中央0.5→終了0)で残す(発声中の閉口を防ぐ)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 4.0, 0.5)])
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (4, 0.0)])


def test_short_vowel_absorbed_between_same_vowels():
    # あ[0,10]op0.8・い[10,11]op0.5・あ[11,21]op0.3、既定。い は L=1<triangle_min=2 で吸収対象。
    # 吸収先タイブレーク=開き量大: 前あ0.8 > 後あ0.3 → 前へ吸収。前あが span を貫いて後あと同母音連結
    # になり、い の独立キーは消える。結果は あ 一系列のみ・内部に閉口0キー無し。
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.8),
            MouthEvent(MouthShape.I, 10.0, 11.0, 0.5),
            MouthEvent(MouthShape.A, 11.0, 21.0, 0.3),
        ]
    )
    assert set(env) == {"あ"}  # い は吸収され独立キーが残らない
    keys = env["あ"]
    assert keys[0] == (0, pytest.approx(0.0))
    assert keys[-1] == (21, pytest.approx(0.0))
    for frame, weight in keys:
        if 0 < frame < 21:
            assert weight > 0.0  # 内部に再アタック・閉口を挟まず連続保持


def test_normal_groups_not_shortened():
    # 通常長(保持に余裕)のグループは最小保持・競合短縮で変化しない(回帰ガード)。
    # あ[0,10]op0.5: L=10、available=7 >= a+r=4 で短縮なし。形状の4点エンベロープのまま。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)])
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])


def test_merged_group_reclassified_to_normal_after_absorption():
    # 同母音の間に挟まる極短の別母音が吸収され同母音が連結したら、結合後の最終長で短区間分類をやり直し、
    # 三角形フラグを引きずらない(三角形のままだと三角形経路で後続イベントのプロファイル・補助フェードが落ちる)。
    groups = generate._normalize_groups(
        [
            MouthEvent(MouthShape.A, 0.0, 3.0, 0.5, ConsonantClass.ROUNDED),
            MouthEvent(MouthShape.I, 3.0, 4.0, 0.5),  # 極短(L=1<triangle_min)→ 吸収
            MouthEvent(MouthShape.A, 4.0, 7.0, 0.5, ConsonantClass.NONE),
        ],
        GenerationParams(),
    )
    assert len(groups) == 1
    g = groups[0]
    assert (g.start, g.end) == (0.0, 7.0)
    assert g.triangle is False  # 結合後 7f は通常長(三角形に再分類しない)
    assert g.attack > 0.0 and g.release > 0.0  # 通常グループはアタック/リリースを持つ


def test_consonant_at_uses_nearest_for_absorbed_gap_and_ends():
    # _consonant_at は区間内ならそのイベント、区間外(吸収で除去された内部ギャップ・延長された端の外側)なら
    # 最近傍イベントの子音種別を返す(常に先頭へ倒さない)。
    events = [
        MouthEvent(MouthShape.A, 0.0, 30.0, 0.5, ConsonantClass.ROUNDED),
        MouthEvent(MouthShape.A, 31.0, 60.0, 0.5, ConsonantClass.NONE),
    ]
    assert generate._consonant_at(events, 15.0) is ConsonantClass.ROUNDED  # 区間内(先頭)
    assert generate._consonant_at(events, 45.0) is ConsonantClass.NONE  # 区間内(後続)
    assert generate._consonant_at(events, 30.1) is ConsonantClass.ROUNDED  # ギャップ・30側が近い
    assert generate._consonant_at(events, 30.9) is ConsonantClass.NONE  # ギャップ・31側が近い
    assert generate._consonant_at(events, -5.0) is ConsonantClass.ROUNDED  # 先頭外側
    assert generate._consonant_at(events, 70.0) is ConsonantClass.NONE  # 末尾外側


def _total_open(env, frame):
    """各モーフを線形補間して総開き量(全モーフ重みの和)を返す(有声区間内の閉口検出用)。"""
    def interp(keys, f):
        if not keys:
            return 0.0
        if f <= keys[0][0]:
            return keys[0][1]
        if f >= keys[-1][0]:
            return keys[-1][1]
        for (f0, v0), (f1, v1) in zip(keys, keys[1:]):
            if f0 <= f <= f1:
                return v0 if f1 == f0 else v0 + (v1 - v0) * (f - f0) / (f1 - f0)
        return keys[-1][1]
    return sum(interp(keys, frame) for keys in env.values())


def test_adjacent_triangles_connect_without_closure():
    # 短い う(三角形)→ 短い え(三角形)が直接隣接。協調調音で繋ぎ、有声区間内で口を閉じない
    # (三角形どうしが各々0端点で閉じる開閉ちらつき=フリッカーを防ぐ)。曲頭・曲末だけ閉口。
    env = _envelope(
        [MouthEvent(MouthShape.U, 0.0, 4.0, 0.5), MouthEvent(MouthShape.E, 4.0, 8.0, 0.5)]
    )
    for f in range(1, 8):
        assert _total_open(env, f) > 0.05  # 内部(境界4含む)は閉口しない
    assert _total_open(env, 0) == pytest.approx(0.0)  # 曲頭は閉口
    assert _total_open(env, 8) == pytest.approx(0.0)  # 曲末は閉口


def test_triangle_closes_on_closure_side_only():
    # 三角形が片側 SILENCE・片側 母音。SILENCE 側は0で閉じ、母音側は協調調音で繋ぐ。
    # SILENCE[0,4]・う[4,8]三角形・え[8,16]通常。う は SILENCE 側(4)で閉口、え 側は閉じない。
    env = _envelope(
        [
            MouthEvent(MouthShape.SILENCE, 0.0, 4.0),
            MouthEvent(MouthShape.U, 4.0, 8.0, 0.5),
            MouthEvent(MouthShape.E, 8.0, 16.0, 0.5),
        ]
    )
    assert _total_open(env, 4) == pytest.approx(0.0)  # SILENCE 側は閉口
    for f in range(6, 15):
        assert _total_open(env, f) > 0.05  # う→え は協調調音で閉口しない


def test_triangle_legato_triangle_bridged_without_closure():
    # 三角形→レガート間隙→三角形(同じ「あ」の繰り返しが間隙で区切られ短く三角形化)。谷で橋渡しし、
    # 有声区間内で閉口しない(間隙の谷でも0に落ちない)。
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 4.0, 0.5),
            MouthEvent(MouthShape.LEGATO_GAP, 4.0, 6.0),
            MouthEvent(MouthShape.A, 6.0, 10.0, 0.5),
        ]
    )
    for f in range(1, 10):
        assert _total_open(env, f) > 0.05  # 間隙の谷を含め内部は閉口しない
    assert _total_open(env, 0) == pytest.approx(0.0)
    assert _total_open(env, 10) == pytest.approx(0.0)
