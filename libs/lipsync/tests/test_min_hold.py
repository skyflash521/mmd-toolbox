"""最小保持・競合短縮・三角形短区間のテスト。

母音グループを長さで3分類することを既知値で検証する: L<triangle_min は吸収/除去、triangle_min≤L<min_hold+2 は
三角形(中央に保持値ピーク1点)で残す、それ以上は通常形状で競合短縮(アタック+リリースが入らない区間は実効
アタック/リリースを詰めて保持を最優先で残す)。三角形は極短母音を吸収せず一瞬開いて見せ、発声中の閉口を防ぐ。
先行・協調調音の干渉を避けるため、競合短縮は時間軸先頭の単一母音、吸収は同一母音アンカーのフィクスチャで確認し、
通常長で不変であることは回帰ガードで併せて確認する。
"""

import pytest

import lipsync
from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape, generate


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
    for (_, aw), (_, ew) in zip(actual, expected, strict=True):
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


def test_triangle_multi_segment_uses_length_weighted_average_no_valley():
    # あ[0,1]op0.4(ApertureClass.NONE)・あ[1,3]op0.8(ApertureClass.FIRM_CLOSURE)、既定。
    # 合体後 L=3 は triangle_min(2)≤L<min_hold+2(5) で三角形。内部小区間数(n=2)によらず単一
    # ピークへ平滑化し、モーラ境界の谷(内部境界のApertureClassがFIRM_CLOSUREでNONEでない)は
    # 適用しない。ピーク重みは各小区間の長さによる長さ加重平均: (1*0.4+2*(0.8*0.75))/3
    # =(0.4+1.2)/3=0.53333...(単純平均0.5とは異なる値)。ピーク位置は中央(0+3)/2=1.5(四捨五入2)。
    events = [
        MouthEvent(MouthShape.A, 0.0, 1.0, 0.4, ConsonantClass.NONE, lipsync.ApertureClass.NONE),
        MouthEvent(
            MouthShape.A, 1.0, 3.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE
        ),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5333333333333333), (3, 0.0)])


def test_single_sided_absorption_reclassifies_without_remerge():
    # 三角形長(4f)の あ の末尾に、別母音の短区間(1.5f<triangle_min=2)が直接隣接し、他に吸収先を
    # 持たず あ へ吸収される(同一口形との再連結を伴わない片側吸収)。吸収の結果 あ の実効長が
    # 4+1.5=5.5f になり三角形の上限(min_hold+2=5)以上になるため、吸収後の再分類により
    # 通常グループとして扱われ(三角形の単一ピークへ丸めず)、通常グループとしてアタック/リリースを
    # 持つ。
    groups = generate._normalize_groups(
        [
            MouthEvent(MouthShape.A, 0.0, 4.0, 0.5),
            MouthEvent(MouthShape.I, 4.0, 5.5, 0.5),
        ],
        GenerationParams(),
    )
    assert len(groups) == 1
    g = groups[0]
    assert (g.start, g.end) == (0.0, 5.5)
    assert g.triangle is False  # 片側吸収だけで通常長に達したので三角形分類を引きずらない
    assert g.attack > 0.0 and g.release > 0.0


def test_absorbed_span_extends_effective_boundary_for_valley_and_midpoints():
    # あ[0,4.5]op0.8(FIRM_CLOSURE)→い[4.5,5.0](短区間、吸収対象)→あ[5.0,9.0]op0.8(FIRM_CLOSURE)。
    # い は前側の あ へ吸収され、あ の実効 end が5.0(元の4.5でなく吸収した0.5ぶん広がった位置)
    # になり、後側の あ と直接隣接して同一母音の連結で1グループへ再連結される。内部境界(実効
    # end=5.0)のApertureClassはFIRM_CLOSURE(後側あ自身のクラス)なので谷が生成される。谷の左肩は
    # mid1の量子化後フレームと一致、右肩はmid2と一致し、両者とも谷の肩の値と等しいため独立した
    # 肩キーは生じず、mid1・谷central・mid2の3キーだけが残る。
    events = [
        MouthEvent(
            MouthShape.A, 0.0, 4.5, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE
        ),
        MouthEvent(MouthShape.I, 4.5, 5.0, 0.5),
        MouthEvent(
            MouthShape.A, 5.0, 9.0, 0.8, ConsonantClass.NONE, lipsync.ApertureClass.FIRM_CLOSURE
        ),
    ]
    env = _envelope(events)
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"],
        [
            (0, 0.0),
            (2, 0.6),
            (3, 0.6),  # mid1(2.5 を四捨五入)と谷の左肩(b-hw=3.0)が一致
            (5, 0.45),  # 谷の中央(b=5.0)
            (7, 0.6),  # mid2(7.0)と谷の右肩(b+hw=7.0)が一致
            (9, 0.0),
        ],
    )


def test_triangle_multi_segment_coarticulation_uses_group_length_not_segment_length():
    # あ[0,2]op0.4→あ[2,3]op0.6(内部小区間2つ、合体長3で三角形)→い[3,13]op0.5(通常長)。
    # 三角形のピーク重みは長さ加重平均 (2*0.4+1*0.6)/3=0.466667。境界(t=3)の協調調音の遷移長は、
    # 「短い側区間長」にこの三角形グループ全体の長さ(3)を使う: shorter=min(3,10)=3,
    # T=clamp(coartic_overlap_max=2,1,3/2=1.5)=1.5(四捨五入2)。遷移始端 b-T/2=2 には、次口形
    # 「い」の重み0.0(片方のプロファイルに無いモーフは重み0.0とする規則によるアンカー)のキーが立つ。
    # 末尾小区間(長さ1)だけを「区間長」に使う誤った実装だと shorter=min(1,10)=1、
    # T=clamp(2,1,0.5)=1(四捨五入1)になり、遷移始端は b-0.5→四捨五入3(境界の中間口形キーと
    # 量子化衝突し統合される)になる。あ 側は三角形のピーク自体が中央(1.5→四捨五入2)に立つため、
    # どちらの実装でもフレーム2に「あ」のキーが存在してしまい判別に使えない。「い」のフレーム2の
    # 有無で区別する。
    events = [
        MouthEvent(MouthShape.A, 0.0, 2.0, 0.4),
        MouthEvent(MouthShape.A, 2.0, 3.0, 0.6),
        MouthEvent(MouthShape.I, 3.0, 13.0, 0.5),
    ]
    env = _envelope(events)
    assert 2 in dict(env["い"])  # 正しい遷移長(グループ全長基準)なら い 側のフレーム2に境界側保持点が立つ


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
        for (f0, v0), (f1, v1) in zip(keys, keys[1:], strict=False):
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


def test_absorb_winner_uses_aperture_decayed_final_weight():
    # _absorb_winner(_opening 経由)は開口減衰(ApertureClass)まで適用した最終重みの最大値で
    # 吸収先を選ぶ。前あ op=0.6(ApertureClass.NONE)→最終重み0.6。
    # 後う op=0.65(ApertureClass.FIRM_CLOSURE=0.75)→最終重み0.65×0.75=0.4875。raw な
    # open_amount だけを比較する誤実装なら後(0.65)が勝ち後方(う)へ延びるが、正しくは
    # 最終重みの大きい前(あ)が勝ち前方へ延びる。
    groups = generate._normalize_groups(
        [
            MouthEvent(
                MouthShape.A, 0.0, 10.0, 0.6, ConsonantClass.NONE, lipsync.ApertureClass.NONE
            ),
            MouthEvent(MouthShape.I, 10.0, 11.0, 0.5),
            MouthEvent(
                MouthShape.U, 11.0, 21.0, 0.65, ConsonantClass.NONE,
                lipsync.ApertureClass.FIRM_CLOSURE,
            ),
        ],
        GenerationParams(),
    )
    assert len(groups) == 2
    a_group = next(g for g in groups if g.shape is MouthShape.A)
    u_group = next(g for g in groups if g.shape is MouthShape.U)
    assert (a_group.start, a_group.end) == (0.0, 11.0)  # 前(あ)が勝ち い の span を吸収
    assert (u_group.start, u_group.end) == (11.0, 21.0)  # 後(う)は自身の区間のまま


def test_absorb_winner_uses_consonant_modulated_final_weight():
    # 吸収先の判定は ConsonantClass の唇形変調(補助モーフ)まで含めた最終重みの最大値で行う。
    # 前あ op=0.7(ConsonantClass.NONE)→最終重み0.7(補助なし・比例縮小非発動)。
    # 後え op=0.75(ConsonantClass.ROUNDED)→有効プロファイル{え:1.0,う:0.3}(え は ROUNDED の
    # 補助対象=う/お ではないので二重計上回避の対象外)、raw={え:0.75,う:0.225}、総量0.975>cap=0.8
    # → factor=0.8/0.975 → 最終重み(主モーフ)=0.75×0.8/0.975≈0.6154。raw な open_amount だけを
    # 比較する誤実装なら後(0.75)が勝つが、正しくは比例縮小後の最終重みが大きい前(あ、0.7)が勝つ。
    groups = generate._normalize_groups(
        [
            MouthEvent(
                MouthShape.A, 0.0, 10.0, 0.7, ConsonantClass.NONE, lipsync.ApertureClass.NONE
            ),
            MouthEvent(MouthShape.I, 10.0, 11.0, 0.5),
            MouthEvent(
                MouthShape.E, 11.0, 21.0, 0.75, ConsonantClass.ROUNDED,
                lipsync.ApertureClass.NONE,
            ),
        ],
        GenerationParams(),
    )
    assert len(groups) == 2
    a_group = next(g for g in groups if g.shape is MouthShape.A)
    e_group = next(g for g in groups if g.shape is MouthShape.E)
    assert (a_group.start, a_group.end) == (0.0, 11.0)  # 前(あ)が勝ち い の span を吸収
    assert (e_group.start, e_group.end) == (11.0, 21.0)  # 後(え)は自身の区間のまま


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
