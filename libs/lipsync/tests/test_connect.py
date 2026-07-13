"""同母音連結のテスト。

連続する同一母音イベントを1つの保持区間へ連結し、先頭にのみアタック・末尾にのみリリースを置き、
内部境界に再アタック・閉口を入れず、各小区間の開き量を中央の強弱節点として残して節点間を線形に
変化させることを既知値で検証する。異なる母音が隣接しても同一保持区間へ連結されず、各母音が別モーフの
identity を保つことを回帰ガードで確認する(境界の具体的なキー形は協調調音が決めるため、ここでは
連結されない不変条件のみを見る。現行実装で既に成立するため印を付けない)。
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
    for (_, aw), (_, ew) in zip(actual, expected):
        assert aw == pytest.approx(ew)


def test_two_same_vowels_merge_into_one_envelope():
    # あ[0,10]op0.4・あ[10,20]op0.7 を連結。hold_start=2/hold_end=18、mid=5,15、w=0.4,0.7。
    # 先頭アタック・末尾リリースのみ、内部境界に閉口/再アタックなし。
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.4), MouthEvent(MouthShape.A, 10.0, 20.0, 0.7)]
    )
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"], [(0, 0.0), (2, 0.4), (5, 0.4), (15, 0.7), (18, 0.7), (20, 0.0)]
    )


def test_no_reattack_or_close_at_internal_boundary():
    # 連結区間の内部境界フレーム10に0キーや再アタックを置かない。端点以外は全て非ゼロ重み。
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.4), MouthEvent(MouthShape.A, 10.0, 20.0, 0.7)]
    )
    keys = env["あ"]
    frames = [f for f, _ in keys]
    assert 10 not in frames
    for frame, weight in keys:
        if 0 < frame < 20:
            assert weight > 0.0


def test_same_vowel_merge_strong_weak_nodes():
    # い[0,10]op0.4・い[10,20]op0.6 を連結(純母音 い 単独)。先頭アタック1回・末尾リリース1回、
    # 内部に開き量差の強弱節点(中央 0.4/0.6)。境界で再アタックや閉口を入れない。
    env = _envelope(
        [MouthEvent(MouthShape.I, 0.0, 10.0, 0.4), MouthEvent(MouthShape.I, 10.0, 20.0, 0.6)]
    )
    assert set(env) == {"い"}
    _approx_envelope(
        env["い"], [(0, 0.0), (2, 0.4), (5, 0.4), (15, 0.6), (18, 0.6), (20, 0.0)]
    )


def test_three_same_vowels_single_attack_release():
    # あ×3(各長12)を連結。開き量は cap 未満の 0.3/0.5/0.7 で、hold_start=2/hold_end=34、
    # mid=6,18,30、w=0.3,0.5,0.7。先頭アタック1回・末尾リリース1回、内部に強弱節点3つ。
    # プラトー長32は伸び表現の閾値を超えるため、連結の検証へ分離する目的で揺らぎを無効化する。
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 12.0, 0.3),
            MouthEvent(MouthShape.A, 12.0, 24.0, 0.5),
            MouthEvent(MouthShape.A, 24.0, 36.0, 0.7),
        ],
        GenerationParams(vibrato_amp=0.0),
    )
    _approx_envelope(
        env["あ"],
        [(0, 0.0), (2, 0.3), (6, 0.3), (18, 0.5), (30, 0.7), (34, 0.7), (36, 0.0)],
    )


def test_same_vowel_different_consonant_merges_and_aux_fades():
    # 同じ母音なら子音種別が違っても1つの連続保持に連結する。あ(ROUNDED)[0,10]→あ(NONE)[10,20] は
    # 連結し、主モーフ「あ」は途中で閉じず連続、補助モーフ「う」はグループ内で 0 へフェード(残留しない)。
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 10.0, 0.5, ConsonantClass.ROUNDED),
            MouthEvent(MouthShape.A, 10.0, 20.0, 0.5, ConsonantClass.NONE),
        ]
    )
    assert set(env) == {"あ", "う"}
    assert all(w > 0.0 for f, w in env["あ"] if 0 < f < 20)  # 主モーフは連続(途中閉口なし)
    assert env["う"][-1][1] == pytest.approx(0.0)  # 補助はグループ端で 0 へ(残留しない)
    assert max(w for _, w in env["う"]) == pytest.approx(0.15)  # ROUNDED 部の丸め量


def test_same_vowel_same_profile_different_consonant_merges():
    # SPREAD×い と NONE×い はどちらも純い(SPREAD の補助 い は主モーフと同名で寄与しない)なので連結し、
    # 途中で閉口しない(しー の途中閉口を防ぐ)。
    env = _envelope(
        [
            MouthEvent(MouthShape.I, 0.0, 4.0, 0.5, ConsonantClass.NONE),
            MouthEvent(MouthShape.I, 4.0, 8.0, 0.5, ConsonantClass.SPREAD),
        ]
    )
    assert set(env) == {"い"}
    assert all(w > 0.0 for f, w in env["い"] if 0 < f < 8)


def test_short_same_vowel_different_consonant_no_mid_flicker():
    # 短い同母音モーラが子音種別違いで連続しても、連結して1つの保持にし途中で閉じない(フリッカー防止)。
    # う(SPREAD)[0,4]→う(NONE)[4,8] は別々に短く三角形化すると境界で開閉のちらつきになるが、連結すれば
    # 主モーフ「う」は連続し、SPREAD の補助「い」だけがフェードする。
    env = _envelope(
        [
            MouthEvent(MouthShape.U, 0.0, 4.0, 0.5, ConsonantClass.SPREAD),
            MouthEvent(MouthShape.U, 4.0, 8.0, 0.5, ConsonantClass.NONE),
        ]
    )
    assert "う" in env and "い" in env
    assert all(w > 0.0 for f, w in env["う"] if 0 < f < 8)  # 主モーフ連続=途中閉口・ちらつきなし
    assert env["い"][-1][1] == pytest.approx(0.0)  # 補助はフェードして残留しない


@pytest.mark.xfail(reason="impl pending: ApertureClass", strict=True)
def test_same_vowel_merge_preserves_per_segment_aperture_decay():
    # 同じ母音(あ)で開き量は同じ(0.4)だが ApertureClass が異なる(FIRM_CLOSURE→NONE)2小区間を連結。
    # 各小区間の強弱節点は開口減衰まで適用した最終重みになる: 前0.4×0.75=0.3、後0.4×1.0=0.4。
    # 1つの連続保持区間へ統合され、内部境界(10)に再アタック・閉口を入れない。
    env = _envelope(
        [
            MouthEvent(
                MouthShape.A, 0.0, 10.0, 0.4, ConsonantClass.NONE,
                lipsync.ApertureClass.FIRM_CLOSURE,
            ),
            MouthEvent(
                MouthShape.A, 10.0, 20.0, 0.4, ConsonantClass.NONE, lipsync.ApertureClass.NONE
            ),
        ]
    )
    assert set(env) == {"あ"}
    _approx_envelope(
        env["あ"], [(0, 0.0), (2, 0.3), (5, 0.3), (15, 0.4), (18, 0.4), (20, 0.0)]
    )


def test_different_adjacent_vowels_not_merged():
    # あ[0,10]・う[10,20] は母音が異なるので1保持区間へ連結しない。あ は あ区間側で、う は う区間側で
    # それぞれ別モーフの identity を保つ(境界の具体形=閉口か協調調音かは協調調音が決めるため見ない)。
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)]
    )
    # あ は あ区間のみ、う は う区間のみで、両母音のモーフ集合が混ざらず別々に立つ(純母音)。
    assert set(env) == {"あ", "う"}
    # あ は前半(境界10より前)で保持値0.5に達し、う は後半(境界10より後)で0.5に達する。
    assert any(f < 10 and w == pytest.approx(0.5) for f, w in env["あ"])
    assert any(f > 10 and w == pytest.approx(0.5) for f, w in env["う"])
