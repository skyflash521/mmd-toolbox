"""L-3 同母音連結のテスト(implementation-plan.md §4.2, lipsync.md §3/§4)。

連続する同一母音イベントを1つの保持区間へ連結し、先頭にのみアタック・末尾にのみリリースを置き、
内部境界に再アタック・閉口を入れず、各小区間の開き量を中央の強弱節点として残して節点間を線形に
変化させることを既知値で検証する。異なる母音が隣接する場合は連結せず各々が §4.9 の単一区間形状を保つ
(回帰ガード。現行 L-2 実装で既に成立するため印を付けない)。
"""

import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape

_L3 = pytest.mark.xfail(reason="impl pending: L-3 同母音連結")


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


@_L3
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


@_L3
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


@_L3
def test_multi_morph_same_vowel_merge():
    # い[0,10]op0.4・い[10,20]op0.6 を連結。い:{あ:0.1,い:1.0}。主・補助モーフとも連結エンベロープ。
    env = _envelope(
        [MouthEvent(MouthShape.I, 0.0, 10.0, 0.4), MouthEvent(MouthShape.I, 10.0, 20.0, 0.6)]
    )
    assert set(env) == {"あ", "い"}
    _approx_envelope(
        env["い"], [(0, 0.0), (2, 0.4), (5, 0.4), (15, 0.6), (18, 0.6), (20, 0.0)]
    )
    _approx_envelope(
        env["あ"], [(0, 0.0), (2, 0.04), (5, 0.04), (15, 0.06), (18, 0.06), (20, 0.0)]
    )


@_L3
def test_three_same_vowels_single_attack_release():
    # あ×3(各長12)を連結。hold_start=2/hold_end=34、mid=6,18,30、w=0.3,0.6,0.9。
    # 先頭アタック1回・末尾リリース1回、内部に強弱節点3つ。
    env = _envelope(
        [
            MouthEvent(MouthShape.A, 0.0, 12.0, 0.3),
            MouthEvent(MouthShape.A, 12.0, 24.0, 0.6),
            MouthEvent(MouthShape.A, 24.0, 36.0, 0.9),
        ]
    )
    _approx_envelope(
        env["あ"],
        [(0, 0.0), (2, 0.3), (6, 0.3), (18, 0.6), (30, 0.9), (34, 0.9), (36, 0.0)],
    )


def test_different_adjacent_vowels_not_merged():
    # あ[0,10]・う[10,20] は母音が異なるので連結しない。各々が §4.9 の単一区間4点エンベロープを保ち、
    # 境界フレーム10で あ は閉口0へ戻り う は0から開く(再アタック・閉口あり)。
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.U, 10.0, 20.0, 0.5)]
    )
    # あ は あ区間のみ、う は う区間のみで、両者のモーフ集合は重ならない。
    assert set(env) == {"あ", "う", "お"}
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])
    # う:{う:1.0,お:0.2}、hold=0.5 → 主 う:0.5・補助 お:0.1。境界10で開き始める単一区間エンベロープ。
    _approx_envelope(env["う"], [(10, 0.0), (12, 0.5), (18, 0.5), (20, 0.0)])
    _approx_envelope(env["お"], [(10, 0.0), (12, 0.1), (18, 0.1), (20, 0.0)])
