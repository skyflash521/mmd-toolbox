"""L-5 先行準備(anticipation)のテスト(implementation-plan.md §4.10, lipsync.md §3/§4)。

無音区間の後に始まる母音グループの先頭アタックを A_eff=min(anticipation_frames, floor(前区間長/2))
だけ前倒しすること、前区間長で自動短縮されること、時間軸先頭・両唇閉鎖直後・極短無音では先行しない
(前区間を侵食せず負フレームに出ない)ことを既知値で検証する。先行が起きる新挙動は xfail で印を付け、
先行が起きない非適用ケースは現行実装でも成立するため印を付けない。
"""

import pytest

import lipsync
from lipsync import GenerationParams, MouthEvent, MouthShape

_L5 = pytest.mark.xfail(reason="impl pending: L-5 先行準備")


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


@_L5
def test_anticipation_after_silence():
    # 無音[0,10]・あ[10,20]op0.5、anticipation_frames=1。前区間長10で floor(10/2)=5、A_eff=min(1,5)=1。
    # 先頭アタックが (10,0)・(12,0.5) から (9,0)・(11,0.5) へ1フレーム前倒し。
    env = _envelope(
        [MouthEvent(MouthShape.SILENCE, 0.0, 10.0), MouthEvent(MouthShape.A, 10.0, 20.0, 0.5)]
    )
    _approx_envelope(env["あ"], [(9, 0.0), (11, 0.5), (18, 0.5), (20, 0.0)])


@_L5
def test_anticipation_auto_shortened_by_short_silence():
    # 無音[0,2]・あ[2,12]op0.5、anticipation_frames=2。前区間長2で floor(2/2)=1、A_eff=min(2,1)=1。
    # 設定は2でも前区間長で1へ自動短縮され、アタックは (2,0)・(4,0.5) から (1,0)・(3,0.5) へ1だけ前倒し。
    p = GenerationParams(anticipation_frames=2)
    env = _envelope(
        [MouthEvent(MouthShape.SILENCE, 0.0, 2.0), MouthEvent(MouthShape.A, 2.0, 12.0, 0.5)], p
    )
    _approx_envelope(env["あ"], [(1, 0.0), (3, 0.5), (10, 0.5), (12, 0.0)])


def test_no_anticipation_at_timeline_start():
    # 時間軸先頭の母音は直前イベントが無いので先行しない(A_eff=0、負フレームに出ない)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)])
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])


def test_no_anticipation_after_bilabial():
    # 両唇閉鎖の直後の母音は閉口完成を優先し先行しない(A_eff=0)。両唇閉鎖の閉口キーは L-8 で置く。
    env = _envelope(
        [
            MouthEvent(MouthShape.SILENCE, 0.0, 8.0),
            MouthEvent(MouthShape.BILABIAL, 8.0, 10.0),
            MouthEvent(MouthShape.A, 10.0, 20.0, 0.5),
        ]
    )
    _approx_envelope(env["あ"], [(10, 0.0), (12, 0.5), (18, 0.5), (20, 0.0)])


def test_no_anticipation_when_silence_too_short():
    # 無音[0,1]は floor(1/2)=0 で A_eff=0。設定 anticipation_frames=2 でも前区間を侵食せず先行しない。
    p = GenerationParams(anticipation_frames=2)
    env = _envelope(
        [MouthEvent(MouthShape.SILENCE, 0.0, 1.0), MouthEvent(MouthShape.A, 1.0, 11.0, 0.5)], p
    )
    _approx_envelope(env["あ"], [(1, 0.0), (3, 0.5), (9, 0.5), (11, 0.0)])
