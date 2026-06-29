"""先行準備(anticipation)・後行残し(release-lag)のテスト(lipsync.md §3/§4.10)。

無音区間の後に始まる母音は、口形を A_eff フレーム手前から緩やかに立ち上げ音符開始で保持値へ達する(窓全体の
ランプ)。後続が無音の母音は、音符終了まで保持し R_eff フレームかけて緩やかに閉じる(先行準備と対称)。
A_eff/R_eff は開き量比例 `half_up(anticipation_frames × clamp(open/open_cap, 0, 1))` を隣接無音長の1/2で
頭打ちした値で、開きが大きいほど長い余韻になる。時間軸先頭・両唇閉鎖直後・極短無音では先行しない(前区間を
侵食せず負フレームに出ない)ことを回帰ガードとして併せて確認する。
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


def test_anticipation_after_silence():
    # 無音[0,10]・あ[10,20]op0.5、anticipation_frames=1。開き量比 0.5/0.8=0.625、A_eff=half_up(1×0.625)=1、
    # floor(10/2)=5 で頭打ちなし。立ち上がりは無音側へ1フレーム入り、(9,0)→音符開始(10,0.5)で保持値へ達する。
    env = _envelope(
        [MouthEvent(MouthShape.SILENCE, 0.0, 10.0), MouthEvent(MouthShape.A, 10.0, 20.0, 0.5)]
    )
    _approx_envelope(env["あ"], [(9, 0.0), (10, 0.5), (18, 0.5), (20, 0.0)])


def test_anticipation_auto_shortened_by_short_silence():
    # 無音[0,2]・あ[2,12]op0.5、anticipation_frames=2。A_eff=half_up(2×0.625)=1、floor(2/2)=1 で頭打ち1。
    # 立ち上がりは (1,0)→音符開始(2,0.5)。前区間長で頭打ちされ無音を侵食しすぎない。
    p = GenerationParams(anticipation_frames=2)
    env = _envelope(
        [MouthEvent(MouthShape.SILENCE, 0.0, 2.0), MouthEvent(MouthShape.A, 2.0, 12.0, 0.5)], p
    )
    _approx_envelope(env["あ"], [(1, 0.0), (2, 0.5), (10, 0.5), (12, 0.0)])


def test_release_lag_before_silence():
    # 後続が無音の母音は急に閉じず余韻を残す(先行準備と対称)。あ[0,10]op0.5・無音[10,20]、
    # anticipation_frames=3。R_eff=half_up(3×0.625)=2、floor(10/2)=5 で頭打ちなし。音符終了10まで保持し、
    # その後2フレームかけて (10,0.5)→(12,0) と緩やかに閉じる(先頭は時間軸先頭なので通常アタック)。
    p = GenerationParams(anticipation_frames=3)
    env = _envelope(
        [MouthEvent(MouthShape.A, 0.0, 10.0, 0.5), MouthEvent(MouthShape.SILENCE, 10.0, 20.0)], p
    )
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (10, 0.5), (12, 0.0)])


def test_lead_lag_scales_with_opening():
    # 先行/後行量は開き量に比例する。anticipation_frames=4、長い無音[0,20]の後の母音で比較。
    # 開き0.8: 比1.0、A_eff=half_up(4×1.0)=4 → 立ち上がり開始フレーム 20−4=16。
    big = _envelope(
        [MouthEvent(MouthShape.SILENCE, 0.0, 20.0), MouthEvent(MouthShape.A, 20.0, 40.0, 0.8)],
        GenerationParams(anticipation_frames=4),
    )
    # 開き0.2: 比0.2/0.8=0.25、A_eff=half_up(4×0.25)=1 → 立ち上がり開始フレーム 20−1=19。
    small = _envelope(
        [MouthEvent(MouthShape.SILENCE, 0.0, 20.0), MouthEvent(MouthShape.A, 20.0, 40.0, 0.2)],
        GenerationParams(anticipation_frames=4),
    )
    assert big["あ"][0][0] == 16
    assert small["あ"][0][0] == 19  # 開きが小さいほど先行は短い(開始フレームが後ろ)


def test_no_anticipation_at_timeline_start():
    # 時間軸先頭の母音は直前イベントが無いので先行しない(A_eff=0、負フレームに出ない)。
    env = _envelope([MouthEvent(MouthShape.A, 0.0, 10.0, 0.5)])
    _approx_envelope(env["あ"], [(0, 0.0), (2, 0.5), (8, 0.5), (10, 0.0)])


def test_no_anticipation_after_bilabial():
    # 両唇閉鎖の直後の母音は閉口完成を優先し先行しない(A_eff=0)。両唇閉鎖は専用キーを持たず閉口は規約で表す。
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
