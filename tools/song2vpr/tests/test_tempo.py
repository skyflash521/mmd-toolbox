"""song2vpr のテンポ・拍子の推定のテスト。

既知の拍で合成した信号から、BPM・拍位相・拍子の分子・最初の小節線が得られることと、指定値と
推定値の組み合わせが規則どおりになることを検証する。
"""

import numpy as np
import pytest

from song2vpr import tempo
from vocal_analysis import AudioPcm

SR = 22050


def _click_track(bpm, seconds=8.0, offset_sec=0.0, accent_every=None, sample_rate=SR):
    """既知の拍でクリックを並べた信号。accent_every を与えるとその周期の拍だけ強くする。"""
    n = int(seconds * sample_rate)
    samples = np.zeros(n, dtype=np.float32)
    period = 60.0 / bpm
    rng = np.random.default_rng(0)
    samples += rng.normal(0.0, 0.001, n).astype(np.float32)  # 無音区間を作らない程度の床
    index = 0
    while True:
        at = offset_sec + index * period
        start = int(at * sample_rate)
        if start >= n:
            break
        length = int(0.02 * sample_rate)
        strong = accent_every is not None and index % accent_every == 0
        envelope = np.exp(-np.arange(length) / (0.004 * sample_rate))
        click = envelope * (1.0 if strong else 0.4)
        samples[start:start + length] += click[:max(0, min(length, n - start))].astype(np.float32)
        index += 1
    return AudioPcm(samples=samples[:, None], sample_rate=sample_rate)


def _silence(seconds=8.0, sample_rate=SR):
    n = int(seconds * sample_rate)
    return AudioPcm(samples=np.zeros((n, 1), dtype=np.float32), sample_rate=sample_rate)


# --- BPM の推定 --------------------------------------------------------------


@pytest.mark.parametrize("bpm", [90.0, 120.0, 150.0])
def test_bpm_is_estimated_from_the_beats(bpm):
    result = tempo.estimate(_click_track(bpm))
    assert result.bpm == pytest.approx(bpm, rel=0.03)


def test_estimated_bpm_is_not_an_octave_off():
    """1/2 倍・2 倍のピークではなく、人が拍と感じる帯域を選ぶ。"""
    result = tempo.estimate(_click_track(120.0))
    assert 60.0 < result.bpm < 240.0


def test_a_fast_beat_is_not_taken_at_half_speed():
    """帯域の重みの中心から離れた速い拍を、半分のテンポとして採らない。

    この入力は相関だけ見れば正解の候補が半分の候補に勝つので、覆しているのは重みになる。重みは
    候補どうしのタイブレークなので、この優位を覆さない幅にする。狭い幅を落とす主張なので、幅の
    下限を押さえる。同じくオクターブ誤りを見る `test_estimated_bpm_is_not_an_octave_off` は入力が
    重みの中心にあり、正解の候補が重みで最も高く評価されるので、幅を狭める側では破れない。広げる側
    では破れるが、`test_a_slow_beat_is_not_taken_at_half_speed` の入力より緩いところで破れる。

    16 秒にするのは長さが判定条件のため。既定の 8 秒では狭い幅でも正解を採るので、落としたい幅が
    通ってしまう。
    """
    result = tempo.estimate(_click_track(190.0, seconds=16.0))
    assert result.bpm == pytest.approx(190.0, rel=0.03)


def test_a_slow_beat_is_not_taken_at_half_speed():
    """帯域の重みの中心から離れた遅い拍を、半分のテンポとして採らない。

    この入力は相関だけ見れば半分の候補の方が高く、正解を保っているのは中心へ寄せる重みになる。
    幅を広げるほどその働きが弱まり、広げすぎるとこの入力が半分へ倒れる。したがってこの主張は幅の
    上限を押さえ、`test_a_fast_beat_is_not_taken_at_half_speed` が押さえる下限と対になる。
    倍の周期には相関がほとんど無いので、倍へ倒れる側は幅を拘束しない。
    """
    result = tempo.estimate(_click_track(90.0, seconds=16.0))
    assert result.bpm == pytest.approx(90.0, rel=0.03)


def test_given_tempo_is_used_as_is():
    """--tempo 指定時はその値を曲全体で一定に使う(推定しない)。"""
    result = tempo.estimate(_click_track(90.0), tempo_bpm=140.0)
    assert result.bpm == 140.0
    assert not result.tempo_defaulted


# --- 拍位相と最初の小節線 ----------------------------------------------------


def test_beat_phase_follows_the_offset():
    """拍位相は拍の位置に追随する。

    包絡は窓の左端を時刻の基準にするので、観測される位置は拍そのものより窓長ぶん早い。この
    系統的なずれは拍の位置に依らないため、2つの音源の差で見る。
    """
    early = tempo.estimate(_click_track(120.0, offset_sec=0.1), tempo_bpm=120.0)
    late = tempo.estimate(_click_track(120.0, offset_sec=0.35), tempo_bpm=120.0)
    assert late.beat_offset_sec - early.beat_offset_sec == pytest.approx(0.25, abs=0.03)


def test_phase_is_estimated_even_when_the_tempo_is_given():
    """位相は小節線の位置だけを決めるので、テンポを指定した実行でも求める。"""
    aligned = tempo.estimate(_click_track(120.0, offset_sec=0.0), tempo_bpm=120.0)
    shifted = tempo.estimate(_click_track(120.0, offset_sec=0.25), tempo_bpm=120.0)
    assert shifted.beat_offset_sec != aligned.beat_offset_sec


def test_first_bar_is_placed_at_the_head_beat_of_the_bar():
    """最初の小節線は、小節の頭として選ばれた拍の位置に置く。"""
    result = tempo.estimate(_click_track(120.0, seconds=16.0, offset_sec=0.25, accent_every=4),
                            tempo_bpm=120.0)
    assert result.first_bar_sec == pytest.approx(result.beat_offset_sec, abs=1e-9)


# --- 拍子 --------------------------------------------------------------------


@pytest.mark.parametrize("numerator", [3, 4])
def test_numerator_is_estimated_from_the_accents(numerator):
    result = tempo.estimate(_click_track(120.0, seconds=16.0, accent_every=numerator))
    assert result.numerator == numerator


def test_denominator_is_a_quarter_note_when_not_given():
    result = tempo.estimate(_click_track(120.0))
    assert result.denominator == 4


def test_given_time_signature_is_used_as_is():
    result = tempo.estimate(_click_track(120.0), time_signature=(6, 8))
    assert (result.numerator, result.denominator) == (6, 8)


# --- 周期が得られない入力 ----------------------------------------------------


def test_silence_falls_back_to_the_default_tempo():
    """周期を取り出せない入力では既定へ倒し、そのことを報告する。"""
    result = tempo.estimate(_silence())
    assert result.bpm == 120.0
    assert (result.numerator, result.denominator) == (4, 4)
    assert result.first_bar_sec == 0.0
    assert result.tempo_defaulted
    assert result.time_signature_defaulted


def test_default_keeps_the_given_time_signature():
    """テンポが既定へ倒れても、指定された拍子はそのまま使う(倒したことにならない)。"""
    result = tempo.estimate(_silence(), time_signature=(3, 4))
    assert result.bpm == 120.0
    assert (result.numerator, result.denominator) == (3, 4)
    assert result.tempo_defaulted
    assert not result.time_signature_defaulted


def test_given_tempo_is_not_defaulted_on_silence():
    """テンポを指定していれば、周期が取れなくても既定へ倒さない。"""
    result = tempo.estimate(_silence(), tempo_bpm=100.0)
    assert result.bpm == 100.0
    assert not result.tempo_defaulted


def test_time_signature_can_be_defaulted_while_the_tempo_is_not():
    """テンポを指定した実行でも、拍子の判定材料が得られなければ拍子だけ既定へ倒す。"""
    result = tempo.estimate(_silence(), tempo_bpm=100.0)
    assert not result.tempo_defaulted
    assert result.time_signature_defaulted
    assert (result.numerator, result.denominator) == (4, 4)


def test_estimated_time_signature_is_not_reported_as_defaulted():
    result = tempo.estimate(_click_track(120.0, seconds=16.0, accent_every=3))
    assert result.numerator == 3
    assert not result.time_signature_defaulted


# --- 秒から tick への変換 ----------------------------------------------------


def test_tick_conversion_starts_at_the_input_origin():
    """入力音声の 0 秒が tick の 0(小節線に合わせて音符をずらさない)。"""
    result = tempo.estimate(_click_track(120.0))
    assert result.to_tick(0.0) == 0


def test_tick_conversion_follows_the_adopted_tempo():
    """120 BPM なら四分音符1つ(0.5秒)が分解能ちょうどになる。"""
    result = tempo.estimate(_click_track(120.0), tempo_bpm=120.0)
    assert result.to_tick(0.5) == result.resolution


def test_tick_conversion_is_monotonic():
    result = tempo.estimate(_click_track(120.0), tempo_bpm=120.0)
    ticks = [result.to_tick(t) for t in (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)]
    assert ticks == sorted(ticks)
    assert len(set(ticks)) == len(ticks)


def test_adopted_tempo_is_representable_in_the_output_format():
    """採用テンポは形式が格納できる粒度へ丸めた値(出力と診断が食い違わないようにする)。

    形式はテンポを BPM の 100 倍の整数で持つので、採用値もその粒度に載る必要がある。
    """
    result = tempo.estimate(_click_track(120.0), tempo_bpm=120.005)
    assert result.bpm * 100 == round(result.bpm * 100)
    assert result.bpm != 120.005  # 丸めずにそのまま持つ実装を落とす


def test_resolution_matches_the_format_layer():
    """分解能は形式が固定する値。書き出し側とずれると音符の位置が丸ごとずれた vpr になる。"""
    from vpr.constants import RESOLUTION

    assert tempo.RESOLUTION == RESOLUTION


# --- 決定論 ------------------------------------------------------------------


def test_same_input_gives_the_same_estimate():
    pcm = _click_track(132.0)
    first = tempo.estimate(pcm)
    second = tempo.estimate(pcm)
    assert (first.bpm, first.numerator, first.denominator, first.first_bar_sec) == \
           (second.bpm, second.numerator, second.denominator, second.first_bar_sec)


# --- 採用値の出どころ --------------------------------------------------------


def test_given_values_are_reported_as_options():
    result = tempo.estimate(_click_track(120.0), tempo_bpm=96.0, time_signature=(3, 4))
    assert (result.tempo_source, result.time_signature_source) == ("option", "option")


def test_estimated_values_are_reported_as_estimated():
    result = tempo.estimate(_click_track(120.0, seconds=16.0, accent_every=3))
    assert (result.tempo_source, result.time_signature_source) == ("estimated", "estimated")


def test_fallback_values_are_reported_as_defaults():
    result = tempo.estimate(_silence())
    assert (result.tempo_source, result.time_signature_source) == ("default", "default")


def test_the_two_ways_of_telling_the_fallback_agree():
    """出どころと仮置きの真偽は同じ事実を指す(食い違うと警告と診断がずれる)。"""
    for result in (tempo.estimate(_silence()),
                   tempo.estimate(_silence(), tempo_bpm=100.0),
                   tempo.estimate(_click_track(120.0))):
        assert result.tempo_defaulted == (result.tempo_source == "default")
        assert result.time_signature_defaulted == (result.time_signature_source == "default")
