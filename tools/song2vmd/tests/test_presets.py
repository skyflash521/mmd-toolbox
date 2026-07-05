"""song2vmd 歌い方スタイルプリセットのテスト(song2vmd.md §8.1)。

--style が選ぶプリセットの具体値(開き量レンジ・アタック/リリース・協調調音・先行・最小保持・
誇張係数・開き量上限・母音別倍率・三角形下限・伸び表現・レガート谷)を確定し、CLIで明示指定
された値(--open-max・--coarticulation・--anticipation・--min-hold)があればプリセット値より優先し、
--vowel-gain はプリセットの母音別倍率へ乗算する presets.resolve を検証する。
"""

import pytest

from song2vmd import presets

# song2vmd.md §8.1 の表(プリセット名 → 開き量レンジ弱・強・アタック・リリース・協調調音重なり・
# 先行・最小保持・誇張係数・開き量上限)。
_EXPECTED = {
    "pop": (0.30, 0.75, 2, 2, 6, 11, 1, 1.0, 0.90),
    "ballad": (0.20, 0.55, 3, 3, 3, 1, 4, 0.8, 0.70),
    "powerful": (0.40, 0.95, 1, 1, 2, 2, 3, 1.3, 0.97),
    "whisper": (0.10, 0.35, 2, 2, 2, 1, 3, 0.7, 0.50),
    "rap": (0.30, 0.70, 1, 1, 1, 1, 2, 1.0, 0.85),
}


def test_style_names_match_song2vmd_md_8_1():
    assert set(presets.STYLE_NAMES) == set(_EXPECTED)


@pytest.mark.parametrize("style", list(_EXPECTED))
def test_resolve_without_overrides_matches_preset_table(style):
    lo, hi, attack, release, coart, anticip, min_hold, exagg, open_max = _EXPECTED[style]
    openness, gen = presets.resolve(style)
    assert openness.open_lo == pytest.approx(lo)
    assert openness.open_hi == pytest.approx(hi)
    assert openness.open_max == pytest.approx(open_max)
    assert gen.attack_frames == attack
    assert gen.release_frames == release
    assert gen.coartic_overlap_max == coart
    assert gen.anticipation_frames == anticip
    assert gen.min_hold_frames == min_hold
    assert gen.exaggeration == pytest.approx(exagg)


def test_different_styles_give_different_values():
    # open_hi・exaggerationはスタイル間で値が異なる項目として比較する(song2vmd.md 8.1)。
    pop_openness, pop_gen = presets.resolve("pop")
    whisper_openness, whisper_gen = presets.resolve("whisper")
    assert pop_openness.open_hi != whisper_openness.open_hi
    assert pop_gen.exaggeration != whisper_gen.exaggeration


def test_cli_override_open_max_takes_precedence_over_preset():
    openness, _gen = presets.resolve("pop", open_max=0.5)
    assert openness.open_max == pytest.approx(0.5)
    # 開き量レンジ自体はプリセット値のまま(--open-maxは上限だけを上書きする。song2vmd.md 8.1)。
    assert openness.open_lo == pytest.approx(0.30)
    assert openness.open_hi == pytest.approx(0.75)


def test_cli_override_coarticulation_takes_precedence_over_preset():
    _openness, gen = presets.resolve("pop", coarticulation=9)
    assert gen.coartic_overlap_max == 9


def test_cli_override_anticipation_takes_precedence_over_preset():
    _openness, gen = presets.resolve("pop", anticipation=9)
    assert gen.anticipation_frames == 9


def test_cli_override_min_hold_takes_precedence_over_preset():
    _openness, gen = presets.resolve("pop", min_hold=9)
    assert gen.min_hold_frames == 9


def test_no_overrides_use_preset_values():
    openness, gen = presets.resolve("ballad")
    assert openness.open_max == pytest.approx(0.70)
    assert gen.coartic_overlap_max == 3
    assert gen.anticipation_frames == 1
    assert gen.min_hold_frames == 4


def test_attack_release_and_exaggeration_are_not_cli_overridable():
    # song2vmd.md 5.2にはアタック/リリース/誇張係数のCLIオプションが無い(プリセット値固定)。
    openness, gen = presets.resolve("powerful")
    assert gen.attack_frames == 1
    assert gen.release_frames == 1
    assert gen.exaggeration == pytest.approx(1.3)


def test_pop_vowel_scale_matches_tuned_values():
    # popの母音別倍率はMMD視覚チューニング済みの標準値(song2vmd.md 8.1)。
    _openness, gen = presets.resolve("pop")
    assert gen.vowel_scale == (1.30, 1.20, 1.70, 0.80, 1.70, 1.00)


def test_vowel_gain_multiplies_preset_vowel_scale_elementwise():
    # --vowel-gain の5母音値はプリセットの母音別倍率へ要素ごとに乗算し、撥音「ん」(第6要素)は
    # プリセット値のまま(song2vmd.md 8.2)。
    _openness, gen = presets.resolve("pop", vowel_gain=(2.0, 1.0, 0.5, 1.0, 1.0))
    assert gen.vowel_scale[0] == pytest.approx(1.30 * 2.0)
    assert gen.vowel_scale[1] == pytest.approx(1.20)
    assert gen.vowel_scale[2] == pytest.approx(1.70 * 0.5)
    assert gen.vowel_scale[3] == pytest.approx(0.80)
    assert gen.vowel_scale[4] == pytest.approx(1.70)
    assert gen.vowel_scale[5] == pytest.approx(1.00)


def test_pop_carries_tuned_generation_parameters_explicitly():
    # popの伸び表現・レガート谷・三角形下限はプリセットが明示的に持つ(lipsync既定への
    # 暗黙依存を残さない。song2vmd.md 8.1)。
    _openness, gen = presets.resolve("pop")
    assert gen.triangle_min_frames == pytest.approx(2.0)
    assert (gen.vibrato_threshold, gen.vibrato_amp, gen.vibrato_period) == (10, 0.05, 22)
    assert gen.legato_valley_shallow == pytest.approx(0.45)
    assert gen.legato_valley_deep == pytest.approx(0.30)
    assert gen.legato_valley_slope == pytest.approx(0.02)
