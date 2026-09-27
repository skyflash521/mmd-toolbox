import numpy as np
import pytest

_VOWEL_BASE_SYMBOLS = [
    "i", "y", "ɨ", "ʉ", "ɯ", "u",
    "ɪ", "ʏ", "ʊ",
    "e", "ø", "ɘ", "ɵ", "ɤ", "o",
    "ə",
    "ɛ", "œ", "ɜ", "ɞ", "ʌ", "ɔ",
    "æ", "ɐ",
    "a", "ɶ", "ɑ", "ɒ",
    "ɚ", "ɝ",
    "ᵻ",
]


@pytest.mark.parametrize("symbol", _VOWEL_BASE_SYMBOLS)
def test_classify_symbol_vowel_base_set_is_complete(symbol):
    from vocal_analysis.recognizer import _classify_symbol

    assert _classify_symbol(symbol) == "vowel"


@pytest.mark.parametrize(
    "symbol",
    [
        pytest.param("aː", id="long-a"),
        pytest.param("iː", id="long-i"),
        pytest.param("uː", id="long-u"),
        pytest.param("eː", id="long-e"),
        pytest.param("oː", id="long-o"),
        pytest.param("ɑ̃", id="nasalized-open-back"),
        pytest.param("ɛ̃", id="nasalized-open-mid-front"),
        pytest.param("aɪ", id="falling-diphthong-ai"),
        pytest.param("eɪ", id="falling-diphthong-ei"),
        pytest.param("oʊ", id="falling-diphthong-ou"),
        pytest.param("aʊ", id="falling-diphthong-au"),
        pytest.param("ɑːɹ", id="r-colored-open-back"),
        pytest.param("ɔːɹ", id="r-colored-open-mid-back"),
        pytest.param("a1", id="tone-number-a"),
        pytest.param("e2", id="tone-number-e"),
    ],
)
def test_classify_symbol_modified_and_compound_vowel_examples(symbol):
    from vocal_analysis.recognizer import _classify_symbol

    assert _classify_symbol(symbol) == "vowel"


@pytest.mark.parametrize(
    "symbol",
    ["n", "s", "t", "k", "d", "m", "p", "z", "f", "v", "b", "ŋ", "θ", "ʃ", "ʒ", "ja", "ju", "wa", ""],
)
def test_classify_symbol_consonant_examples_including_approximant_initial_and_empty(symbol):
    from vocal_analysis.recognizer import _classify_symbol

    assert _classify_symbol(symbol) == "consonant"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("FRAME_DURATION_SEC", 0.02),
        ("_SILENCE_FRAME_SEC", 0.1),
        ("_SILENCE_MIN_RUN_SEC", 0.6),
        ("_SEGMENT_MIN_SEC", 1.5),
        ("_SEGMENT_MAX_SEC", 25.0),
        ("_HALLUCINATION_PHONEME_RATE", 20.0),
        ("_ECHO_FRAGMENT_PREFIX_LEN", 5),
        ("_REPEAT_MIN_COUNT", 3),
        ("_REPEAT_MIN_TOTAL_CHARS", 8),
        ("_FORCED_ALIGN_BAND_SEC", 1.0),
        ("_MIN_STAY_FRAMES", 6),
        ("_VOICED_BLANK_PENALTY", 7.0),
        ("_WORD_WINDOW_MARGIN_SEC", 0.75),
        ("_EARLY_COMMIT_BONUS", 2.0),
    ],
)
def test_alignment_tuning_constants_are_pinned(name, expected):
    from vocal_analysis import recognizer

    assert getattr(recognizer, name) == expected


def test_frame_rms_computes_rms_per_frame():
    from vocal_analysis.recognizer import _frame_rms

    frame_samples = 1600
    mono = np.concatenate([
        np.full(frame_samples, 0.5, dtype=np.float32), np.zeros(frame_samples, dtype=np.float32)])

    result = _frame_rms(mono, sample_rate=16000, frame_sec=0.1)

    assert result.shape == (2,)
    assert result[0] == pytest.approx(0.5)
    assert result[1] == pytest.approx(0.0)


def test_frame_rms_truncates_incomplete_final_frame():
    from vocal_analysis.recognizer import _frame_rms

    mono = np.concatenate([np.full(1600, 0.5, dtype=np.float32), np.full(800, 0.9, dtype=np.float32)])

    result = _frame_rms(mono, sample_rate=16000, frame_sec=0.1)

    assert result.shape == (1,)


def test_frame_rms_empty_input_returns_empty_array():
    from vocal_analysis.recognizer import _frame_rms

    assert _frame_rms(np.array([], dtype=np.float32), sample_rate=16000, frame_sec=0.1).shape == (0,)


def test_silence_threshold_is_30db_below_95th_percentile():
    from vocal_analysis.recognizer import _silence_threshold

    frame_rms = np.array([1.0] * 95 + [2.0] * 5)
    expected_peak = np.percentile(frame_rms, 95)

    threshold = _silence_threshold(frame_rms)

    assert threshold == pytest.approx(expected_peak * (10 ** (-30.0 / 20.0)))


def test_silence_threshold_zero_peak_gives_zero_threshold():
    from vocal_analysis.recognizer import _silence_threshold

    assert _silence_threshold(np.zeros(10)) == pytest.approx(0.0)


def test_silence_threshold_empty_input_is_zero():
    from vocal_analysis.recognizer import _silence_threshold

    assert _silence_threshold(np.array([])) == pytest.approx(0.0)


def test_detect_silence_split_points_finds_midpoint_of_qualifying_run():
    from vocal_analysis.recognizer import _detect_silence_split_points

    loud = np.full(16000, 0.5, dtype=np.float32)
    silence = np.zeros(16000, dtype=np.float32)
    mono = np.concatenate([loud, silence, loud])

    splits = _detect_silence_split_points(mono, sample_rate=16000)

    assert len(splits) == 1
    assert splits[0] == pytest.approx((1.0 + 2.0) / 2)


def test_detect_silence_split_points_ignores_silence_shorter_than_minimum_run():
    from vocal_analysis.recognizer import _detect_silence_split_points

    loud = np.full(16000, 0.5, dtype=np.float32)
    silence = np.zeros(4800, dtype=np.float32)
    mono = np.concatenate([loud, silence, loud])

    assert _detect_silence_split_points(mono, sample_rate=16000) == []


def test_detect_silence_split_points_handles_trailing_silence_run():
    from vocal_analysis.recognizer import _detect_silence_split_points

    loud = np.full(16000, 0.5, dtype=np.float32)
    silence = np.zeros(16000, dtype=np.float32)
    mono = np.concatenate([loud, silence])

    splits = _detect_silence_split_points(mono, sample_rate=16000)

    assert len(splits) == 1
    assert splits[0] == pytest.approx(1.5)


def test_build_segment_bounds_no_splits_is_single_segment():
    from vocal_analysis.recognizer import _build_segment_bounds

    assert _build_segment_bounds(duration_sec=10.0, split_points=[]) == [(0.0, 10.0)]


def test_build_segment_bounds_merges_too_short_segment_forward():
    from vocal_analysis.recognizer import _build_segment_bounds

    bounds = _build_segment_bounds(duration_sec=10.0, split_points=[0.5, 5.0])

    assert bounds == [(0.0, 5.0), (5.0, 10.0)]


def test_build_segment_bounds_keeps_short_final_segment_when_no_neighbor():
    from vocal_analysis.recognizer import _build_segment_bounds

    assert _build_segment_bounds(duration_sec=1.0, split_points=[]) == [(0.0, 1.0)]


def test_build_segment_bounds_splits_too_long_segment_evenly():
    from vocal_analysis.recognizer import _build_segment_bounds

    bounds = _build_segment_bounds(duration_sec=60.0, split_points=[])

    assert len(bounds) == 3
    assert bounds[0] == pytest.approx((0.0, 20.0))
    assert bounds[1] == pytest.approx((20.0, 40.0))
    assert bounds[2] == pytest.approx((40.0, 60.0))


def test_is_segment_silent_below_threshold_is_silent():
    from vocal_analysis.recognizer import _is_segment_silent

    assert _is_segment_silent(np.full(100, 0.001, dtype=np.float32), threshold=0.01) is True


def test_is_segment_silent_above_threshold_is_not_silent():
    from vocal_analysis.recognizer import _is_segment_silent

    assert _is_segment_silent(np.full(100, 0.5, dtype=np.float32), threshold=0.01) is False


def test_is_segment_silent_empty_segment_is_silent():
    from vocal_analysis.recognizer import _is_segment_silent

    assert _is_segment_silent(np.array([], dtype=np.float32), threshold=0.01) is True


def test_is_hallucinated_phoneme_density_exactly_at_threshold_is_not_hallucinated():
    from vocal_analysis.recognizer import _HALLUCINATION_PHONEME_RATE, _is_hallucinated_phoneme_density

    assert _is_hallucinated_phoneme_density(
        phoneme_count=round(_HALLUCINATION_PHONEME_RATE * 2.0), duration_sec=2.0) is False


def test_is_hallucinated_phoneme_density_above_threshold_is_hallucinated():
    from vocal_analysis.recognizer import _is_hallucinated_phoneme_density

    assert _is_hallucinated_phoneme_density(phoneme_count=863, duration_sec=21.15) is True


def test_is_hallucinated_phoneme_density_zero_duration_is_not_hallucinated():
    from vocal_analysis.recognizer import _is_hallucinated_phoneme_density

    assert _is_hallucinated_phoneme_density(phoneme_count=100, duration_sec=0.0) is False


def test_strip_prompt_echo_removes_exact_prompt_sentences():
    from vocal_analysis.recognizer import _strip_prompt_echo

    text, removed = _strip_prompt_echo("かんじは つかわないでください。" * 24)
    assert text == ""
    assert removed is True


def test_strip_prompt_echo_removes_trailing_fragment_after_echo():
    from vocal_analysis.recognizer import _strip_prompt_echo

    text, removed = _strip_prompt_echo("かんじは つかわないでください。かんじは つかん")
    assert text == ""
    assert removed is True


def test_strip_prompt_echo_keeps_normal_text_untouched():
    from vocal_analysis.recognizer import _strip_prompt_echo

    original = "きょうは あさから あめが ふっていて さんぽに いけなかった。"
    text, removed = _strip_prompt_echo(original)
    assert text == original
    assert removed is False


def test_strip_prompt_echo_keeps_real_sentences_between_echoes():
    from vocal_analysis.recognizer import _strip_prompt_echo

    text, removed = _strip_prompt_echo("かんじは つかわないでください。きょうはてんきがいい。")
    assert text == "きょうはてんきがいい。"
    assert removed is True


def test_strip_prompt_echo_keeps_prompt_prefix_fragment_when_no_echo_was_removed():
    from vocal_analysis.recognizer import _strip_prompt_echo

    original = "かんじはじめた こころ"
    text, removed = _strip_prompt_echo(original)
    assert text == original
    assert removed is False


def test_find_suffix_repetition_detects_trailing_unit_run():
    from vocal_analysis.recognizer import _find_suffix_repetition

    result = _find_suffix_repetition("ハテシナイミチノムコウデ" + "ラ" * 300)
    assert result is not None
    unit, count, head = result
    assert unit == "ラ"
    assert count == 300
    assert head == "ハテシナイミチノムコウデ"


def test_find_suffix_repetition_detects_whole_text_repetition():
    from vocal_analysis.recognizer import _find_suffix_repetition

    result = _find_suffix_repetition("ララ" * 200)
    assert result is not None
    unit, count, head = result
    assert head == ""
    assert unit * count == "ララ" * 200


def test_find_suffix_repetition_ignores_runs_below_minimum_count():
    from vocal_analysis.recognizer import _find_suffix_repetition

    assert _find_suffix_repetition("たのしいね たのしいね") is None


def _resolve(monkeypatch, text, words=None, duration=10.0, retry_result=None, retry_enabled=True,
             g2p=None, transcribe_calls=None, text_only_result=None, text_only_calls=None):
    from vocal_analysis import recognizer as R

    primary = R.DEFAULT_CONTENT_RECOGNIZER_MODEL

    def fake_transcribe(pipeline, samples):
        if transcribe_calls is not None:
            transcribe_calls.append(pipeline)
        if retry_result is None:
            raise AssertionError("リトライが呼ばれてはならないケースで _transcribe_segment が呼ばれた")
        return retry_result

    def fake_transcribe_text_only(samples, model, on_progress=None):
        if text_only_calls is not None:
            text_only_calls.append(model)
        if text_only_result is None:
            raise AssertionError("リトライが呼ばれてはならないケースで _transcribe_text_only が呼ばれた")
        return text_only_result

    def one_phoneme_per_non_space_char(t, method=None, **kwargs):
        return ["a"] * len(t.replace(" ", ""))

    monkeypatch.setattr(R, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(R, "_transcribe_text_only", fake_transcribe_text_only)
    monkeypatch.setattr(R, "_g2p", g2p or one_phoneme_per_non_space_char)
    return R._resolve_transcription(
        np.zeros(16000, dtype=np.float32), duration, text, words, primary, retry_enabled)


def test_resolve_transcription_retries_once_with_primary_model_when_echo_leaves_empty_text(monkeypatch):
    calls = []
    text, words = _resolve(
        monkeypatch, "かんじは つかわないでください。",
        words=[("かんじは", 0.0, 1.0)], duration=15.0,
        text_only_result="げんきです", text_only_calls=calls)
    from vocal_analysis.recognizer import DEFAULT_CONTENT_RECOGNIZER_MODEL

    assert text == "げんきです"
    assert words is None
    assert len(calls) == 1
    assert calls[0] is DEFAULT_CONTENT_RECOGNIZER_MODEL


def test_resolve_transcription_does_not_retry_on_ascii_words(monkeypatch):
    text, _words = _resolve(
        monkeypatch, "hello world つづける", duration=10.0,
        retry_result=None, text_only_result=None)
    assert text == "hello world つづける"


def test_resolve_transcription_retries_on_high_density(monkeypatch):
    text, _words = _resolve(
        monkeypatch, "あ" * 40, duration=1.0,
        text_only_result="あいうえお")
    assert text == "あいうえお"


def test_resolve_transcription_does_not_retry_normal_text(monkeypatch):
    text, words = _resolve(
        monkeypatch, "てんきがいいですね", words=[("てんきが", 0.0, 1.0)], duration=10.0,
        retry_result=None)
    assert text == "てんきがいいですね"
    assert words == [("てんきが", 0.0, 1.0)]


def test_resolve_transcription_disabled_retry_still_rescues_by_keeping_head_before_suffix_repetition(monkeypatch):
    text, _words = _resolve(
        monkeypatch, "アイウエオカキクケコ" + "ラ" * 90, duration=1.0,
        retry_result=None, retry_enabled=False)
    assert text == "アイウエオカキクケコ"


def test_resolve_transcription_normalizes_whole_text_repetition_to_3_5_morae_per_second(monkeypatch):
    from vocal_analysis import recognizer as R

    def g2p(t, method=None, **kwargs):
        return ["a"] * len(t.replace(" ", ""))

    monkeypatch.setattr(R, "_g2p", g2p)
    duration_sec = 10.0
    text, words = R._resolve_transcription(
        np.zeros(16000, dtype=np.float32), duration_sec, "ラ" * 400, None,
        R.DEFAULT_CONTENT_RECOGNIZER_MODEL, False)
    assert text == "ラ" * round(duration_sec * 3.5)
    assert words is None


def test_resolve_transcription_repetition_rescue_passes_english_katakana_method(monkeypatch):
    from vocal_analysis import recognizer as R

    g2p_methods = []

    def g2p(t, method=None, **kwargs):
        g2p_methods.append(method)
        return ["a"] * len(t.replace(" ", ""))

    monkeypatch.setattr(R, "_g2p", g2p)
    text, words = R._resolve_transcription(
        np.zeros(16000, dtype=np.float32), 10.0, "ラ" * 400, None,
        R.DEFAULT_CONTENT_RECOGNIZER_MODEL, False,
        english_katakana_method="tinyllama-katakana-converter")

    assert text == "ラ" * 35
    assert len(g2p_methods) >= 1
    assert all(m == "tinyllama-katakana-converter" for m in g2p_methods)


def test_resolve_transcription_rescue_keeps_normal_density_text(monkeypatch):
    text, words = _resolve(
        monkeypatch, "すき すき すき", words=[("すき", 0.0, 0.5)], duration=10.0,
        retry_result=None, retry_enabled=False)
    assert text == "すき すき すき"
    assert words == [("すき", 0.0, 0.5)]


def test_voiced_trim_bounds_trims_leading_and_trailing_silence_with_margin():
    from vocal_analysis.recognizer import _voiced_trim_bounds

    sr = 16000
    silence_head = np.zeros(2 * sr, dtype=np.float32)
    voiced = np.full(1 * sr, 0.5, dtype=np.float32)
    silence_tail = np.zeros(3 * sr, dtype=np.float32)
    samples = np.concatenate([silence_head, voiced, silence_tail])

    lo, hi = _voiced_trim_bounds(samples, sr, threshold=0.01)

    margin_sec = 0.1
    assert lo == round((2.0 - margin_sec) * sr)
    assert hi == round((3.0 + margin_sec) * sr)


def test_voiced_trim_bounds_clamps_margin_to_segment_edges():
    from vocal_analysis.recognizer import _voiced_trim_bounds

    sr = 16000
    samples = np.full(1 * sr, 0.5, dtype=np.float32)

    lo, hi = _voiced_trim_bounds(samples, sr, threshold=0.01)

    assert lo == 0
    assert hi == len(samples)


def test_voiced_trim_bounds_no_voiced_frame_returns_whole_segment():
    from vocal_analysis.recognizer import _voiced_trim_bounds

    sr = 16000
    samples = np.full(1 * sr, 0.001, dtype=np.float32)

    lo, hi = _voiced_trim_bounds(samples, sr, threshold=0.01)

    assert (lo, hi) == (0, len(samples))


def test_merge_adjacent_segments_combines_same_type_and_phoneme():
    from vocal_analysis.recognizer import _merge_adjacent_segments
    from vocal_analysis.types import Segment

    segments = [
        Segment(type="vowel", start_sec=0.0, end_sec=1.0, phoneme="a", confidence=None),
        Segment(type="vowel", start_sec=1.0, end_sec=2.0, phoneme="a", confidence=None),
    ]

    result = _merge_adjacent_segments(segments)

    assert len(result) == 1
    assert result[0].start_sec == pytest.approx(0.0)
    assert result[0].end_sec == pytest.approx(2.0)


def test_merge_adjacent_segments_keeps_different_phoneme_separate():
    from vocal_analysis.recognizer import _merge_adjacent_segments
    from vocal_analysis.types import Segment

    segments = [
        Segment(type="vowel", start_sec=0.0, end_sec=1.0, phoneme="a", confidence=None),
        Segment(type="vowel", start_sec=1.0, end_sec=2.0, phoneme="i", confidence=None),
    ]

    result = _merge_adjacent_segments(segments)

    assert len(result) == 2


def test_sanitize_word_timestamps_clamps_to_duration_range():
    from vocal_analysis.recognizer import _sanitize_word_timestamps

    result = _sanitize_word_timestamps([("a", -0.5, 3.0)], duration_sec=2.0)

    assert result == [("a", 0.0, 2.0)]


def test_sanitize_word_timestamps_raises_start_that_goes_back_to_previous_start():
    from vocal_analysis.recognizer import _sanitize_word_timestamps

    result = _sanitize_word_timestamps([("a", 0.5, 0.8), ("b", 0.3, 0.9)], duration_sec=2.0)

    assert result[0] == ("a", 0.5, 0.8)
    assert result[1][1] == pytest.approx(0.5)


def test_sanitize_word_timestamps_enforces_minimum_length():
    from vocal_analysis.recognizer import _MIN_WORD_DURATION_SEC, _sanitize_word_timestamps

    result = _sanitize_word_timestamps([("a", 1.0, 1.0)], duration_sec=2.0)

    assert result[0][2] == pytest.approx(1.0 + _MIN_WORD_DURATION_SEC)


def test_sanitize_word_timestamps_minimum_length_is_applied_after_clamp_and_may_exceed_duration():
    from vocal_analysis.recognizer import _sanitize_word_timestamps

    result = _sanitize_word_timestamps([("a", 1.98, 2.5)], duration_sec=2.0)

    assert result == [("a", 1.98, pytest.approx(1.98 + 0.05))]


def test_widen_cramped_words_widens_below_30ms_per_phoneme_to_40ms_keeping_end():
    from vocal_analysis.recognizer import _widen_cramped_words

    result = _widen_cramped_words([(["a", "b", "c", "d"], 0.94, 1.0)], duration_sec=2.0)

    assert result[0][0] == ["a", "b", "c", "d"]
    assert result[0][1] == pytest.approx(1.0 - 4 * 0.04)
    assert result[0][2] == pytest.approx(1.0)


def test_widen_cramped_words_keeps_words_at_or_above_threshold():
    from vocal_analysis.recognizer import _widen_cramped_words

    words = [(["a", "b"], 0.0, 0.06), (["a"], 0.5, 1.0)]
    assert _widen_cramped_words(words, duration_sec=2.0) == words


def test_widen_cramped_words_widens_just_below_the_threshold():
    from vocal_analysis.recognizer import _widen_cramped_words

    result = _widen_cramped_words([(["a", "b"], 1.0, 1.058)], duration_sec=2.0)

    assert result[0][1] == pytest.approx(1.058 - 0.08)
    assert result[0][2] == pytest.approx(1.058)


def test_widen_cramped_words_does_not_remonotonize_starts():
    from vocal_analysis.recognizer import _widen_cramped_words

    words = [(["a"], 0.98, 1.03), (["b", "c", "d", "e"], 1.04, 1.10)]
    result = _widen_cramped_words(words, duration_sec=2.0)

    assert result[0] == (["a"], 0.98, 1.03)
    assert result[1][1] == pytest.approx(0.94)
    assert result[1][2] == pytest.approx(1.10)


def test_widen_cramped_words_sends_overflow_past_start_to_the_end():
    from vocal_analysis.recognizer import _widen_cramped_words

    result = _widen_cramped_words([(["a", "b", "c", "d"], 0.02, 0.1)], duration_sec=2.0)

    assert result[0][1] == pytest.approx(0.0)
    assert result[0][2] == pytest.approx(0.16)


def test_widen_cramped_words_clamps_to_the_trimmed_range():
    from vocal_analysis.recognizer import _widen_cramped_words

    result = _widen_cramped_words([(["a", "b", "c", "d"], 0.05, 0.1)], duration_sec=0.12)

    assert result[0][1] == pytest.approx(0.0)
    assert result[0][2] == pytest.approx(0.12)


def test_widen_cramped_words_ignores_words_without_phonemes():
    from vocal_analysis.recognizer import _widen_cramped_words

    words = [([], 0.5, 0.5)]
    assert _widen_cramped_words(words, duration_sec=2.0) == words


def test_widen_cramped_words_clamps_the_end_before_widening():
    from vocal_analysis.recognizer import _widen_cramped_words

    result = _widen_cramped_words([(["a", "b"], 1.98, 2.03)], duration_sec=2.0)

    assert result[0][1] == pytest.approx(1.92)
    assert result[0][2] == pytest.approx(2.0)


def test_widen_cramped_words_tolerates_float_noise_at_the_threshold():
    from vocal_analysis.recognizer import _widen_cramped_words

    words = [(["a", "b"], 30.64, 30.70)]
    assert _widen_cramped_words(words, duration_sec=40.0) == words


def test_extract_word_timestamps_skips_empty_text_and_missing_start():
    from vocal_analysis.recognizer import _extract_word_timestamps

    chunks = [
        {"text": "あ", "timestamp": (0.0, 0.5)},
        {"text": "  ", "timestamp": (0.5, 1.0)},
        {"text": "い", "timestamp": (None, 1.5)},
        {"text": "う", "timestamp": (1.5, 2.0)},
    ]

    result = _extract_word_timestamps(chunks, duration_sec=2.0)

    assert [w[0] for w in result] == ["あ", "う"]


def test_extract_word_timestamps_fills_missing_end_with_minimum_length():
    from vocal_analysis.recognizer import _MIN_WORD_DURATION_SEC, _extract_word_timestamps

    chunks = [{"text": "あ", "timestamp": (1.0, None)}]

    result = _extract_word_timestamps(chunks, duration_sec=2.0)

    assert result == [("あ", 1.0, 1.0 + _MIN_WORD_DURATION_SEC)]


def test_assemble_phoneme_sequence_wraps_single_chunk_with_pau():
    from vocal_analysis.recognizer import _assemble_phoneme_sequence

    result = _assemble_phoneme_sequence([["a", "i"]])

    assert result == ["pau", "a", "i", "pau"]


def test_assemble_phoneme_sequence_inserts_pau_between_chunks():
    from vocal_analysis.recognizer import _assemble_phoneme_sequence

    result = _assemble_phoneme_sequence([["a"], ["k", "i"]])

    assert result == ["pau", "a", "pau", "k", "i", "pau"]


def test_assemble_phoneme_sequence_empty_chunk_still_gets_boundary_pau():
    from vocal_analysis.recognizer import _assemble_phoneme_sequence

    result = _assemble_phoneme_sequence([[], ["a"]])

    assert result == ["pau", "pau", "a", "pau"]


def test_assemble_phoneme_sequence_no_chunks_is_leading_and_trailing_pau_only():
    from vocal_analysis.recognizer import _assemble_phoneme_sequence

    result = _assemble_phoneme_sequence([])

    assert result == ["pau", "pau"]


def test_assemble_with_word_windows_single_word_leading_word_and_trailing_windows():
    from vocal_analysis.recognizer import _assemble_with_word_windows

    seq, windows = _assemble_with_word_windows([(["a"], 1.0, 1.5)], margin_sec=0.5, duration_sec=3.0)

    assert seq == ["pau", "a", "pau"]
    leading_pau, word, trailing_pau = windows
    assert leading_pau == (0.0, 1.0 + 0.5)
    assert word == (1.0 - 0.5, 1.5 + 0.5)
    assert trailing_pau == (1.5 - 0.5, 3.0)


def test_assemble_with_word_windows_inter_word_pau_spans_neighbors_and_word_phonemes_share_window():
    from vocal_analysis.recognizer import _assemble_with_word_windows

    words_phonemes = [(["a"], 0.5, 1.0), (["k", "i"], 2.0, 2.5)]

    seq, windows = _assemble_with_word_windows(words_phonemes, margin_sec=0.3, duration_sec=3.0)

    assert seq == ["pau", "a", "pau", "k", "i", "pau"]
    inter_word_window = windows[2]
    assert inter_word_window == pytest.approx((1.0 - 0.3, 2.0 + 0.3))
    assert windows[3] == windows[4]


def test_assemble_with_word_windows_swaps_inverted_window():
    from vocal_analysis.recognizer import _assemble_with_word_windows

    words_phonemes = [(["a"], 0.0, 2.0), (["i"], 2.1, 2.5)]

    _, windows = _assemble_with_word_windows(words_phonemes, margin_sec=0.5, duration_sec=3.0)

    inter_word_window = windows[2]
    assert inter_word_window[0] <= inter_word_window[1]


def test_g2p_symbols_to_token_ids_maps_known_symbols():
    from vocal_analysis.recognizer import _g2p_symbols_to_token_ids

    vocab = {"<pad>": 0, "a": 5, "k": 7}

    result = _g2p_symbols_to_token_ids(["a", "k"], vocab, blank_token_id=0)

    assert result == [5, 7]


def test_g2p_symbols_to_token_ids_maps_pau_and_cl_to_blank():
    from vocal_analysis.recognizer import _g2p_symbols_to_token_ids

    vocab = {"<pad>": 0, "a": 5}

    result = _g2p_symbols_to_token_ids(["pau", "cl"], vocab, blank_token_id=0)

    assert result == [0, 0]


def test_g2p_symbols_to_token_ids_devoiced_vowels_collapse_to_voiced():
    from vocal_analysis.recognizer import _g2p_symbols_to_token_ids

    vocab = {"<pad>": 0, "i": 3, "ɯ": 4}

    result = _g2p_symbols_to_token_ids(["I", "U"], vocab, blank_token_id=0)

    assert result == [3, 4]


def test_g2p_symbols_to_token_ids_unmapped_symbol_raises_recognition_error():
    from vocal_analysis.recognizer import RecognitionError, _g2p_symbols_to_token_ids

    with pytest.raises(RecognitionError):
        _g2p_symbols_to_token_ids(["xx"], {"<pad>": 0}, blank_token_id=0)


def test_g2p_symbols_to_token_ids_missing_vocab_entry_raises_recognition_error():
    from vocal_analysis.recognizer import RecognitionError, _g2p_symbols_to_token_ids

    with pytest.raises(RecognitionError):
        _g2p_symbols_to_token_ids(["a"], {"<pad>": 0}, blank_token_id=0)


def test_voiced_blank_penalty_applies_only_to_voiced_frames_blank_column_without_mutating_input():
    from vocal_analysis.recognizer import _VOICED_BLANK_PENALTY, _apply_voiced_blank_penalty

    samples_per_frame = 320
    samples = np.concatenate([
        np.zeros(2 * samples_per_frame, dtype=np.float32),
        np.full(2 * samples_per_frame, 0.5, dtype=np.float32),
    ])
    log_probs = np.zeros((4, 3))

    result = _apply_voiced_blank_penalty(log_probs, samples, threshold=0.01, blank_token_id=1)

    expected = np.zeros((4, 3))
    expected[2, 1] = -_VOICED_BLANK_PENALTY
    expected[3, 1] = -_VOICED_BLANK_PENALTY
    assert np.array_equal(result, expected)
    assert np.array_equal(log_probs, np.zeros((4, 3)))


def test_voiced_blank_penalty_handles_frame_count_mismatch():
    from vocal_analysis.recognizer import _apply_voiced_blank_penalty

    samples = np.full(3 * 320, 0.5, dtype=np.float32)
    log_probs = np.zeros((4, 2))

    result = _apply_voiced_blank_penalty(log_probs, samples, threshold=0.01, blank_token_id=0)

    assert result.shape == (4, 2)
    assert result[3, 0] == 0.0
    assert result[0, 0] < 0.0


def test_expand_min_stay_expands_phoneme_tokens_only():
    from vocal_analysis.recognizer import _MIN_STAY_FRAMES, _expand_min_stay

    sub_ids, sub_to_token = _expand_min_stay([0, 7, 0], blank_token_id=0, num_frames=100)

    assert sub_ids == [0] + [7] * _MIN_STAY_FRAMES + [0]
    assert sub_to_token == [0] + [1] * _MIN_STAY_FRAMES + [2]


def test_expand_min_stay_reduces_stay_when_frames_are_scarce():
    from vocal_analysis.recognizer import _expand_min_stay

    num_frames = 6
    num_blank = 2
    sub_ids, sub_to_token = _expand_min_stay([0, 7, 0], blank_token_id=0, num_frames=num_frames)

    stay = num_frames - num_blank
    assert sub_ids == [0] + [7] * stay + [0]
    assert sub_to_token == [0] + [1] * stay + [2]


def test_expand_min_stay_never_reduces_below_one():
    from vocal_analysis.recognizer import _expand_min_stay

    sub_ids, _ = _expand_min_stay([0, 7, 0], blank_token_id=0, num_frames=2)

    assert sub_ids == [0, 7, 0]


def test_expand_min_stay_all_blank_sequence_is_unchanged():
    from vocal_analysis.recognizer import _expand_min_stay

    sub_ids, sub_to_token = _expand_min_stay([0, 0], blank_token_id=0, num_frames=10)

    assert sub_ids == [0, 0]
    assert sub_to_token == [0, 1]


def test_expand_min_stay_local_uses_word_duration_over_phoneme_count():
    from vocal_analysis.recognizer import _expand_min_stay_local

    words_phonemes = [(["a", "i"], 0.0, 0.12)]
    sub_ids, sub_to_token = _expand_min_stay_local([0, 1, 2, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0, 1, 1, 1, 2, 2, 2, 0]
    assert sub_to_token == [0, 1, 1, 1, 2, 2, 2, 3]


def test_expand_min_stay_local_varies_per_word_within_same_chunk():
    from vocal_analysis.recognizer import _expand_min_stay_local

    words_phonemes = [(["a"], 0.0, 0.12), (["i"], 0.12, 0.14)]
    sub_ids, sub_to_token = _expand_min_stay_local([0, 1, 0, 2, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0] + [1] * 6 + [0] + [2] * 1 + [0]
    assert sub_to_token == [0] + [1] * 6 + [2] + [3] * 1 + [4]


def test_expand_min_stay_local_clamps_to_global_max_stay():
    from vocal_analysis.recognizer import _MIN_STAY_FRAMES, _expand_min_stay_local

    words_phonemes = [(["a"], 0.0, 2.0)]
    sub_ids, _ = _expand_min_stay_local([0, 1, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0] + [1] * _MIN_STAY_FRAMES + [0]


def test_expand_min_stay_local_never_reduces_below_one():
    from vocal_analysis.recognizer import _expand_min_stay_local

    words_phonemes = [(["a", "i", "u"], 0.0, 0.02)]
    sub_ids, _ = _expand_min_stay_local([0, 1, 2, 3, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0, 1, 2, 3, 0]


def test_expand_min_stay_local_rounds_quotient_just_below_integer_by_float_error_up():
    from vocal_analysis.recognizer import _expand_min_stay_local

    words_phonemes = [(["a"], 0.02, 0.06)]
    sub_ids, _ = _expand_min_stay_local([0, 1, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0, 1, 1, 0]


def test_expand_min_stay_local_rounds_half_up_not_half_to_even():
    from vocal_analysis.recognizer import _expand_min_stay_local

    words_phonemes = [(["a"], 0.0, 0.05)]
    sub_ids, _ = _expand_min_stay_local([0, 1, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0, 1, 1, 1, 0]


def test_expand_min_stay_local_empty_phonemes_word_contributes_no_substates():
    from vocal_analysis.recognizer import _expand_min_stay_local

    words_phonemes = [([], 0.0, 0.02), (["a"], 0.02, 0.14)]
    sub_ids, sub_to_token = _expand_min_stay_local([0, 0, 1, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0, 0] + [1] * 6 + [0]
    assert sub_to_token == [0, 1] + [2] * 6 + [3]


def test_expand_min_stay_local_word_internal_blank_symbol_counts_toward_divisor_but_stays_one_frame():
    from vocal_analysis.recognizer import _expand_min_stay_local

    words_phonemes = [(["a", "cl", "t", "e"], 0.0, 0.24)]
    sub_ids, sub_to_token = _expand_min_stay_local(
        [0, 5, 0, 7, 6, 0], words_phonemes, blank_token_id=0
    )

    assert sub_ids == [0] + [5] * 3 + [0] + [7] * 3 + [6] * 3 + [0]
    assert sub_to_token == [0] + [1] * 3 + [2] + [3] * 3 + [4] * 3 + [5]


def test_min_stay_expansion_forces_phoneme_to_occupy_expanded_frames():
    from vocal_analysis.recognizer import _MIN_STAY_FRAMES, _expand_min_stay, _forced_align

    num_frames = _MIN_STAY_FRAMES + 4
    log_probs = np.array(
        [[5.0, -5.0]] * 4 + [[-5.0, 5.0]] + [[5.0, -5.0]] * (num_frames - 5)
    )
    plain_path = _forced_align(log_probs, [0, 1, 0])
    assert sum(1 for s in plain_path if s == 1) == 1

    sub_ids, sub_to_token = _expand_min_stay([0, 1, 0], blank_token_id=0, num_frames=num_frames)
    sub_path = _forced_align(log_probs, sub_ids)
    path = [sub_to_token[s] for s in sub_path]

    assert sum(1 for s in path if s == 1) >= _MIN_STAY_FRAMES


def test_forced_align_single_state_stays_for_all_frames():
    from vocal_analysis.recognizer import _forced_align

    log_probs = np.array([[0.1], [-1.0], [0.2]])

    path = _forced_align(log_probs, token_ids=[0])

    assert path == [0, 0, 0]


def test_forced_align_follows_dominant_emission_monotonically():
    from vocal_analysis.recognizer import _forced_align

    log_probs = np.array(
        [
            [5.0, -5.0],
            [5.0, -5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
        ]
    )

    path = _forced_align(log_probs, token_ids=[0, 1])

    assert path == [0, 0, 1, 1]


def test_forced_align_raises_when_fewer_frames_than_tokens():
    from vocal_analysis.recognizer import RecognitionError, _forced_align

    log_probs = np.zeros((2, 1))

    with pytest.raises(RecognitionError):
        _forced_align(log_probs, token_ids=[0, 0, 0])


def test_forced_align_band_limit_excludes_out_of_band_favorable_evidence():
    from vocal_analysis.recognizer import _forced_align

    num_frames = 200
    log_probs = np.zeros((num_frames, 2))
    log_probs[:, 0] = 0.0
    log_probs[:, 1] = -8.0
    favorable_frames = range(180, 186)
    log_probs[favorable_frames.start:favorable_frames.stop, 1] = 0.0
    token_ids = [0, 1, 0]

    path = _forced_align(log_probs, token_ids)

    state1_frames = [i for i, state in enumerate(path) if state == 1]
    assert state1_frames
    assert all(49 <= f <= 150 for f in state1_frames)
    assert all(f not in favorable_frames for f in state1_frames)


def test_forced_align_windowed_follows_dominant_emission_within_window():
    from vocal_analysis.recognizer import _forced_align_windowed

    log_probs = np.array(
        [
            [5.0, -5.0],
            [5.0, -5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
        ]
    )
    loose_windows = [(0.0, 10.0), (0.0, 10.0)]

    path = _forced_align_windowed(log_probs, token_ids=[0, 1], windows_sec=loose_windows)

    assert path == [0, 0, 1, 1]


def test_forced_align_windowed_excludes_evidence_outside_window():
    from vocal_analysis.recognizer import _forced_align_windowed

    num_frames = 200
    log_probs = np.zeros((num_frames, 3))
    log_probs[:, 1] = -8.0
    favorable_frames = range(180, 186)
    log_probs[favorable_frames.start:favorable_frames.stop, 1] = 0.0
    state1_window_end_frame = 150
    windows = [(0.0, 4.0), (0.0, state1_window_end_frame * 0.02), (0.0, 4.0)]

    path = _forced_align_windowed(log_probs, token_ids=[0, 1, 0], windows_sec=windows)

    state1_frames = [i for i, state in enumerate(path) if state == 1]
    assert state1_frames
    assert all(f < state1_window_end_frame for f in state1_frames)
    assert all(f not in favorable_frames for f in state1_frames)


def test_forced_align_windowed_raises_when_window_unreachable():
    from vocal_analysis.recognizer import RecognitionError, _forced_align_windowed

    log_probs = np.zeros((4, 2))
    windows = [(0.0, 10.0), (100.0, 200.0)]

    with pytest.raises(RecognitionError):
        _forced_align_windowed(log_probs, token_ids=[0, 1], windows_sec=windows)


def test_forced_align_windowed_raises_when_fewer_frames_than_tokens():
    from vocal_analysis.recognizer import RecognitionError, _forced_align_windowed

    log_probs = np.zeros((2, 1))

    with pytest.raises(RecognitionError):
        _forced_align_windowed(log_probs, token_ids=[0, 0, 0], windows_sec=[(0, 1), (0, 1), (0, 1)])


def test_forced_align_windowed_early_commit_bonus_pulls_transition_toward_window_open():
    from vocal_analysis.recognizer import FRAME_DURATION_SEC, _forced_align_windowed

    num_frames = int(8.0 / FRAME_DURATION_SEC)
    log_probs = np.zeros((num_frames, 2))
    log_probs[:, 1] = -0.05
    windows = [(0.0, 8.0), (2.0, 8.0)]

    path = _forced_align_windowed(log_probs, token_ids=[0, 1], windows_sec=windows)

    first_state1_frame = path.index(1)
    assert first_state1_frame == round(2.0 / FRAME_DURATION_SEC)


def test_viterbi_monotonic_without_state_bias_delays_transition_to_deadline():
    from vocal_analysis.recognizer import FRAME_DURATION_SEC, _viterbi_monotonic

    num_frames = int(8.0 / FRAME_DURATION_SEC)
    log_probs = np.zeros((num_frames, 2))
    log_probs[:, 1] = -0.05
    windows = [(0.0, 8.0), (2.0, 8.0)]

    frame_lo = np.arange(num_frames) * FRAME_DURATION_SEC
    frame_hi = frame_lo + FRAME_DURATION_SEC
    win_lo = np.array([w[0] for w in windows])
    win_hi = np.array([w[1] for w in windows])
    out_of_window = ~((win_lo[None, :] < frame_hi[:, None]) & (win_hi[None, :] > frame_lo[:, None]))

    path = _viterbi_monotonic(log_probs, token_ids=[0, 1], out_of_bounds=out_of_window, state_bias=None)

    first_state1_frame = path.index(1)
    assert first_state1_frame == num_frames - 1


def test_path_to_segments_builds_gap_and_vowel_segments():
    from vocal_analysis.recognizer import _path_to_segments

    segments = _path_to_segments([0, 0, 1, 1], ["pau", "a"], frame_duration_sec=0.02)

    assert len(segments) == 2
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.04)
    assert segments[1].type == "vowel"
    assert segments[1].phoneme == "a"
    assert segments[1].start_sec == pytest.approx(0.04)
    assert segments[1].end_sec == pytest.approx(0.08)


def test_path_to_segments_merges_adjacent_states_with_same_output_symbol():
    from vocal_analysis.recognizer import _path_to_segments

    segments = _path_to_segments([0, 0, 1, 1], ["a", "a"], frame_duration_sec=0.02)

    assert len(segments) == 1
    assert segments[0].type == "vowel"
    assert segments[0].phoneme == "a"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.08)


def test_path_to_segments_merges_pau_and_cl_as_same_gap():
    from vocal_analysis.recognizer import _path_to_segments

    segments = _path_to_segments([0, 0, 1, 1], ["pau", "cl"], frame_duration_sec=0.02)

    assert len(segments) == 1
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.08)


def test_path_to_segments_last_token_extends_to_audio_end():
    from vocal_analysis.recognizer import _path_to_segments

    segments = _path_to_segments([0, 1, 1, 1, 1], ["pau", "a"], frame_duration_sec=0.02)

    assert segments[-1].end_sec == pytest.approx(0.10)


def test_g2p_applies_english_katakana_fallback_before_pyopenjtalk(monkeypatch):
    import pyopenjtalk

    import vocal_analysis.recognizer as recognizer_module

    convert_calls = []
    monkeypatch.setattr(
        recognizer_module, "convert_target_words",
        lambda text, method=None, **kwargs: convert_calls.append(text) or "スカイ",
    )

    g2p_calls = []

    def fake_g2p(text, kana=None, join=None):
        g2p_calls.append((text, kana, join))
        return ["s", "u", "k", "a", "i"]

    monkeypatch.setattr(pyopenjtalk, "g2p", fake_g2p)

    result = recognizer_module._g2p("空を見上げてsky")

    assert convert_calls == ["空を見上げてsky"]
    assert g2p_calls == [("スカイ", False, False)]
    assert result == ["s", "u", "k", "a", "i"]


def test_g2p_method_reaches_real_convert_target_words_dispatch(monkeypatch):
    import vocal_analysis.english_katakana as katakana_module
    import vocal_analysis.recognizer as recognizer_module

    katakana_module._conversion_cache.clear()
    try:
        arpakana_calls = []
        tinyllama_calls = []
        monkeypatch.setattr(
            katakana_module, "_generate_katakana_arpakana",
            lambda word, phonemes: arpakana_calls.append(word) or "スカイ",
        )
        monkeypatch.setattr(
            katakana_module, "_generate_katakana_tinyllama",
            lambda word, phonemes, **kwargs: tinyllama_calls.append(word) or "スカイ",
        )

        recognizer_module._g2p("空を見上げてsky", method="tinyllama-katakana-converter")

        assert tinyllama_calls == ["sky"]
        assert arpakana_calls == []
    finally:
        katakana_module._conversion_cache.clear()


@pytest.mark.parametrize(
    "raised,match",
    [
        (ImportError("no nltk"), "nltk"),
        (LookupError("cmudict not found"), "cmudict"),
        (OSError("network error"), "カタカナ生成モデル"),
    ],
)
def test_g2p_converts_katakana_fallback_errors_to_recognition_error(monkeypatch, raised, match):
    import vocal_analysis.recognizer as recognizer_module

    def fake_convert_target_words(text, method=None, **kwargs):
        raise raised

    monkeypatch.setattr(recognizer_module, "convert_target_words", fake_convert_target_words)

    with pytest.raises(recognizer_module.RecognitionError, match=match):
        recognizer_module._g2p("空を見上げてsky")


def _isolated_g2p_len(text):
    import pyopenjtalk

    return len(pyopenjtalk.g2p(text, kana=False, join=False)) if text else 0


def _slice_by_segment_lengths(sequence, fragment_lens, target_lens):
    pos = 0
    slices = []
    for i, fragment_len in enumerate(fragment_lens):
        slices.append(sequence[pos:pos + fragment_len])
        pos += fragment_len
        if i < len(target_lens):
            pos += target_lens[i]
    return slices


@pytest.mark.parametrize(
    "text,target_spans",
    [
        pytest.param("空を見上げてsky", [(6, 9)], id="target-at-end"),
        pytest.param("skyを見上げて", [(0, 3)], id="target-followed-by-kanji-word"),
        pytest.param("skyだから", [(0, 3)], id="target-followed-by-particle"),
        pytest.param("愛してるsky", [(4, 7)], id="target-after-inflection"),
        pytest.param("skyを見上げて空", [(0, 3)], id="target-at-start"),
        pytest.param("skyとsky", [(0, 3), (4, 7)], id="two-targets"),
    ],
)
def test_g2p_full_reanalysis_preserves_phonemes_outside_target(monkeypatch, text, target_spans):
    import pyopenjtalk

    import vocal_analysis.english_katakana as katakana_module
    from vocal_analysis.recognizer import _g2p

    katakana_module._conversion_cache.clear()
    try:
        generate_calls = []
        monkeypatch.setattr(
            katakana_module, "_generate_katakana",
            lambda word, phonemes, method=None, **kwargs: generate_calls.append(word) or "スカイ",
        )

        fragments = []
        cursor = 0
        for start, end in target_spans:
            fragments.append(text[cursor:start])
            cursor = end
        fragments.append(text[cursor:])
        fragment_lens = [_isolated_g2p_len(fragment) for fragment in fragments]
        before_target_lens = [_isolated_g2p_len(text[start:end]) for start, end in target_spans]
        after_target_lens = [_isolated_g2p_len("スカイ") for _ in target_spans]

        before_full = pyopenjtalk.g2p(text, kana=False, join=False)
        after_full = _g2p(text)

        assert generate_calls == ["sky"]

        assert sum(fragment_lens) + sum(before_target_lens) == len(before_full)
        assert sum(fragment_lens) + sum(after_target_lens) == len(after_full)

        before_fragments = _slice_by_segment_lengths(before_full, fragment_lens, before_target_lens)
        after_fragments = _slice_by_segment_lengths(after_full, fragment_lens, after_target_lens)

        assert before_fragments == after_fragments
    finally:
        katakana_module._conversion_cache.clear()


def _fake_file_count_bar(tqdm_class, n=2):
    import tqdm.auto

    # huggingface_hub は tqdm_class に tqdm.auto.tqdm の派生を求める。
    assert issubclass(tqdm_class, tqdm.auto.tqdm)
    # snapshot_download のファイル数バーは unit を持たず、キャッシュ済みでもファイル数ぶん進む。
    bar = tqdm_class(total=n, desc=f"Fetching {n} files")
    for _ in range(n):
        bar.update(1)
    bar.close()


def test_load_model_and_processor_reports_live_download_percentage(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from vocal_analysis import recognizer as recognizer_module

    calls = []
    timeline = []

    def fake_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
        calls.append((repo_id, revision))
        timeline.append("snapshot")
        _fake_file_count_bar(tqdm_class)
        byte_bar = tqdm_class(total=0, desc="Downloading (incomplete total...)", unit="B", unit_scale=True)
        byte_bar.total += 100
        byte_bar.update(40)
        byte_bar.update(60)
        byte_bar.close()

    monkeypatch.setattr(recognizer_module, "_hf_snapshot_download", fake_snapshot_download)

    class _FakeTokenizer:
        pad_token_id = 0

        def get_vocab(self):
            return {}

    class _FakeProc:
        tokenizer = _FakeTokenizer()

    class _FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    def fake_processor_from_pretrained(*a, **k):
        timeline.append("processor")
        return _FakeProc()

    def fake_model_from_pretrained(*a, **k):
        timeline.append("model")
        return _FakeModel()

    monkeypatch.setattr(recognizer_module, "_transformers_auto_processor_from_pretrained",
                        fake_processor_from_pretrained)
    monkeypatch.setattr(recognizer_module, "_transformers_auto_model_for_ctc_from_pretrained",
                        fake_model_from_pretrained)

    def on_progress(note):
        timeline.append(("on_progress", note))

    recognizer_module._load_model_and_processor(on_progress=on_progress)

    model_id = recognizer_module.RECOGNIZER_CONFIG.model_id
    model_revision = recognizer_module.RECOGNIZER_CONFIG.model_revision
    assert calls == [(model_id, model_revision)]
    assert timeline == [
        "snapshot",
        ("on_progress", f"ダウンロード中: {model_id} 40%"),
        ("on_progress", f"ダウンロード中: {model_id} 100%"),
        ("on_progress", f"音素モデル読み込み中: {model_id}"),
        "processor", "model",
        ("on_progress", ""),
    ]


def test_load_model_and_processor_shows_loading_note_but_no_download_note_when_already_cached(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from vocal_analysis import recognizer as recognizer_module

    def fake_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
        _fake_file_count_bar(tqdm_class)
        byte_bar = tqdm_class(total=0, desc="Downloading (incomplete total...)", unit="B", unit_scale=True)
        byte_bar.close()

    monkeypatch.setattr(recognizer_module, "_hf_snapshot_download", fake_snapshot_download)

    class _FakeTokenizer:
        pad_token_id = 0

        def get_vocab(self):
            return {}

    class _FakeProc:
        tokenizer = _FakeTokenizer()

    class _FakeModel:
        def to(self, device):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(recognizer_module, "_transformers_auto_processor_from_pretrained",
                        lambda *a, **k: _FakeProc())
    monkeypatch.setattr(recognizer_module, "_transformers_auto_model_for_ctc_from_pretrained",
                        lambda *a, **k: _FakeModel())

    calls = []
    recognizer_module._load_model_and_processor(on_progress=lambda note: calls.append(note))

    model_id = recognizer_module.RECOGNIZER_CONFIG.model_id
    assert calls == [f"音素モデル読み込み中: {model_id}"]


def test_load_content_recognizer_pipeline_reports_live_download_percentage(monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module

    monkeypatch.setattr(recognizer_module, "_content_recognizer_pipeline_cache", None)
    calls = []
    timeline = []

    def fake_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
        calls.append((repo_id, revision))
        timeline.append("snapshot")
        _fake_file_count_bar(tqdm_class)
        byte_bar = tqdm_class(total=0, desc="Downloading (incomplete total...)", unit="B", unit_scale=True)
        byte_bar.total += 100
        byte_bar.update(100)
        byte_bar.close()

    def fake_transformers_pipeline(*a, **k):
        timeline.append("pipeline")
        return object()

    monkeypatch.setattr(recognizer_module, "_hf_snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(recognizer_module, "_transformers_pipeline", fake_transformers_pipeline)

    content_recognizer_model = ContentRecognizerModel(model_id="dummy/model", model_revision="main")
    recognizer_module._load_content_recognizer_pipeline(
        content_recognizer_model, on_progress=lambda note: timeline.append(("on_progress", note))
    )

    assert calls == [("dummy/model", "main")]
    assert timeline == [
        "snapshot",
        ("on_progress", "ダウンロード中: dummy/model 100%"),
        ("on_progress", "内容認識モデル読み込み中: dummy/model"),
        "pipeline",
        ("on_progress", ""),
    ]


def test_load_content_recognizer_pipeline_shows_loading_note_but_no_download_note_when_already_cached(
    monkeypatch,
):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module

    monkeypatch.setattr(recognizer_module, "_content_recognizer_pipeline_cache", None)

    def fake_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
        _fake_file_count_bar(tqdm_class)
        byte_bar = tqdm_class(total=0, desc="Downloading (incomplete total...)", unit="B", unit_scale=True)
        byte_bar.close()

    monkeypatch.setattr(recognizer_module, "_hf_snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(recognizer_module, "_transformers_pipeline", lambda *a, **k: object())

    calls = []
    content_recognizer_model = ContentRecognizerModel(model_id="dummy/model", model_revision="main")
    recognizer_module._load_content_recognizer_pipeline(
        content_recognizer_model, on_progress=lambda note: calls.append(note)
    )

    assert calls == ["内容認識モデル読み込み中: dummy/model"]


def test_prepare_english_katakana_conversion_arpakana_does_nothing(monkeypatch):
    import vocal_analysis.recognizer as R

    monkeypatch.setattr(
        R, "uncached_target_words",
        lambda text, method: (_ for _ in ()).throw(AssertionError("arpakanaでは対象語検出を行わない")),
    )
    monkeypatch.setattr(
        R, "_release_content_recognizer_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("arpakanaでは内容認識モデルを解放しない")),
    )

    R._prepare_english_katakana_conversion("hello を見上げて", "arpakana")


def test_prepare_english_katakana_conversion_tinyllama_releases_pipeline_before_converting(monkeypatch):
    import vocal_analysis.recognizer as R

    calls = []
    monkeypatch.setattr(R, "uncached_target_words", lambda text, method: ["hello"])
    monkeypatch.setattr(
        R, "_release_content_recognizer_pipeline", lambda: calls.append("release_pipeline")
    )
    monkeypatch.setattr(
        R, "convert_words",
        lambda words, method, **kwargs: calls.append(("convert", tuple(words), method)),
    )

    R._prepare_english_katakana_conversion("hello を見上げて", "tinyllama-katakana-converter")

    assert calls == [
        "release_pipeline",
        ("convert", ("hello",), "tinyllama-katakana-converter"),
    ]


def test_prepare_english_katakana_conversion_tinyllama_forwards_on_progress_to_convert_words(monkeypatch):
    import vocal_analysis.recognizer as R

    monkeypatch.setattr(R, "uncached_target_words", lambda text, method: ["hello"])
    monkeypatch.setattr(R, "_release_content_recognizer_pipeline", lambda: None)
    captured = {}
    monkeypatch.setattr(
        R, "convert_words",
        lambda words, method, on_progress=None: captured.setdefault("on_progress", on_progress),
    )

    sentinel = lambda note: None  # noqa: E731
    R._prepare_english_katakana_conversion("hello を見上げて", "tinyllama-katakana-converter", on_progress=sentinel)

    assert captured["on_progress"] is sentinel


def test_prepare_english_katakana_conversion_tinyllama_skips_release_when_no_uncached_words(monkeypatch):
    import vocal_analysis.recognizer as R

    monkeypatch.setattr(R, "uncached_target_words", lambda text, method: [])
    monkeypatch.setattr(
        R, "_release_content_recognizer_pipeline",
        lambda: (_ for _ in ()).throw(AssertionError("未変換対象語が無ければ解放しない")),
    )
    monkeypatch.setattr(
        R, "convert_words",
        lambda words, method: (_ for _ in ()).throw(AssertionError("未変換対象語が無ければ変換しない")),
    )

    R._prepare_english_katakana_conversion("hello を見上げて", "tinyllama-katakana-converter")
