"""S2 音素認識のテスト(vocal_analysis.md §5・§5.1・§5.2)。

母音/子音判定基準(アダプタ共通)と、複合構成(内容認識+G2P+強制アライメント)の区間化を担う純関数の
核を合成フィクスチャで決定論的に検証する。実モデル呼び出し(wav2vec2・Whisper)を伴う統合部分
(recognize 関数本体)は別途扱う。
"""

import numpy as np
import pytest


# §5.1 の母音記号基準集合(IPA母音チャートの基本母音28記号+R音性母音2記号+拡張母音記号1)を
# 過不足なく列挙する(仕様の有限集合をそのまま固定する)。
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

    # §5.1: 母音記号基準集合(31記号)は単体でいずれも母音と判定される。
    assert _classify_symbol(symbol) == "vowel"


@pytest.mark.parametrize(
    "symbol",
    [
        "aː", "iː", "uː", "eː", "oː",  # 長音記号付き(NFD正規化で基底文字が先頭になる)
        "ɑ̃", "ɛ̃",  # 鼻音化(結合チルダ)
        "aɪ", "eɪ", "oʊ", "aʊ",  # 下降二重母音(母音記号で始まる)
        "ɑːɹ", "ɔːɹ",  # 母音+R音性子音の連続(米語のr音化母音表記)
        "a1", "e2",  # 中国語声調番号付き表記
    ],
)
def test_classify_symbol_modified_and_compound_vowel_examples(symbol):
    from vocal_analysis.recognizer import _classify_symbol

    # §5.1: NFD正規化後の先頭基底文字が母音記号基準集合に含まれる限り、修飾付き・複合表記でも
    # 母音側に判定する。
    assert _classify_symbol(symbol) == "vowel"


@pytest.mark.parametrize(
    "symbol",
    ["n", "s", "t", "k", "d", "m", "p", "z", "f", "v", "b", "ŋ", "θ", "ʃ", "ʒ", "ja", "ju", "wa", ""],
)
def test_classify_symbol_consonant_examples(symbol):
    from vocal_analysis.recognizer import _classify_symbol

    # §5.1: 母音記号基準集合に無い記号は子音。先頭が接近音(j/w)始まりの ja/ju/wa のような
    # 記号も、先頭基底文字が基準集合に無いため子音側になる。空文字列も子音側。
    assert _classify_symbol(symbol) == "consonant"


# --- §5.2 手順1・2: 無音検出による区間分割 ---


def test_frame_rms_computes_rms_per_frame():
    from vocal_analysis.recognizer import _frame_rms

    # 100ms=1600サンプル@16kHzのフレーム2つ分。フレーム1は振幅0.5一定、フレーム2は無音。
    mono = np.concatenate([np.full(1600, 0.5, dtype=np.float32), np.zeros(1600, dtype=np.float32)])

    result = _frame_rms(mono, sample_rate=16000, frame_sec=0.1)

    assert result.shape == (2,)
    assert result[0] == pytest.approx(0.5)
    assert result[1] == pytest.approx(0.0)


def test_frame_rms_truncates_incomplete_final_frame():
    from vocal_analysis.recognizer import _frame_rms

    # 1600サンプルちょうど(1フレーム分)+末尾に800サンプル(不完全フレーム、切り捨てる)。
    mono = np.concatenate([np.full(1600, 0.5, dtype=np.float32), np.full(800, 0.9, dtype=np.float32)])

    result = _frame_rms(mono, sample_rate=16000, frame_sec=0.1)

    assert result.shape == (1,)


def test_frame_rms_empty_input_returns_empty_array():
    from vocal_analysis.recognizer import _frame_rms

    assert _frame_rms(np.array([], dtype=np.float32), sample_rate=16000, frame_sec=0.1).shape == (0,)


def test_silence_threshold_is_30db_below_95th_percentile():
    from vocal_analysis.recognizer import _silence_threshold

    frame_rms = np.array([1.0] * 95 + [2.0] * 5)
    expected_peak = np.percentile(frame_rms, 95)  # numpyの補間規則をそのまま基準値に使う

    threshold = _silence_threshold(frame_rms)

    assert threshold == pytest.approx(expected_peak * (10 ** (-30.0 / 20.0)))


def test_silence_threshold_zero_peak_gives_zero_threshold():
    from vocal_analysis.recognizer import _silence_threshold

    # §5.2手順1: 全フレームRMSが0(完全無音入力)の場合、しきい値も0になる。
    assert _silence_threshold(np.zeros(10)) == pytest.approx(0.0)


def test_silence_threshold_empty_input_is_zero():
    from vocal_analysis.recognizer import _silence_threshold

    assert _silence_threshold(np.array([])) == pytest.approx(0.0)


def test_detect_silence_split_points_finds_midpoint_of_qualifying_run():
    from vocal_analysis.recognizer import _detect_silence_split_points

    # 前後1秒の大音量(振幅0.5)に挟まれた1秒の無音(§5.2手順1の最小長0.6秒を満たす)。
    loud = np.full(16000, 0.5, dtype=np.float32)
    silence = np.zeros(16000, dtype=np.float32)
    mono = np.concatenate([loud, silence, loud])

    splits = _detect_silence_split_points(mono, sample_rate=16000)

    assert len(splits) == 1
    assert splits[0] == pytest.approx(1.5)  # 無音区間[1.0, 2.0)の中点


def test_detect_silence_split_points_ignores_short_silence_gap():
    from vocal_analysis.recognizer import _detect_silence_split_points

    # 無音区間が0.3秒(§5.2手順1の最小長0.6秒未満)なので分割点にならない。
    loud = np.full(16000, 0.5, dtype=np.float32)
    silence = np.zeros(4800, dtype=np.float32)
    mono = np.concatenate([loud, silence, loud])

    assert _detect_silence_split_points(mono, sample_rate=16000) == []


def test_detect_silence_split_points_handles_trailing_silence_run():
    from vocal_analysis.recognizer import _detect_silence_split_points

    # 末尾まで続く無音区間(1秒)も分割点として検出する。
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

    # 分割点3.0秒による最初の区間[0,3.0)が最小長1.5秒未満(実際は3.0秒なので単体では十分だが、
    # 分割点0.5秒を加えると先頭区間[0,0.5)が最小長未満になり、次の分割点まで結合される)。
    bounds = _build_segment_bounds(duration_sec=10.0, split_points=[0.5, 5.0])

    assert bounds == [(0.0, 5.0), (5.0, 10.0)]


def test_build_segment_bounds_keeps_short_final_segment_when_no_neighbor():
    from vocal_analysis.recognizer import _build_segment_bounds

    # 音声全体が1.0秒(最小長1.5秒未満)で分割点が無い場合、結合先が無いのでそのまま採用する。
    assert _build_segment_bounds(duration_sec=1.0, split_points=[]) == [(0.0, 1.0)]


def test_build_segment_bounds_splits_too_long_segment_evenly():
    from vocal_analysis.recognizer import _build_segment_bounds

    # 60秒の区間(最大長25秒を超過)は3等分(20秒ずつ)される。
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


def test_is_hallucinated_phoneme_density_below_threshold_is_not_hallucinated():
    from vocal_analysis.recognizer import _HALLUCINATION_PHONEME_RATE, _is_hallucinated_phoneme_density

    # しきい値ちょうど(20音素/秒)は超過ではない。
    assert _is_hallucinated_phoneme_density(
        phoneme_count=round(_HALLUCINATION_PHONEME_RATE * 2.0), duration_sec=2.0) is False


def test_is_hallucinated_phoneme_density_above_threshold_is_hallucinated():
    from vocal_analysis.recognizer import _is_hallucinated_phoneme_density

    # 実測(反復幻覚)相当: 21.15秒に863音素 ≈ 40.8音素/秒。
    assert _is_hallucinated_phoneme_density(phoneme_count=863, duration_sec=21.15) is True


def test_is_hallucinated_phoneme_density_zero_duration_is_not_hallucinated():
    from vocal_analysis.recognizer import _is_hallucinated_phoneme_density

    assert _is_hallucinated_phoneme_density(phoneme_count=100, duration_sec=0.0) is False


# --- プロンプト文エコーの除去 ---


def test_strip_prompt_echo_removes_exact_prompt_sentences():
    from vocal_analysis.recognizer import _strip_prompt_echo

    text, removed = _strip_prompt_echo("かんじは つかわないでください。" * 24)
    assert text == ""
    assert removed is True


def test_strip_prompt_echo_removes_trailing_fragment_after_echo():
    from vocal_analysis.recognizer import _strip_prompt_echo

    # 区間末尾で切れたエコーの断片(プロンプト文と共通接頭辞5文字以上)は、エコー除去が起きた
    # 場合に限り除去される。
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


def test_strip_prompt_echo_keeps_short_prefix_fragment_without_echo():
    from vocal_analysis.recognizer import _strip_prompt_echo

    # エコー除去が起きていないテキストの通常文は、プロンプト文と接頭辞が重なっても除去しない。
    original = "かんじはじめた こころ"
    text, removed = _strip_prompt_echo(original)
    assert text == original
    assert removed is False


# --- 末尾反復の検出と救済 ---


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


def test_find_suffix_repetition_ignores_short_runs():
    from vocal_analysis.recognizer import _find_suffix_repetition

    # 繰り返し数3未満・総長8文字未満は反復として検出しない(実在の歌詞の軽い反復を巻き込まない)。
    assert _find_suffix_repetition("たのしいね たのしいね") is None


# --- 書き起こしの後処理(エコー除去→トリガ式リトライ→反復救済) ---


def _resolve(monkeypatch, text, words=None, duration=10.0, retry_result=None, retry_enabled=True,
             g2p=None, transcribe_calls=None, text_only_result=None, text_only_calls=None):
    """_resolve_transcription をモック済み依存で呼ぶ共通ハーネス。

    retry_result が None のとき _transcribe_segment 呼び出し(_resolve_transcription は本来
    呼ばない。呼んだら実装退行)が起きたら失敗させる。text_only_result が None のとき
    トリガ式リトライ(_transcribe_text_only)呼び出しが起きたら失敗させる。
    g2p 省略時は「空白以外の1文字=1音素(母音扱い)」の決定的な模擬。
    """
    from vocal_analysis import recognizer as R

    primary = R.DEFAULT_CONTENT_RECOGNIZER_MODEL

    def fake_transcribe(pipeline, samples):
        if transcribe_calls is not None:
            transcribe_calls.append(pipeline)
        if retry_result is None:
            raise AssertionError("リトライが呼ばれてはならないケースで _transcribe_segment が呼ばれた")
        return retry_result

    def fake_transcribe_text_only(samples, model):
        if text_only_calls is not None:
            text_only_calls.append(model)
        if text_only_result is None:
            raise AssertionError("リトライが呼ばれてはならないケースで _transcribe_text_only が呼ばれた")
        return text_only_result

    monkeypatch.setattr(R, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(R, "_transcribe_text_only", fake_transcribe_text_only)
    monkeypatch.setattr(R, "_g2p", g2p or (lambda t: ["a"] * len(t.replace(" ", ""))))
    return R._resolve_transcription(
        np.zeros(16000, dtype=np.float32), duration, text, words, primary, retry_enabled)


def test_resolve_transcription_retries_when_echo_leaves_empty_text(monkeypatch):
    calls = []
    text, words = _resolve(
        monkeypatch, "かんじは つかわないでください。",
        words=[("かんじは", 0.0, 1.0)], duration=15.0,
        text_only_result="げんきです", text_only_calls=calls)
    from vocal_analysis.recognizer import DEFAULT_CONTENT_RECOGNIZER_MODEL

    assert text == "げんきです"
    assert words is None
    # トリガ式リトライは主モデルへ・プロンプト無し・タイムスタンプ無し・1回だけ。
    assert len(calls) == 1
    assert calls[0] is DEFAULT_CONTENT_RECOGNIZER_MODEL


def test_resolve_transcription_does_not_retry_on_ascii_words(monkeypatch):
    # ASCII英字はリトライ条件に含めない(リトライ条件はエコー幻覚と反復幻覚のみ)。
    # text_only_result=None のためリトライ呼び出しが起きればハーネスが失敗させる。
    text, _words = _resolve(
        monkeypatch, "hello world つづける", duration=10.0,
        retry_result=None, text_only_result=None)
    assert text == "hello world つづける"


def test_resolve_transcription_retries_on_high_density(monkeypatch):
    # 反復幻覚(密度超過)はリトライのトリガ。密度: 模擬G2Pは空白以外1文字=1音素。
    # 40文字/1.0秒=40音素/秒 > 20。
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


def test_resolve_transcription_disabled_retry_rescues_suffix_repetition(monkeypatch):
    # retry=False でも反復救済(密度超過ゲート)は効く。先頭部が残る形は先頭部だけを採用する。
    text, _words = _resolve(
        monkeypatch, "アイウエオカキクケコ" + "ラ" * 90, duration=1.0,
        retry_result=None, retry_enabled=False)
    assert text == "アイウエオカキクケコ"


def test_resolve_transcription_normalizes_whole_text_repetition(monkeypatch):
    # 全文反復は 区間長×3.5モーラ/秒÷単位モーラ数 へ個数正規化する。
    # 模擬G2P: 1文字=1音素(母音扱い)→ 単位「ラ」=1モーラ。10秒×3.5=35個。
    from vocal_analysis import recognizer as R

    def g2p(t):
        return ["a"] * len(t.replace(" ", ""))

    monkeypatch.setattr(R, "_g2p", g2p)
    text, words = R._resolve_transcription(
        np.zeros(16000, dtype=np.float32), 10.0, "ラ" * 400, None,
        R.DEFAULT_CONTENT_RECOGNIZER_MODEL, False)
    assert text == "ラ" * 35
    assert words is None


def test_resolve_transcription_rescue_keeps_normal_density_text(monkeypatch):
    # 密度がしきい値以下のテキストには反復救済を一切適用しない(実在の歌詞の反復を守る)。
    text, words = _resolve(
        monkeypatch, "すき すき すき", words=[("すき", 0.0, 0.5)], duration=10.0,
        retry_result=None, retry_enabled=False)
    assert text == "すき すき すき"
    assert words == [("すき", 0.0, 0.5)]


def test_voiced_trim_bounds_trims_leading_and_trailing_silence_with_margin():
    from vocal_analysis.recognizer import _voiced_trim_bounds

    # 16kHz・100msフレーム。先頭2.0秒無音 + 有声1.0秒 + 末尾3.0秒無音。
    sr = 16000
    silence_head = np.zeros(2 * sr, dtype=np.float32)
    voiced = np.full(1 * sr, 0.5, dtype=np.float32)
    silence_tail = np.zeros(3 * sr, dtype=np.float32)
    samples = np.concatenate([silence_head, voiced, silence_tail])

    lo, hi = _voiced_trim_bounds(samples, sr, threshold=0.01)

    # 有声スパン[2.0,3.0]秒の外側に余白100ms → [1.9,3.1]秒。
    assert lo == round(1.9 * sr)
    assert hi == round(3.1 * sr)


def test_voiced_trim_bounds_clamps_margin_to_segment_edges():
    from vocal_analysis.recognizer import _voiced_trim_bounds

    # 全体が有声。余白を加えても区間の外へ出ない。
    sr = 16000
    samples = np.full(1 * sr, 0.5, dtype=np.float32)

    lo, hi = _voiced_trim_bounds(samples, sr, threshold=0.01)

    assert lo == 0
    assert hi == len(samples)


def test_voiced_trim_bounds_no_voiced_frame_returns_whole_segment():
    from vocal_analysis.recognizer import _voiced_trim_bounds

    # フレーム単位では全て以下(端ケース)。トリムせず全体を返す。
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


# --- §5.2 手順3: 単語タイムスタンプの単調化・抽出 ---


def test_sanitize_word_timestamps_clamps_to_duration_range():
    from vocal_analysis.recognizer import _sanitize_word_timestamps

    # 開始が負・終了が範囲長を超える値は [0, duration_sec] へクランプする。
    result = _sanitize_word_timestamps([("a", -0.5, 3.0)], duration_sec=2.0)

    assert result == [("a", 0.0, 2.0)]


def test_sanitize_word_timestamps_enforces_start_order():
    from vocal_analysis.recognizer import _sanitize_word_timestamps

    # 単語2の開始(0.3)が単語1の開始(0.5)より前退している場合、単語1の開始以上へ引き上げる。
    result = _sanitize_word_timestamps([("a", 0.5, 0.8), ("b", 0.3, 0.9)], duration_sec=2.0)

    assert result[0] == ("a", 0.5, 0.8)
    assert result[1][1] == pytest.approx(0.5)


def test_sanitize_word_timestamps_enforces_minimum_length():
    from vocal_analysis.recognizer import _MIN_WORD_DURATION_SEC, _sanitize_word_timestamps

    # 終了が開始+最小長に満たない場合、開始+最小長へ引き上げる。
    result = _sanitize_word_timestamps([("a", 1.0, 1.0)], duration_sec=2.0)

    assert result[0][2] == pytest.approx(1.0 + _MIN_WORD_DURATION_SEC)


def test_sanitize_word_timestamps_clamp_precedes_minimum_length_enforcement():
    from vocal_analysis.recognizer import _sanitize_word_timestamps

    # §5.2手順3: 「まず[0,duration_sec]へクランプし、続けて…終了時刻を自身の開始時刻+最小長以上へ」
    # の順序どおりだと、終了は先にduration(2.0)へクランプされてから+最小長(0.05)されるため、
    # 最終的な終了(2.03)がduration自体を上回りうる(クランプを後段でやり直すなら2.0のまま)。
    result = _sanitize_word_timestamps([("a", 1.98, 2.5)], duration_sec=2.0)

    assert result == [("a", 1.98, pytest.approx(2.03))]


def test_extract_word_timestamps_skips_empty_text_and_missing_start():
    from vocal_analysis.recognizer import _extract_word_timestamps

    chunks = [
        {"text": "あ", "timestamp": (0.0, 0.5)},
        {"text": "  ", "timestamp": (0.5, 1.0)},  # 空白のみ: 読み飛ばす
        {"text": "い", "timestamp": (None, 1.5)},  # 開始無し: 読み飛ばす
        {"text": "う", "timestamp": (1.5, 2.0)},
    ]

    result = _extract_word_timestamps(chunks, duration_sec=2.0)

    assert [w[0] for w in result] == ["あ", "う"]


def test_extract_word_timestamps_fills_missing_end_with_minimum_length():
    from vocal_analysis.recognizer import _MIN_WORD_DURATION_SEC, _extract_word_timestamps

    # 生成が単語途中で打ち切られ終了時刻が無い場合、開始時刻+最小長で補う。
    chunks = [{"text": "あ", "timestamp": (1.0, None)}]

    result = _extract_word_timestamps(chunks, duration_sec=2.0)

    assert result == [("あ", 1.0, 1.0 + _MIN_WORD_DURATION_SEC)]


# --- §5.2 手順5: 音素列の組み立て(pau挿入) ---


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

    # Whisperのチャンクがテキストを持たずG2P結果が空でも、チャンク境界のpau挿入は行う。
    result = _assemble_phoneme_sequence([[], ["a"]])

    assert result == ["pau", "pau", "a", "pau"]


def test_assemble_phoneme_sequence_no_chunks_is_leading_and_trailing_pau_only():
    from vocal_analysis.recognizer import _assemble_phoneme_sequence

    result = _assemble_phoneme_sequence([])

    assert result == ["pau", "pau"]


# --- §5.2 手順5: 単語窓の割り当て(単語タイムスタンプ取得時) ---


def test_assemble_with_word_windows_single_word():
    from vocal_analysis.recognizer import _assemble_with_word_windows

    # 単語「あ」(音素["a"])が[1.0,1.5]秒。余白0.5秒。
    seq, windows = _assemble_with_word_windows([(["a"], 1.0, 1.5)], margin_sec=0.5, duration_sec=3.0)

    assert seq == ["pau", "a", "pau"]
    assert windows[0] == (0.0, 1.5)  # 先頭pau: [0, 最初の開始+余白]
    assert windows[1] == (0.5, 2.0)  # 単語: [開始-余白, 終了+余白]
    assert windows[2] == (1.0, 3.0)  # 末尾pau: [最後の終了-余白, duration_sec]


def test_assemble_with_word_windows_inter_word_pau_window():
    from vocal_analysis.recognizer import _assemble_with_word_windows

    # 単語間pau窓は隣接2単語の[前の終了-余白, 次の開始+余白]。
    words_phonemes = [(["a"], 0.5, 1.0), (["k", "i"], 2.0, 2.5)]

    seq, windows = _assemble_with_word_windows(words_phonemes, margin_sec=0.3, duration_sec=3.0)

    assert seq == ["pau", "a", "pau", "k", "i", "pau"]
    inter_word_window = windows[2]
    assert inter_word_window == pytest.approx((0.7, 2.3))
    # 単語内の全音素記号(句読点由来のpauを含む)は自身の単語の窓を共有する。
    assert windows[3] == windows[4]


def test_assemble_with_word_windows_swaps_inverted_window():
    from vocal_analysis.recognizer import _assemble_with_word_windows

    # 単調化は開始順序のみ保証するため、前の単語の終了が次の単語の開始を余白の2倍を超えて
    # 上回る(重なる)場合、単語間pau窓は下端が上端を上回る。小さい方を下端として入れ替える。
    words_phonemes = [(["a"], 0.0, 2.0), (["i"], 2.1, 2.5)]  # 重なり(2.0-2.1=1.9) > 余白の2倍(1.0)

    _, windows = _assemble_with_word_windows(words_phonemes, margin_sec=0.5, duration_sec=3.0)

    inter_word_window = windows[2]
    assert inter_word_window[0] <= inter_word_window[1]


# --- §5.2 手順6: G2P記号→音素モデル語彙のトークンID変換 ---


def test_g2p_symbols_to_token_ids_maps_known_symbols():
    from vocal_analysis.recognizer import _g2p_symbols_to_token_ids

    vocab = {"<pad>": 0, "a": 5, "k": 7}

    result = _g2p_symbols_to_token_ids(["a", "k"], vocab, blank_token_id=0)

    assert result == [5, 7]


def test_g2p_symbols_to_token_ids_maps_pau_and_cl_to_blank():
    from vocal_analysis.recognizer import _g2p_symbols_to_token_ids

    # §5.2: pau・cl は語彙記号への対応付けを持たず、blank トークンへ変換する。
    vocab = {"<pad>": 0, "a": 5}

    result = _g2p_symbols_to_token_ids(["pau", "cl"], vocab, blank_token_id=0)

    assert result == [0, 0]


def test_g2p_symbols_to_token_ids_devoiced_vowels_collapse_to_voiced():
    from vocal_analysis.recognizer import _g2p_symbols_to_token_ids

    # §5.2 写像表: 無声化母音 I/U は有声母音と同じ語彙記号(i/ɯ)へ収束する。
    vocab = {"<pad>": 0, "i": 3, "ɯ": 4}

    result = _g2p_symbols_to_token_ids(["I", "U"], vocab, blank_token_id=0)

    assert result == [3, 4]


def test_g2p_symbols_to_token_ids_unmapped_symbol_raises_recognition_error():
    from vocal_analysis.recognizer import RecognitionError, _g2p_symbols_to_token_ids

    # §5.2 手順6: 写像表に無い記号が現れたら RecognitionError で停止する(黙って捨てない)。
    with pytest.raises(RecognitionError):
        _g2p_symbols_to_token_ids(["xx"], {"<pad>": 0}, blank_token_id=0)


def test_g2p_symbols_to_token_ids_missing_vocab_entry_raises_recognition_error():
    from vocal_analysis.recognizer import RecognitionError, _g2p_symbols_to_token_ids

    # 写像表上は既知の記号でも、実際のモデル語彙にその記号が無ければ RecognitionError。
    with pytest.raises(RecognitionError):
        _g2p_symbols_to_token_ids(["a"], {"<pad>": 0}, blank_token_id=0)


# --- §5.2 手順7: 強制アライメント(バンド制限Viterbi) ---


def test_voiced_blank_penalty_applies_only_to_voiced_frames_blank_column():
    from vocal_analysis.recognizer import _VOICED_BLANK_PENALTY, _apply_voiced_blank_penalty

    # 16kHz・20msフレーム(320サンプル)×4フレーム。フレーム0-1は無音、フレーム2-3は有声。
    sr = 16000
    samples = np.concatenate([np.zeros(640, dtype=np.float32), np.full(640, 0.5, dtype=np.float32)])
    log_probs = np.zeros((4, 3))

    result = _apply_voiced_blank_penalty(log_probs, samples, threshold=0.01, blank_token_id=1)

    # 有声フレーム(2,3)のblank列(id=1)だけがペナルティ分下がる。他は不変。
    expected = np.zeros((4, 3))
    expected[2, 1] = -_VOICED_BLANK_PENALTY
    expected[3, 1] = -_VOICED_BLANK_PENALTY
    assert np.array_equal(result, expected)
    assert np.array_equal(log_probs, np.zeros((4, 3)))  # 入力は破壊しない


def test_voiced_blank_penalty_handles_frame_count_mismatch():
    from vocal_analysis.recognizer import _apply_voiced_blank_penalty

    # 音声由来のフレーム数(3)と対数確率行列のフレーム数(4)がずれても短い方だけ処理して例外を出さない
    # (音素モデルの畳み込み丸めでずれうる)。
    samples = np.full(960, 0.5, dtype=np.float32)  # 20msフレーム(320サンプル)×3フレーム分
    log_probs = np.zeros((4, 2))

    result = _apply_voiced_blank_penalty(log_probs, samples, threshold=0.01, blank_token_id=0)

    assert result.shape == (4, 2)
    assert result[3, 0] == 0.0  # 音声側に対応フレームが無い行は不変
    assert result[0, 0] < 0.0  # 有声フレームの行は下がる


def test_expand_min_stay_expands_phoneme_tokens_only():
    from vocal_analysis.recognizer import _MIN_STAY_FRAMES, _expand_min_stay

    # blank(id=0)は1状態のまま、非blank(音素)は最小滞在フレーム数ぶんの連鎖サブ状態になる。
    sub_ids, sub_to_token = _expand_min_stay([0, 7, 0], blank_token_id=0, num_frames=100)

    assert sub_ids == [0] + [7] * _MIN_STAY_FRAMES + [0]
    assert sub_to_token == [0] + [1] * _MIN_STAY_FRAMES + [2]


def test_expand_min_stay_reduces_stay_when_frames_are_scarce():
    from vocal_analysis.recognizer import _expand_min_stay

    # フレーム数がサブ状態総数に足りない場合、成立する最大の滞在数へ引き下げる。
    # blank2個+音素1個・6フレーム → 滞在数 = (6-2)//1 = 4。
    sub_ids, sub_to_token = _expand_min_stay([0, 7, 0], blank_token_id=0, num_frames=6)

    assert sub_ids == [0, 7, 7, 7, 7, 0]
    assert sub_to_token == [0, 1, 1, 1, 1, 2]


def test_expand_min_stay_never_reduces_below_one():
    from vocal_analysis.recognizer import _expand_min_stay

    # 極端にフレームが少なくても滞在数は1未満にしない(不足自体は _forced_align が検出して停止する)。
    sub_ids, _ = _expand_min_stay([0, 7, 0], blank_token_id=0, num_frames=2)

    assert sub_ids == [0, 7, 0]


def test_expand_min_stay_all_blank_sequence_is_unchanged():
    from vocal_analysis.recognizer import _expand_min_stay

    sub_ids, sub_to_token = _expand_min_stay([0, 0], blank_token_id=0, num_frames=10)

    assert sub_ids == [0, 0]
    assert sub_to_token == [0, 1]


# --- §5.2 手順7: 最小滞在の局所適応(単語タイムスタンプ取得時) ---


def test_expand_min_stay_local_uses_word_duration_over_phoneme_count():
    from vocal_analysis.recognizer import _expand_min_stay_local

    # token_ids = [pau, a, i, pau](単語1個・音素2個)。単語の実時間0.12s(6フレーム)を
    # 音素数2で割った3フレームが最小滞在になる。
    words_phonemes = [(["a", "i"], 0.0, 0.12)]
    sub_ids, sub_to_token = _expand_min_stay_local([0, 1, 2, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0, 1, 1, 1, 2, 2, 2, 0]
    assert sub_to_token == [0, 1, 1, 1, 2, 2, 2, 3]


def test_expand_min_stay_local_varies_per_word_within_same_chunk():
    from vocal_analysis.recognizer import _expand_min_stay_local

    # 単語1「あ」(音素1個・0.12s=6フレーム→stay6)と単語2「い」(音素1個・0.02s=1フレーム→stay1)。
    # チャンク全体で1つの平均を使う _expand_min_stay と異なり、テンポが速い単語だけ短縮される。
    # token_ids = [pau, a, pau(単語間), i, pau(末尾)]
    words_phonemes = [(["a"], 0.0, 0.12), (["i"], 0.12, 0.14)]
    sub_ids, sub_to_token = _expand_min_stay_local([0, 1, 0, 2, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0] + [1] * 6 + [0] + [2] * 1 + [0]
    assert sub_to_token == [0] + [1] * 6 + [2] + [3] * 1 + [4]


def test_expand_min_stay_local_clamps_to_global_max_stay():
    from vocal_analysis.recognizer import _MIN_STAY_FRAMES, _expand_min_stay_local

    # 実時間が長く音素数が少ない単語でも、最小滞在は _MIN_STAY_FRAMES を超えない。
    words_phonemes = [(["a"], 0.0, 2.0)]  # 100フレーム分の実時間・音素1個
    sub_ids, _ = _expand_min_stay_local([0, 1, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0] + [1] * _MIN_STAY_FRAMES + [0]


def test_expand_min_stay_local_never_reduces_below_one():
    from vocal_analysis.recognizer import _expand_min_stay_local

    # 実時間が音素数に対して極端に短くても、最小滞在は1未満にしない。
    words_phonemes = [(["a", "i", "u"], 0.0, 0.02)]  # 1フレームに音素3個
    sub_ids, _ = _expand_min_stay_local([0, 1, 2, 3, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0, 1, 2, 3, 0]


def test_expand_min_stay_local_rounds_floating_point_boundary_correctly():
    from vocal_analysis.recognizer import _expand_min_stay_local

    # (0.06-0.02)/0.02 は浮動小数点誤差で1.9999999999999996になり、floor()なら1へ切り捨てて
    # しまう境界値。真の四捨五入(int(x+0.5))で正しく2フレームへ丸められることを確認する。
    words_phonemes = [(["a"], 0.02, 0.06)]
    sub_ids, _ = _expand_min_stay_local([0, 1, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0, 1, 1, 0]


def test_expand_min_stay_local_rounds_half_up_not_half_to_even():
    from vocal_analysis.recognizer import _expand_min_stay_local

    # 単語の最小長クランプ(§5.2手順3の_MIN_WORD_DURATION_SEC=0.05秒)により、実時間/音素数が
    # ちょうど2.5フレーム(0.05/0.02)になる商へ実際に到達しうる。Python組み込みのround()は
    # 偶数丸めで2.5→2になるが、実装は真の四捨五入(0.5は常に切り上げ)を使うため3になる。
    words_phonemes = [(["a"], 0.0, 0.05)]
    sub_ids, _ = _expand_min_stay_local([0, 1, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0, 1, 1, 1, 0]


def test_expand_min_stay_local_empty_phonemes_word_contributes_no_substates():
    from vocal_analysis.recognizer import _expand_min_stay_local

    # 音素0個の単語(句読点のみ等)を挟んでも、その単語自身はトークンを持たないため
    # token_ids・words_phonemesの対応関係は崩れない(隣接pauのみが1状態で並ぶ)。
    words_phonemes = [([], 0.0, 0.02), (["a"], 0.02, 0.14)]
    sub_ids, sub_to_token = _expand_min_stay_local([0, 0, 1, 0], words_phonemes, blank_token_id=0)

    assert sub_ids == [0, 0] + [1] * 6 + [0]
    assert sub_to_token == [0, 1] + [2] * 6 + [3]


def test_expand_min_stay_local_word_internal_blank_symbol_stays_one_frame():
    from vocal_analysis.recognizer import _expand_min_stay_local

    # 単語内に促音(cl。blank写像)を含む場合、その単語の音素記号列の要素数(blank記号を含む4個)で
    # 実時間を割ってstay(0.24s=12フレーム/4個=3)を計算するが、blank記号自身は展開時に常に
    # 1フレームへ落ちる(非blank記号だけがstay分だけ展開される)。
    words_phonemes = [(["a", "cl", "t", "e"], 0.0, 0.24)]
    sub_ids, sub_to_token = _expand_min_stay_local(
        [0, 5, 0, 7, 6, 0], words_phonemes, blank_token_id=0
    )

    assert sub_ids == [0] + [5] * 3 + [0] + [7] * 3 + [6] * 3 + [0]
    assert sub_to_token == [0] + [1] * 3 + [2] + [3] * 3 + [4] * 3 + [5]


def test_min_stay_expansion_forces_phoneme_to_occupy_expanded_frames():
    from vocal_analysis.recognizer import _MIN_STAY_FRAMES, _expand_min_stay, _forced_align

    # blankが全面的に優勢で音素(id=1)は1フレームしか優勢でない放出確率。展開なしなら
    # 音素に1フレームだけ滞在する経路が最尤になる構成だが、サブ状態連鎖(展開後は
    # blank2個+音素_MIN_STAY_FRAMES個の合計フレームに経路の自由度が残る。フレーム総数は
    # _MIN_STAY_FRAMES より十分大きく取り、(フレーム数-blank数)がボトルネックにならないようにする)
    # により元トークンの滞在は最小滞在フレーム数以上になる。
    num_frames = _MIN_STAY_FRAMES + 4
    log_probs = np.array(
        [[5.0, -5.0]] * 4 + [[-5.0, 5.0]] + [[5.0, -5.0]] * (num_frames - 5)
    )
    plain_path = _forced_align(log_probs, [0, 1, 0])
    assert sum(1 for s in plain_path if s == 1) == 1  # 展開なしでは1フレーム通過が最尤

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

    # 状態0(blank)がフレーム0-1で優勢、状態1(母音)がフレーム2-3で優勢になるよう設計した
    # 対数確率行列。状態を飛ばさず単調に進むため、最尤経路は [0,0,1,1] になる。
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

    # §5.2 手順7: フレーム数がトークン数未満だと末尾トークンへ理論上到達不能。RecognitionErrorで停止する。
    log_probs = np.zeros((2, 1))

    with pytest.raises(RecognitionError):
        _forced_align(log_probs, token_ids=[0, 0, 0])


def test_forced_align_band_limit_excludes_out_of_band_favorable_evidence():
    from vocal_analysis.recognizer import _forced_align

    # 200フレーム・3状態(pau, 内容, pau)。状態1(内容)の期待位置は中央(99.5フレーム)で、
    # バンド幅は1.0秒=50フレームなので許容範囲は[49.5, 149.5]。内容にとって最も有利な区間
    # (フレーム180-185)はこの範囲の外にあるため、バンド制限により状態1はそこへ到達できず、
    # 許容範囲内のフレームだけを占有する。
    num_frames = 200
    log_probs = np.zeros((num_frames, 2))
    log_probs[:, 0] = 0.0
    log_probs[:, 1] = -8.0
    log_probs[180:186, 1] = 0.0
    token_ids = [0, 1, 0]

    path = _forced_align(log_probs, token_ids)

    state1_frames = [i for i, state in enumerate(path) if state == 1]
    assert state1_frames  # 状態1は必ずどこかのフレームを占有する
    assert all(49 <= f <= 150 for f in state1_frames)
    assert all(f not in range(180, 186) for f in state1_frames)


# --- §5.2 手順7: 単語窓制約Viterbi(主経路) ---


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
    windows = [(0.0, 10.0), (0.0, 10.0)]  # 制約が緩ければ _forced_align と同じ経路になる

    path = _forced_align_windowed(log_probs, token_ids=[0, 1], windows_sec=windows)

    assert path == [0, 0, 1, 1]


def test_forced_align_windowed_excludes_evidence_outside_window():
    from vocal_analysis.recognizer import _forced_align_windowed

    # 200フレーム・3状態(pau, 内容, pau)。状態1(内容)の窓は[0,3.0]秒(フレーム0-149相当)に
    # 限定する。内容にとって最も有利な区間(フレーム180-185)はこの窓の外にあるため、状態1は
    # そこへ到達できず、窓内のフレームだけを占有する(状態0はフレーム0固定・状態2は末尾固定
    # のため両端の窓は範囲全体を許容する)。
    num_frames = 200
    log_probs = np.zeros((num_frames, 3))
    log_probs[:, 1] = -8.0
    log_probs[180:186, 1] = 0.0
    windows = [(0.0, 4.0), (0.0, 3.0), (0.0, 4.0)]

    path = _forced_align_windowed(log_probs, token_ids=[0, 1, 0], windows_sec=windows)

    state1_frames = [i for i, state in enumerate(path) if state == 1]
    assert state1_frames  # 状態1は必ずどこかのフレームを占有する
    assert all(f < 150 for f in state1_frames)
    assert all(f not in range(180, 186) for f in state1_frames)


def test_forced_align_windowed_raises_when_window_unreachable():
    from vocal_analysis.recognizer import RecognitionError, _forced_align_windowed

    # 状態1の窓がフレーム範囲と一切交差しないため、末尾状態へ到達する経路が存在しない。
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

    # 状態0(pau)は窓[0.0,8.0]秒でフレーム全域に到達可能、状態1(音素)は窓[2.0,8.0]秒で
    # 状態0と広く重なるが開始が遅い。放出は状態0がわずかに優勢(0.0 対 -0.05。blank優勢の
    # 歌唱を模した固定差)なため、窓の到達可能性だけでは遷移点が定まらず、早期遷移ボーナス
    # (状態1の窓内相対位置に応じた加算)が無ければ状態1への遷移を可能な限り遅らせる
    # (フレーム全域の末尾まで留まる)方が総対数確率上有利になる(次のテストで対照確認)。
    # ボーナスが効けば、状態1の窓が開いた直後(2.0秒)へ遷移が引き寄せられる。
    num_frames = int(8.0 / FRAME_DURATION_SEC)
    log_probs = np.zeros((num_frames, 2))
    log_probs[:, 1] = -0.05
    windows = [(0.0, 8.0), (2.0, 8.0)]

    path = _forced_align_windowed(log_probs, token_ids=[0, 1], windows_sec=windows)

    first_state1_frame = path.index(1)
    assert first_state1_frame == round(2.0 / FRAME_DURATION_SEC)


def test_viterbi_monotonic_without_state_bias_delays_transition_to_deadline():
    from vocal_analysis.recognizer import FRAME_DURATION_SEC, _viterbi_monotonic

    # 前テストと同じ放出・窓構成を state_bias=None で直接 _viterbi_monotonic に渡す
    # (_forced_align のフォールバック経路が使う形)。早期遷移ボーナスが無いと、状態0優勢の
    # 放出により状態1への遷移が窓の上端(=フレーム全域の末尾)まで遅延することを確認する。
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


# --- §5.2 手順8・9: 区切りの確定とSegment化 ---


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

    # 連続する2状態がともに母音"a"(長母音が2モーラに分かれた場合等)は1区間へ結合する(§5.2手順8)。
    segments = _path_to_segments([0, 0, 1, 1], ["a", "a"], frame_duration_sec=0.02)

    assert len(segments) == 1
    assert segments[0].type == "vowel"
    assert segments[0].phoneme == "a"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.08)


def test_path_to_segments_merges_pau_and_cl_as_same_gap():
    from vocal_analysis.recognizer import _path_to_segments

    # pau由来とcl由来はいずれもblank扱いで同一視し、1つのgap区間へ結合する(§5.2手順8)。
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


def test_g2p_applies_english_oov_katakana_fallback_before_pyopenjtalk(monkeypatch):
    import pyopenjtalk

    import vocal_analysis.recognizer as recognizer_module

    # _g2p は pyopenjtalk.g2p() へ渡す前に convert_oov_words で元テキストを変換する。recognizer
    # モジュール自身が公開する convert_oov_words 属性をモックする(トップレベルimportで束縛される
    # 想定の実装契約。_g2pが独自に別名でインポートし直すとこのモックは効かず、それ自体が配線漏れの
    # 検出になる)。pyopenjtalk.g2p も呼び出し記録付きモックにし、convert_oov_wordsの戻り値が
    # そのまま渡されることを直接検証する(最終結果の値が偶然一致するだけの弱い検証にしない)。
    oov_calls = []
    monkeypatch.setattr(
        recognizer_module, "convert_oov_words",
        lambda text: oov_calls.append(text) or "スカイ",
    )

    g2p_calls = []

    def fake_g2p(text, kana=None, join=None):
        g2p_calls.append((text, kana, join))
        return ["s", "u", "k", "a", "i"]

    monkeypatch.setattr(pyopenjtalk, "g2p", fake_g2p)

    result = recognizer_module._g2p("空を見上げてsky")

    assert oov_calls == ["空を見上げてsky"]
    assert g2p_calls == [("スカイ", False, False)]
    assert result == ["s", "u", "k", "a", "i"]


@pytest.mark.parametrize(
    "raised,match",
    [
        (ImportError("no nltk"), "nltk"),
        (LookupError("cmudict not found"), "cmudict"),
        (OSError("network error"), "変換モデル"),
    ],
)
def test_g2p_converts_oov_fallback_errors_to_recognition_error(monkeypatch, raised, match):
    import vocal_analysis.recognizer as recognizer_module

    # convert_oov_words内部のCMUdict未取得(LookupError)・変換モデル取得失敗(OSError)・
    # nltk/torch/transformers未導入(ImportError)は、他のモデルロード処理
    # (_load_model_and_processor・_load_content_recognizer_pipeline)と同様に、生の例外のまま
    # 公開APIから漏らさずRecognitionErrorへ変換する。
    def fake_convert_oov_words(text):
        raise raised

    monkeypatch.setattr(recognizer_module, "convert_oov_words", fake_convert_oov_words)

    with pytest.raises(recognizer_module.RecognitionError, match=match):
        recognizer_module._g2p("空を見上げてsky")
