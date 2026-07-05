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
