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


# --- §5.2 手順3: 音素列の結合(pau挿入) ---


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


# --- §5.2 手順4: G2P記号→音素モデル語彙のトークンID変換 ---


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

    # §5.2 手順4: 写像表に無い記号が現れたら RecognitionError で停止する(黙って捨てない)。
    with pytest.raises(RecognitionError):
        _g2p_symbols_to_token_ids(["xx"], {"<pad>": 0}, blank_token_id=0)


def test_g2p_symbols_to_token_ids_missing_vocab_entry_raises_recognition_error():
    from vocal_analysis.recognizer import RecognitionError, _g2p_symbols_to_token_ids

    # 写像表上は既知の記号でも、実際のモデル語彙にその記号が無ければ RecognitionError。
    with pytest.raises(RecognitionError):
        _g2p_symbols_to_token_ids(["a"], {"<pad>": 0}, blank_token_id=0)


# --- §5.2 手順5: 強制アライメント(Viterbi) ---


def test_forced_align_single_state_stays_for_all_frames():
    from vocal_analysis.recognizer import _forced_align

    log_probs = np.array([[0.1], [-1.0], [0.2]])

    path = _forced_align(log_probs, token_ids=[0])

    assert path == [0, 0, 0]


def test_forced_align_follows_dominant_emission_monotonically():
    from vocal_analysis.recognizer import _forced_align

    # 状態0(blank)がフレーム0-1で優勢、状態1(母音)がフレーム2-3で優勢になるよう設計した
    # 対数確率行列。最尤経路は手計算で [0,0,1,1] と確認済み(状態を飛ばさず単調に進む)。
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

    # §5.2 手順5: フレーム数がトークン数未満だと末尾トークンへ理論上到達不能。RecognitionErrorで停止する。
    log_probs = np.zeros((2, 1))

    with pytest.raises(RecognitionError):
        _forced_align(log_probs, token_ids=[0, 0, 0])


# --- §5.2 手順6・7: 区間の確定とSegment化 ---


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

    # 連続する2状態がともに母音"a"(長母音が2モーラに分かれた場合等)は1区間へ結合する(§5.2手順6)。
    segments = _path_to_segments([0, 0, 1, 1], ["a", "a"], frame_duration_sec=0.02)

    assert len(segments) == 1
    assert segments[0].type == "vowel"
    assert segments[0].phoneme == "a"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.08)


def test_path_to_segments_merges_pau_and_cl_as_same_gap():
    from vocal_analysis.recognizer import _path_to_segments

    # pau由来とcl由来はいずれもblank扱いで同一視し、1つのgap区間へ結合する(§5.2手順6)。
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
