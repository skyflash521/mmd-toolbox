"""S2 音素認識のテスト(vocal_analysis.md §5・§5.1・§5.2)。

CTC出力をセグメント列へ正規化する純関数の核(母音/子音判定・フレーム区間化・60ms吸収)と、複合構成
(内容認識+G2P+強制アライメント)の区間化を担う純関数の核を合成フィクスチャで決定論的に検証する。
実モデル呼び出し(wav2vec2・Whisper)を伴う統合部分(recognize 関数本体)は別途扱う。
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


def test_merge_ctc_frames_collapses_consecutive_same_symbol():
    from vocal_analysis.recognizer import _merge_ctc_frames

    # フレーム長は §5.1 の固定値 20ms(採用モデルの畳み込み総ストライド320サンプル@16kHz)。
    frame_duration = 0.02
    frame_symbols = ["a", "a", "a", "n", "n"]

    segments = _merge_ctc_frames(frame_symbols, frame_duration)

    assert len(segments) == 2
    assert segments[0].type == "vowel"
    assert segments[0].phoneme == "a"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.06)
    assert segments[1].type == "consonant"
    assert segments[1].phoneme == "n"
    assert segments[1].start_sec == pytest.approx(0.06)
    assert segments[1].end_sec == pytest.approx(0.10)


def test_merge_ctc_frames_blank_is_gap():
    from vocal_analysis.recognizer import _merge_ctc_frames

    # §5: gap は認識器が音素を割り当てなかった区間(CTCのblank等)。blank は None で表す。
    frame_symbols = ["a", None, None, "n"]

    segments = _merge_ctc_frames(frame_symbols, 0.02)

    assert [s.type for s in segments] == ["vowel", "gap", "consonant"]
    assert segments[1].phoneme is None


def test_merge_ctc_frames_covers_full_timeline_without_gaps_or_overlaps():
    from vocal_analysis.recognizer import _merge_ctc_frames

    frame_symbols = ["a", "n", None, "i", "i", "s"]
    frame_duration = 0.02

    segments = _merge_ctc_frames(frame_symbols, frame_duration)

    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[-1].end_sec == pytest.approx(len(frame_symbols) * frame_duration)
    for prev, cur in zip(segments, segments[1:]):
        assert prev.end_sec == pytest.approx(cur.start_sec)


def test_merge_ctc_frames_confidence_is_none_for_all_segment_types():
    from vocal_analysis.recognizer import _merge_ctc_frames

    # §2.1: confidence は任意。本アダプタでは算出せず None を返す(母音・子音・gap いずれも)。
    segments = _merge_ctc_frames(["a", "n", None], 0.02)

    assert [s.confidence for s in segments] == [None, None, None]


def test_merge_ctc_frames_empty_input_returns_empty_list():
    from vocal_analysis.recognizer import _merge_ctc_frames

    assert _merge_ctc_frames([], 0.02) == []


def _seg(type_, start, end, phoneme=None):
    from vocal_analysis.types import Segment

    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=phoneme, confidence=None)


def test_absorb_short_segments_absorbs_into_longer_neighbor():
    from vocal_analysis.recognizer import _absorb_short_segments

    # 短区間(20ms)の左は閾値以上(70ms、それ自体は吸収対象でない)・右はより長い(100ms)
    # → より長い右へ吸収する(§5.1)。左を閾値未満にすると左自体も吸収対象になり
    # 比較の意図(どちらへ吸収するか)が曖昧になるため、左は閾値以上で固定する。
    segments = [
        _seg("vowel", 0.00, 0.07, "a"),
        _seg("consonant", 0.07, 0.09, "n"),
        _seg("vowel", 0.09, 0.19, "i"),
    ]

    result = _absorb_short_segments(segments, min_duration_sec=0.06)

    assert len(result) == 2
    assert result[0] == _seg("vowel", 0.00, 0.07, "a")
    assert result[1].type == "vowel"
    assert result[1].phoneme == "i"
    assert result[1].start_sec == pytest.approx(0.07)
    assert result[1].end_sec == pytest.approx(0.19)


def test_absorb_short_segments_tie_absorbs_into_prior():
    from vocal_analysis.recognizer import _absorb_short_segments

    # 左右の隣接区間が同長(80ms)のときは直前へ吸収する(§5.1)。
    segments = [
        _seg("vowel", 0.00, 0.08, "a"),
        _seg("consonant", 0.08, 0.10, "n"),
        _seg("vowel", 0.10, 0.18, "i"),
    ]

    result = _absorb_short_segments(segments, min_duration_sec=0.06)

    assert len(result) == 2
    assert result[0].phoneme == "a"
    assert result[0].end_sec == pytest.approx(0.10)
    assert result[1] == _seg("vowel", 0.10, 0.18, "i")


def test_absorb_short_segments_boundary_only_one_side_available():
    from vocal_analysis.recognizer import _absorb_short_segments

    # 先頭区間が短く片側(右)しか隣接が無ければその側へ吸収する(§5.1)。
    segments = [
        _seg("consonant", 0.00, 0.02, "n"),
        _seg("vowel", 0.02, 0.12, "a"),
    ]

    result = _absorb_short_segments(segments, min_duration_sec=0.06)

    assert len(result) == 1
    assert result[0].type == "vowel"
    assert result[0].start_sec == pytest.approx(0.00)
    assert result[0].end_sec == pytest.approx(0.12)


def test_absorb_short_segments_never_absorbs_into_gap():
    from vocal_analysis.recognizer import _absorb_short_segments

    # 短区間の左がgap(120ms)・右が母音(80ms)。gapは吸収先にしないため、
    # 左がより長くても右(母音)へ吸収する(§5.1: gapは吸収先にしない)。
    segments = [
        _seg("gap", 0.00, 0.12, None),
        _seg("consonant", 0.12, 0.14, "n"),
        _seg("vowel", 0.14, 0.22, "a"),
    ]

    result = _absorb_short_segments(segments, min_duration_sec=0.06)

    assert len(result) == 2
    assert result[0] == _seg("gap", 0.00, 0.12, None)
    assert result[1].type == "vowel"
    assert result[1].start_sec == pytest.approx(0.12)
    assert result[1].end_sec == pytest.approx(0.22)


def test_absorb_short_segments_becomes_gap_when_no_non_gap_neighbor():
    from vocal_analysis.recognizer import _absorb_short_segments

    # 短区間の両隣がgapのみ(非gapの隣接が無い)場合、その短区間自体をgapにする(§5.1)。
    segments = [
        _seg("gap", 0.00, 0.10, None),
        _seg("consonant", 0.10, 0.12, "n"),
        _seg("gap", 0.12, 0.22, None),
    ]

    result = _absorb_short_segments(segments, min_duration_sec=0.06)

    assert len(result) == 1
    assert result[0].type == "gap"
    assert result[0].start_sec == pytest.approx(0.00)
    assert result[0].end_sec == pytest.approx(0.22)


def test_absorb_short_segments_short_gap_absorbs_into_non_gap_neighbor():
    from vocal_analysis.recognizer import _absorb_short_segments

    # §5.1 の60ms吸収は「最小区間長を満たさない区間」全般が対象で、短い gap 自身も例外ではない。
    # gap が「吸収先にしない」のは吸収"先"としての禁止であり、短い gap 自身が吸収"される側"に
    # なることは妨げない。短い gap(20ms)の両隣は母音(左100ms・右150ms)で、より長い右へ吸収する。
    segments = [
        _seg("vowel", 0.00, 0.10, "a"),
        _seg("gap", 0.10, 0.12, None),
        _seg("vowel", 0.12, 0.27, "i"),
    ]

    result = _absorb_short_segments(segments, min_duration_sec=0.06)

    assert len(result) == 2
    assert result[0] == _seg("vowel", 0.00, 0.10, "a")
    assert result[1].type == "vowel"
    assert result[1].phoneme == "i"
    assert result[1].start_sec == pytest.approx(0.10)
    assert result[1].end_sec == pytest.approx(0.27)


def test_absorb_short_segments_leaves_long_segments_unchanged():
    from vocal_analysis.recognizer import _absorb_short_segments

    segments = [_seg("vowel", 0.00, 0.10, "a"), _seg("consonant", 0.10, 0.20, "n")]

    result = _absorb_short_segments(segments, min_duration_sec=0.06)

    assert result == segments


def test_absorb_short_segments_single_segment_with_no_neighbor_terminates():
    from vocal_analysis.recognizer import _absorb_short_segments

    # 非gapの隣接を一切持たない単独の短区間(吸収先が無い)。§5.1どおり自身をgapにして
    # 終了する(隣接が無いためこれ以上連結しようがなく、無限に再判定してはならない)。
    segments = [_seg("consonant", 0.00, 0.02, "n")]

    result = _absorb_short_segments(segments, min_duration_sec=0.06)

    assert result == [_seg("gap", 0.00, 0.02, None)]


def test_absorb_short_segments_all_short_gap_run_terminates():
    from vocal_analysis.recognizer import _absorb_short_segments

    # 短区間が連続し非gapの隣接が一切無い(全体が短いblank連続)。すべてgapになり1区間へ
    # 連結されて終了する(無限ループしない)。
    segments = [
        _seg("gap", 0.00, 0.02, None),
        _seg("gap", 0.02, 0.04, None),
        _seg("gap", 0.04, 0.06, None),
    ]

    result = _absorb_short_segments(segments, min_duration_sec=0.06)

    assert result == [_seg("gap", 0.00, 0.06, None)]


# --- §5.2 手順3: 音素列の結合(pau挿入) ---

_XFAIL_COMPOSITE = pytest.mark.xfail(reason="impl pending: composite recognizer helpers", strict=True)


@_XFAIL_COMPOSITE
def test_assemble_phoneme_sequence_wraps_single_chunk_with_pau():
    from vocal_analysis.recognizer import _assemble_phoneme_sequence

    result = _assemble_phoneme_sequence([["a", "i"]])

    assert result == ["pau", "a", "i", "pau"]


@_XFAIL_COMPOSITE
def test_assemble_phoneme_sequence_inserts_pau_between_chunks():
    from vocal_analysis.recognizer import _assemble_phoneme_sequence

    result = _assemble_phoneme_sequence([["a"], ["k", "i"]])

    assert result == ["pau", "a", "pau", "k", "i", "pau"]


@_XFAIL_COMPOSITE
def test_assemble_phoneme_sequence_empty_chunk_still_gets_boundary_pau():
    from vocal_analysis.recognizer import _assemble_phoneme_sequence

    # Whisperのチャンクがテキストを持たずG2P結果が空でも、チャンク境界のpau挿入は行う。
    result = _assemble_phoneme_sequence([[], ["a"]])

    assert result == ["pau", "pau", "a", "pau"]


@_XFAIL_COMPOSITE
def test_assemble_phoneme_sequence_no_chunks_is_leading_and_trailing_pau_only():
    from vocal_analysis.recognizer import _assemble_phoneme_sequence

    result = _assemble_phoneme_sequence([])

    assert result == ["pau", "pau"]


# --- §5.2 手順4: G2P記号→音素モデル語彙のトークンID変換 ---


@_XFAIL_COMPOSITE
def test_g2p_symbols_to_token_ids_maps_known_symbols():
    from vocal_analysis.recognizer import _g2p_symbols_to_token_ids

    vocab = {"<pad>": 0, "a": 5, "k": 7}

    result = _g2p_symbols_to_token_ids(["a", "k"], vocab, blank_token_id=0)

    assert result == [5, 7]


@_XFAIL_COMPOSITE
def test_g2p_symbols_to_token_ids_maps_pau_and_cl_to_blank():
    from vocal_analysis.recognizer import _g2p_symbols_to_token_ids

    # §5.2: pau・cl は語彙記号への対応付けを持たず、blank トークンへ変換する。
    vocab = {"<pad>": 0, "a": 5}

    result = _g2p_symbols_to_token_ids(["pau", "cl"], vocab, blank_token_id=0)

    assert result == [0, 0]


@_XFAIL_COMPOSITE
def test_g2p_symbols_to_token_ids_devoiced_vowels_collapse_to_voiced():
    from vocal_analysis.recognizer import _g2p_symbols_to_token_ids

    # §5.2 写像表: 無声化母音 I/U は有声母音と同じ語彙記号(i/ɯ)へ収束する。
    vocab = {"<pad>": 0, "i": 3, "ɯ": 4}

    result = _g2p_symbols_to_token_ids(["I", "U"], vocab, blank_token_id=0)

    assert result == [3, 4]


@_XFAIL_COMPOSITE
def test_g2p_symbols_to_token_ids_unmapped_symbol_raises_recognition_error():
    from vocal_analysis.recognizer import RecognitionError, _g2p_symbols_to_token_ids

    # §5.2 手順4: 写像表に無い記号が現れたら RecognitionError で停止する(黙って捨てない)。
    with pytest.raises(RecognitionError):
        _g2p_symbols_to_token_ids(["xx"], {"<pad>": 0}, blank_token_id=0)


@_XFAIL_COMPOSITE
def test_g2p_symbols_to_token_ids_missing_vocab_entry_raises_recognition_error():
    from vocal_analysis.recognizer import RecognitionError, _g2p_symbols_to_token_ids

    # 写像表上は既知の記号でも、実際のモデル語彙にその記号が無ければ RecognitionError。
    with pytest.raises(RecognitionError):
        _g2p_symbols_to_token_ids(["a"], {"<pad>": 0}, blank_token_id=0)


# --- §5.2 手順5: 強制アライメント(Viterbi) ---


@_XFAIL_COMPOSITE
def test_forced_align_single_state_stays_for_all_frames():
    from vocal_analysis.recognizer import _forced_align

    log_probs = np.array([[0.1], [-1.0], [0.2]])

    path = _forced_align(log_probs, token_ids=[0], blank_token_id=0)

    assert path == [0, 0, 0]


@_XFAIL_COMPOSITE
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

    path = _forced_align(log_probs, token_ids=[0, 1], blank_token_id=0)

    assert path == [0, 0, 1, 1]


@_XFAIL_COMPOSITE
def test_forced_align_raises_when_fewer_frames_than_tokens():
    from vocal_analysis.recognizer import RecognitionError, _forced_align

    # §5.2 手順5: フレーム数がトークン数未満だと末尾トークンへ理論上到達不能。RecognitionErrorで停止する。
    log_probs = np.zeros((2, 1))

    with pytest.raises(RecognitionError):
        _forced_align(log_probs, token_ids=[0, 0, 0], blank_token_id=0)


# --- §5.2 手順6・7: 区間の確定とSegment化 ---


@_XFAIL_COMPOSITE
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


@_XFAIL_COMPOSITE
def test_path_to_segments_merges_adjacent_states_with_same_output_symbol():
    from vocal_analysis.recognizer import _path_to_segments

    # 連続する2状態がともに母音"a"(長母音が2モーラに分かれた場合等)は1区間へ結合する(§5.2手順6)。
    segments = _path_to_segments([0, 0, 1, 1], ["a", "a"], frame_duration_sec=0.02)

    assert len(segments) == 1
    assert segments[0].type == "vowel"
    assert segments[0].phoneme == "a"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.08)


@_XFAIL_COMPOSITE
def test_path_to_segments_merges_pau_and_cl_as_same_gap():
    from vocal_analysis.recognizer import _path_to_segments

    # pau由来とcl由来はいずれもblank扱いで同一視し、1つのgap区間へ結合する(§5.2手順6)。
    segments = _path_to_segments([0, 0, 1, 1], ["pau", "cl"], frame_duration_sec=0.02)

    assert len(segments) == 1
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.08)


@_XFAIL_COMPOSITE
def test_path_to_segments_last_token_extends_to_audio_end():
    from vocal_analysis.recognizer import _path_to_segments

    segments = _path_to_segments([0, 1, 1, 1, 1], ["pau", "a"], frame_duration_sec=0.02)

    assert segments[-1].end_sec == pytest.approx(0.10)
