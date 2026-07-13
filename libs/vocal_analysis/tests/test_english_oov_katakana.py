"""英語未知語カタカナ化フォールバックのテスト。

pyopenjtalk-plusが正しく読めない英単語(1形態素ノードで完結し、品詞がフィラーと判定される
未知語)を検出し、CMUdictで発音記号を引いて変換モデルへ渡し、カタカナへ補完変換する機能を
検証する。対象判定条件(_is_target_node)・CMUdict参照(_lookup_cmudict_phonemes)・
出力妥当性検証(_is_valid_katakana)・全角/半角変換は実際のpyopenjtalk-plus・nltk cmudictを
使い決定論的に検証する(いずれもローカル・高速でGPU/ネットワークを要さない)。モデル呼び出しを
伴う変換全体は _generate_katakana をモックして検証する(実モデル・ネットワークを必須にしない)。
"""

import pytest

pytestmark = pytest.mark.xfail(
    reason="impl pending: vocal_analysis.english_oov_katakana module not yet implemented",
    strict=True,
)


@pytest.mark.parametrize(
    "surface,pos,expected",
    [
        ("sky", "フィラー", True),
        ("sky", "名詞", False),  # 既に正しく変換済みの語は変換不要
        ("sky", "記号", False),  # 複数ノードへ分裂した断片は対象外
        ("sky3", "フィラー", False),  # ASCII英字以外(数字)を含む
        ("sky!", "フィラー", False),  # ASCII英字以外(記号)を含む
        ("空", "フィラー", False),  # ASCII文字列でない
        ("", "フィラー", False),  # 空文字列
    ],
)
def test_is_target_node(surface, pos, expected):
    from vocal_analysis.english_oov_katakana import _is_target_node

    assert _is_target_node(surface, pos) is expected


def test_find_target_words_detects_filler_pos_ascii_word():
    from vocal_analysis.english_oov_katakana import _find_target_words

    # 実機確認: 「空を見上げてsky」の"sky"は1形態素ノード・品詞フィラーとして検出される。
    assert _find_target_words("空を見上げてsky") == ["sky"]


def test_find_target_words_excludes_already_recognized_noun():
    from vocal_analysis.english_oov_katakana import _find_target_words

    # 実機確認: 「love」は品詞が名詞(既に正しく変換済み)のため対象にしない。
    assert _find_target_words("空を見上げてlove") == []


def test_find_target_words_excludes_non_target_when_no_ascii_word():
    from vocal_analysis.english_oov_katakana import _find_target_words

    assert _find_target_words("空を見上げて雲をながめて") == []


def test_find_target_words_excludes_symbol_fragments_of_split_word():
    from vocal_analysis.english_oov_katakana import _find_target_words

    # 実機確認: "dog's"は"do"(名詞)・"g"(記号)・"'"(記号)・"s"(記号)の4ノードへ分裂し、
    # いずれも対象条件(品詞フィラー)を満たさないため対象語は1つも検出されない。
    assert _find_target_words("空を見上げてdog's") == []


def test_find_target_words_returns_each_occurrence_in_order():
    from vocal_analysis.english_oov_katakana import _find_target_words

    # 同じ対象語が複数回出現する場合、出現順にそのまま列挙する(重複除去しない)。
    assert _find_target_words("skyとsky") == ["sky", "sky"]


def test_lookup_cmudict_phonemes_known_word():
    from vocal_analysis.english_oov_katakana import _lookup_cmudict_phonemes

    # CMUdictは小文字キーで発音記号(ARPAbet)を持つ。
    assert _lookup_cmudict_phonemes("sky") == "S K AY1"


def test_lookup_cmudict_phonemes_is_case_insensitive():
    from vocal_analysis.english_oov_katakana import _lookup_cmudict_phonemes

    assert _lookup_cmudict_phonemes("Sky") == "S K AY1"


def test_lookup_cmudict_phonemes_unknown_word_returns_none():
    from vocal_analysis.english_oov_katakana import _lookup_cmudict_phonemes

    # 実機確認: CMUdict原典(1993〜2008年収録)より新しい固有名詞は未収録。
    assert _lookup_cmudict_phonemes("tiktok") is None


@pytest.mark.parametrize(
    "text,expected",
    [
        ("スカイ", True),
        ("すかい", True),
        ("スカイー", True),
        ("sky", False),
        ("スカイER", False),
        ("", False),
    ],
)
def test_is_valid_katakana(text, expected):
    from vocal_analysis.english_oov_katakana import _is_valid_katakana

    # ひらがな・カタカナ(長音符含む)のみで構成されるかを判定する。
    assert _is_valid_katakana(text) is expected


def test_to_halfwidth_converts_fullwidth_latin():
    from vocal_analysis.english_oov_katakana import _to_halfwidth

    # 実機確認: pyopenjtalk-plusが返す表層形は全角(Unicode fullwidth)文字列になる。
    assert _to_halfwidth("ｓｋｙ") == "sky"


def test_to_halfwidth_leaves_non_fullwidth_characters_unchanged():
    from vocal_analysis.english_oov_katakana import _to_halfwidth

    assert _to_halfwidth("空をｓｋｙ見上げて") == "空をsky見上げて"


def test_locate_and_replace_replaces_single_target():
    from vocal_analysis.english_oov_katakana import _locate_and_replace

    result = _locate_and_replace("空を見上げてsky", ["sky"], {"sky": "スカイ"})

    assert result == "空を見上げてスカイ"


def test_locate_and_replace_leaves_unconverted_word_untouched():
    from vocal_analysis.english_oov_katakana import _locate_and_replace

    # convertedに値が無い語(CMUdict未収録・生成失敗)は元のテキストのまま残す(安全側フォールバック)。
    result = _locate_and_replace("空を見上げてTikTok", ["TikTok"], {})

    assert result == "空を見上げてTikTok"


def test_locate_and_replace_handles_multiple_occurrences_in_order():
    from vocal_analysis.english_oov_katakana import _locate_and_replace

    # 対象語が2回出現する場合、両方とも正しい位置で置換する。
    result = _locate_and_replace("skyとsky", ["sky", "sky"], {"sky": "スカイ"})

    assert result == "スカイとスカイ"


def test_locate_and_replace_mixed_converted_and_unconverted():
    from vocal_analysis.english_oov_katakana import _locate_and_replace

    # 変換に成功した語と失敗した語が混在しても、成功分だけ正しい位置で置換する。
    result = _locate_and_replace(
        "TikTokを見てskyを見上げる", ["TikTok", "sky"], {"sky": "スカイ"}
    )

    assert result == "TikTokを見てスカイを見上げる"


def test_convert_oov_words_no_target_returns_text_unchanged():
    from vocal_analysis.english_oov_katakana import convert_oov_words

    assert convert_oov_words("空を見上げて雲をながめて") == "空を見上げて雲をながめて"


def test_convert_oov_words_converts_target_via_model(monkeypatch):
    import vocal_analysis.english_oov_katakana as module

    calls = []
    monkeypatch.setattr(
        module, "_generate_katakana",
        lambda word, phonemes: calls.append((word, phonemes)) or "スカイ",
    )

    result = module.convert_oov_words("空を見上げてsky")

    assert result == "空を見上げてスカイ"
    assert calls == [("sky", "S K AY1")]


def test_convert_oov_words_leaves_text_unchanged_when_cmudict_misses(monkeypatch):
    import vocal_analysis.english_oov_katakana as module

    # CMUdict未収録語(実機確認: TikTok)は変換せず元のまま残る。モデルは一切呼ばれない。
    calls = []
    monkeypatch.setattr(module, "_generate_katakana", lambda word, phonemes: calls.append(word) or "スカイ")

    result = module.convert_oov_words("空を見上げてTikTok")

    assert result == "空を見上げてTikTok"
    assert calls == []


def test_convert_oov_words_leaves_text_unchanged_when_model_output_invalid(monkeypatch):
    import vocal_analysis.english_oov_katakana as module

    # 生成結果に非カタカナ文字が混入した場合(実機確認例: 別語でアルファベットが混入する破綻を
    # 観測済み)は変換せず元のまま残る安全側フォールバック。対象語はフィラー・CMUdict収録済みの
    # "sky"を使い、モデルが実際に呼ばれた上でこのフォールバックが働くことを確認する。
    calls = []
    monkeypatch.setattr(
        module, "_generate_katakana",
        lambda word, phonemes: calls.append(word) or "スカイER",
    )

    result = module.convert_oov_words("空を見上げてsky")

    assert result == "空を見上げてsky"
    assert calls == ["sky"]
