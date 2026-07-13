"""英語未知語カタカナ化フォールバックのテスト。

pyopenjtalk-plusが正しく読めない英単語(1形態素ノードで完結し、品詞がフィラーと判定される
未知語)を検出し、CMUdictで発音記号を引いて変換モデルへ渡し、カタカナへ補完変換する機能を
検証する。対象判定条件(_is_target_node)・CMUdict参照(_lookup_cmudict_phonemes)・
出力妥当性検証(_is_valid_katakana)・全角/半角変換は実際のpyopenjtalk-plus・nltk cmudictを
使い決定論的に検証する(いずれもローカル・高速でGPU/ネットワークを要さない)。モデル呼び出しを
伴う変換全体は _generate_katakana をモックして検証する(実モデル・ネットワークを必須にしない)。
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_conversion_cache():
    """convert_oov_wordsが語ごとの変換結果を保持するプロセス内キャッシュを、テスト間で
    汚染しないよう各テストの前後でクリアする。
    """
    import vocal_analysis.english_oov_katakana as module

    module._conversion_cache.clear()
    yield
    module._conversion_cache.clear()


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

    # convertedに値が無い語(呼び出し元でCMUdict未収録・生成失敗と判定された語を想定)は元のテキスト
    # のまま残す(安全側フォールバック)。このテストは_locate_and_replace単体の置換ロジックのみを
    # 検証するもので、対象語がCMUdictに実在するかどうかとは無関係(任意の語で成立する)。
    result = _locate_and_replace("空を見上げてwordx", ["wordx"], {})

    assert result == "空を見上げてwordx"


def test_locate_and_replace_handles_multiple_occurrences_in_order():
    from vocal_analysis.english_oov_katakana import _locate_and_replace

    # 対象語が2回出現する場合、両方とも正しい位置で置換する。
    result = _locate_and_replace("skyとsky", ["sky", "sky"], {"sky": "スカイ"})

    assert result == "スカイとスカイ"


def test_locate_and_replace_mixed_converted_and_unconverted():
    from vocal_analysis.english_oov_katakana import _locate_and_replace

    # 変換に成功した語と失敗した語が混在しても、成功分だけ正しい位置で置換する。convertedに
    # 値が無い語(wordx)がCMUdict未収録かどうかはこのテストの対象外(_locate_and_replace単体の
    # 置換ロジックのみを検証する)。
    result = _locate_and_replace(
        "wordxを見てskyを見上げる", ["wordx", "sky"], {"sky": "スカイ"}
    )

    assert result == "wordxを見てスカイを見上げる"


def test_locate_and_replace_matches_fullwidth_original_text():
    from vocal_analysis.english_oov_katakana import _locate_and_replace

    # 元テキスト自体が全角の場合(半角検索が失敗する場合)は全角表記でも検索する。
    result = _locate_and_replace("空を見上げてｓｋｙ", ["sky"], {"sky": "スカイ"})

    assert result == "空を見上げてスカイ"


def test_locate_and_replace_advances_cursor_past_unconverted_word():
    from vocal_analysis.english_oov_katakana import _locate_and_replace

    # 変換に失敗した語(位置は特定できるが置換しない)の直後に別の対象語が続く場合、変換失敗語の
    # 位置を読み飛ばさずカーソルを前進させることで、後続語の検索がその前の位置を誤って拾わない。
    result = _locate_and_replace("skyline sky", ["skyline", "sky"], {"sky": "スカイ"})

    assert result == "skyline スカイ"


def test_locate_and_replace_prefers_nearer_occurrence_when_widths_mixed():
    from vocal_analysis.english_oov_katakana import _locate_and_replace

    # 同じ対象語が全角表記・半角表記の順で混在する場合、半角検索を無条件に優先すると手前の
    # 全角の出現を飛び越して後方の半角の出現を誤って拾う。cursorに近い方を優先して両方を
    # 出現順どおりに正しく置換する。
    result = _locate_and_replace("ｓｋｙ sky", ["sky", "sky"], {"sky": "スカイ"})

    assert result == "スカイ スカイ"


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


def test_convert_oov_words_caches_conversion_result_across_calls(monkeypatch):
    import vocal_analysis.english_oov_katakana as module

    # CMUdict参照・モデル推論はいずれもコストが大きいため、同じ語をまたがる複数回の
    # convert_oov_words呼び出し(_g2pがrecognize()実行中に同じ内容へ繰り返し呼ばれる状況を想定)で
    # 再計算しない。2回目の呼び出しでは_lookup_cmudict_phonemes・_generate_katakanaのどちらも
    # 呼ばれないことを確認する。
    lookup_calls = []
    monkeypatch.setattr(
        module, "_lookup_cmudict_phonemes",
        lambda word: lookup_calls.append(word) or "S K AY1",
    )
    generate_calls = []
    monkeypatch.setattr(
        module, "_generate_katakana",
        lambda word, phonemes: generate_calls.append(word) or "スカイ",
    )

    first = module.convert_oov_words("空を見上げてsky")
    second = module.convert_oov_words("空を見上げてsky")

    assert first == "空を見上げてスカイ"
    assert second == "空を見上げてスカイ"
    assert lookup_calls == ["sky"]
    assert generate_calls == ["sky"]


def test_convert_oov_words_leaves_text_unchanged_when_cmudict_misses(monkeypatch):
    import vocal_analysis.english_oov_katakana as module

    # CMUdict参照が失敗した場合(戻り値None)は変換せず元のまま残り、モデルは一切呼ばれない。この
    # 分岐そのものを検証したいので、_lookup_cmudict_phonemesをモックして直接Noneを返させる
    # (特定の実在語が現時点でCMUdictに収録されているか否かという外部データの事実には依存しない。
    # その事実自体は_lookup_cmudict_phonemes単体のテストで別途検証済み)。
    calls = []
    monkeypatch.setattr(module, "_lookup_cmudict_phonemes", lambda word: None)
    monkeypatch.setattr(module, "_generate_katakana", lambda word, phonemes: calls.append(word) or "スカイ")

    result = module.convert_oov_words("空を見上げてsky")

    assert result == "空を見上げてsky"
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
