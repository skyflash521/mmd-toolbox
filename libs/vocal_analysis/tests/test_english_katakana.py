"""英語カタカナ化フォールバックのテスト。

pyopenjtalk-plusが正しく読めない英単語(1形態素ノードで完結し、品詞がフィラーと判定される
語)を検出し、CMUdictで発音記号を引いて変換方式(既定: arpakanaによるルールベース変換、
選択式: tinyllama-katakana-converterモデルによる生成)へ渡し、カタカナへ補完変換する機能を
検証する。対象判定条件(_is_target_node)・CMUdict参照(_lookup_cmudict_phonemes)・
出力妥当性検証(_is_valid_katakana)・全角/半角変換・arpakanaによる変換は実際のpyopenjtalk-plus・
nltk cmudict・arpakanaを使い決定論的に検証する(いずれもローカル・高速でGPU/ネットワークを
要さない)。tinyllama-katakana-converterモデル呼び出しを伴う変換は_generate_katakana_tinyllama
をモックして検証する(実モデル・ネットワークを必須にしない)。
"""

import pytest


@pytest.fixture(autouse=True)
def _reset_conversion_cache():
    """convert_target_wordsが語ごとの変換結果を保持するプロセス内キャッシュを、テスト間で
    汚染しないよう各テストの前後でクリアする。
    """
    import vocal_analysis.english_katakana as module

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
    from vocal_analysis.english_katakana import _is_target_node

    assert _is_target_node(surface, pos) is expected


def test_find_target_words_detects_filler_pos_ascii_word():
    from vocal_analysis.english_katakana import _find_target_words

    # 実機確認: 「空を見上げてsky」の"sky"は1形態素ノード・品詞フィラーとして検出される。
    assert _find_target_words("空を見上げてsky") == ["sky"]


def test_find_target_words_excludes_already_recognized_noun():
    from vocal_analysis.english_katakana import _find_target_words

    # 実機確認: 「love」は品詞が名詞(既に正しく変換済み)のため対象にしない。
    assert _find_target_words("空を見上げてlove") == []


def test_find_target_words_excludes_non_target_when_no_ascii_word():
    from vocal_analysis.english_katakana import _find_target_words

    assert _find_target_words("空を見上げて雲をながめて") == []


def test_find_target_words_excludes_symbol_fragments_of_split_word():
    from vocal_analysis.english_katakana import _find_target_words

    # 実機確認: "dog's"は"do"(名詞)・"g"(記号)・"'"(記号)・"s"(記号)の4ノードへ分裂し、
    # いずれも対象条件(品詞フィラー)を満たさないため対象語は1つも検出されない。
    assert _find_target_words("空を見上げてdog's") == []


def test_find_target_words_returns_each_occurrence_in_order():
    from vocal_analysis.english_katakana import _find_target_words

    # 同じ対象語が複数回出現する場合、出現順にそのまま列挙する(重複除去しない)。
    assert _find_target_words("skyとsky") == ["sky", "sky"]


def test_lookup_cmudict_phonemes_known_word():
    from vocal_analysis.english_katakana import _lookup_cmudict_phonemes

    # CMUdictは小文字キーで発音記号(ARPAbet)を持つ。
    assert _lookup_cmudict_phonemes("sky") == "S K AY1"


def test_lookup_cmudict_phonemes_is_case_insensitive():
    from vocal_analysis.english_katakana import _lookup_cmudict_phonemes

    assert _lookup_cmudict_phonemes("Sky") == "S K AY1"


def test_lookup_cmudict_phonemes_unknown_word_returns_none():
    from vocal_analysis.english_katakana import _lookup_cmudict_phonemes

    # 実機確認: CMUdict原典(1993〜2008年収録)より新しい固有名詞は未収録。
    assert _lookup_cmudict_phonemes("tiktok") is None


def test_get_cmudict_entries_auto_downloads_when_not_cached(monkeypatch):
    import nltk.corpus

    import vocal_analysis.english_katakana as module

    monkeypatch.setattr(module, "_cmudict_cache", None)
    download_calls = []
    attempts = {"n": 0}

    class _FakeCmudict:
        # 実物のnltk.corpus.cmudict(LazyCorpusLoader)は初回ロードで自身をコーパスへ
        # 変身させる特殊な実装のため、そのオブジェクト自体をこの偽物へ丸ごと差し替えて
        # 検証する(dict属性だけをモックすると、他テストの実行順で実物が既にロード済みかに
        # 挙動が左右され、環境・順序依存になる)。
        def dict(self):
            attempts["n"] += 1
            if attempts["n"] == 1:
                raise LookupError("missing")
            return {"sky": [["S", "K", "AY1"]]}

    monkeypatch.setattr(nltk.corpus, "cmudict", _FakeCmudict())
    monkeypatch.setattr(nltk, "download", lambda name, quiet=False: download_calls.append((name, quiet)))

    entries = module._get_cmudict_entries()

    assert download_calls == [("cmudict", True)]
    assert entries == {"sky": [["S", "K", "AY1"]]}


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
    from vocal_analysis.english_katakana import _is_valid_katakana

    # ひらがな・カタカナ(長音符含む)のみで構成されるかを判定する。
    assert _is_valid_katakana(text) is expected


def test_to_halfwidth_converts_fullwidth_latin():
    from vocal_analysis.english_katakana import _to_halfwidth

    # 実機確認: pyopenjtalk-plusが返す表層形は全角(Unicode fullwidth)文字列になる。
    assert _to_halfwidth("ｓｋｙ") == "sky"


def test_to_halfwidth_leaves_non_fullwidth_characters_unchanged():
    from vocal_analysis.english_katakana import _to_halfwidth

    assert _to_halfwidth("空をｓｋｙ見上げて") == "空をsky見上げて"


def test_locate_and_replace_replaces_single_target():
    from vocal_analysis.english_katakana import _locate_and_replace

    result = _locate_and_replace("空を見上げてsky", ["sky"], {"sky": "スカイ"})

    assert result == "空を見上げてスカイ"


def test_locate_and_replace_leaves_unconverted_word_untouched():
    from vocal_analysis.english_katakana import _locate_and_replace

    # convertedに値が無い語(呼び出し元でCMUdict未収録・生成失敗と判定された語を想定)は元のテキスト
    # のまま残す(安全側フォールバック)。このテストは_locate_and_replace単体の置換ロジックのみを
    # 検証するもので、対象語がCMUdictに実在するかどうかとは無関係(任意の語で成立する)。
    result = _locate_and_replace("空を見上げてwordx", ["wordx"], {})

    assert result == "空を見上げてwordx"


def test_locate_and_replace_handles_multiple_occurrences_in_order():
    from vocal_analysis.english_katakana import _locate_and_replace

    # 対象語が2回出現する場合、両方とも正しい位置で置換する。
    result = _locate_and_replace("skyとsky", ["sky", "sky"], {"sky": "スカイ"})

    assert result == "スカイとスカイ"


def test_locate_and_replace_mixed_converted_and_unconverted():
    from vocal_analysis.english_katakana import _locate_and_replace

    # 変換に成功した語と失敗した語が混在しても、成功分だけ正しい位置で置換する。convertedに
    # 値が無い語(wordx)がCMUdict未収録かどうかはこのテストの対象外(_locate_and_replace単体の
    # 置換ロジックのみを検証する)。
    result = _locate_and_replace(
        "wordxを見てskyを見上げる", ["wordx", "sky"], {"sky": "スカイ"}
    )

    assert result == "wordxを見てスカイを見上げる"


def test_locate_and_replace_matches_fullwidth_original_text():
    from vocal_analysis.english_katakana import _locate_and_replace

    # 元テキスト自体が全角の場合(半角検索が失敗する場合)は全角表記でも検索する。
    result = _locate_and_replace("空を見上げてｓｋｙ", ["sky"], {"sky": "スカイ"})

    assert result == "空を見上げてスカイ"


def test_locate_and_replace_advances_cursor_past_unconverted_word():
    from vocal_analysis.english_katakana import _locate_and_replace

    # 変換に失敗した語(位置は特定できるが置換しない)の直後に別の対象語が続く場合、変換失敗語の
    # 位置を読み飛ばさずカーソルを前進させることで、後続語の検索がその前の位置を誤って拾わない。
    result = _locate_and_replace("skyline sky", ["skyline", "sky"], {"sky": "スカイ"})

    assert result == "skyline スカイ"


def test_locate_and_replace_prefers_nearer_occurrence_when_widths_mixed():
    from vocal_analysis.english_katakana import _locate_and_replace

    # 同じ対象語が全角表記・半角表記の順で混在する場合、半角検索を無条件に優先すると手前の
    # 全角の出現を飛び越して後方の半角の出現を誤って拾う。cursorに近い方を優先して両方を
    # 出現順どおりに正しく置換する。
    result = _locate_and_replace("ｓｋｙ sky", ["sky", "sky"], {"sky": "スカイ"})

    assert result == "スカイ スカイ"


def test_generate_katakana_arpakana_converts_phonemes_to_kana():
    from vocal_analysis.english_katakana import _generate_katakana_arpakana

    # arpakanaはARPAbet音素をルールベースでカタカナへ変換する(生成モデル・GPU不要)。
    assert _generate_katakana_arpakana("sky", "S K AY1") == "スカイ"


def test_generate_katakana_arpakana_returns_invalid_output_when_conversion_raises(monkeypatch):
    # arpabet_to_kana自体が例外を送出しても、CMUdict未収録・出力不正と同じ安全側フォールバック
    # (呼び出し元が変換不可と判定できる値)にする。ライブラリ未導入時のインポート例外はこの経路の
    # 対象外(呼び出し元まで伝播させる。設計上の意図は_generate_katakana_arpakanaのdocstring参照)。
    import arpakana

    import vocal_analysis.english_katakana as module

    def _raise(phonemes):
        raise ValueError("boom")

    monkeypatch.setattr(arpakana, "arpabet_to_kana", _raise)

    result = module._generate_katakana_arpakana("sky", "S K AY1")

    assert module._is_valid_katakana(result) is False


def test_generate_katakana_tinyllama_returns_invalid_output_when_generation_raises(monkeypatch):
    import vocal_analysis.english_katakana as module

    # モデルロード後の生成処理自体が例外を送出しても、arpakanaと同じ安全側フォールバック
    # (呼び出し元が変換不可と判定できる値)にする。_load_katakana_model自体が送出する例外
    # (未導入・モデル取得失敗)はこの経路の対象外(呼び出し元まで伝播させる)。

    class _RaisingTokenizer:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        module, "_load_katakana_model", lambda on_progress=None: (_RaisingTokenizer(), object()))

    result = module._generate_katakana_tinyllama("sky", "S K AY1")

    assert module._is_valid_katakana(result) is False


def test_generate_katakana_dispatches_to_arpakana_by_default():
    from vocal_analysis.english_katakana import _generate_katakana

    # methodを省略した既定の呼び出しが、method="arpakana"を明示した呼び出しと同じ結果になる
    # ことを確認する(=既定値がarpakanaであることの検証。単に出力値が既定のTinyLlama実装と
    # 偶然一致するかどうかでは検証にならないため、method引数の受理そのものを確認する)。
    assert _generate_katakana("sky", "S K AY1") == _generate_katakana("sky", "S K AY1", method="arpakana")


def test_generate_katakana_dispatches_to_tinyllama_when_selected(monkeypatch):
    import vocal_analysis.english_katakana as module

    calls = []
    monkeypatch.setattr(
        module, "_generate_katakana_tinyllama",
        lambda word, phonemes, **kwargs: calls.append((word, phonemes)) or "スカイ",
    )

    result = module._generate_katakana("sky", "S K AY1", method="tinyllama-katakana-converter")

    assert result == "スカイ"
    assert calls == [("sky", "S K AY1")]


def test_generate_katakana_raises_for_unknown_method():
    from vocal_analysis.english_katakana import _generate_katakana

    # 未知のmethod値は、無言でtinyllama-katakana-converter(GPU・生成モデル)側へ流さず
    # ValueErrorにする(誤字・将来の値追加が意図せず重いモデルロードにつながることを防ぐ)。
    with pytest.raises(ValueError):
        _generate_katakana("sky", "S K AY1", method="unknown-method")


def test_convert_target_words_no_target_returns_text_unchanged():
    from vocal_analysis.english_katakana import convert_target_words

    assert convert_target_words("空を見上げて雲をながめて") == "空を見上げて雲をながめて"


def test_convert_target_words_uses_arpakana_by_default():
    from vocal_analysis.english_katakana import convert_target_words

    # methodを指定しない既定の呼び出しは、実際のarpakana(生成モデル不要)で変換する。
    assert convert_target_words("空を見上げてsky") == "空を見上げてスカイ"


def test_convert_target_words_converts_target_via_tinyllama_model(monkeypatch):
    import vocal_analysis.english_katakana as module

    calls = []
    monkeypatch.setattr(
        module, "_generate_katakana_tinyllama",
        lambda word, phonemes, **kwargs: calls.append((word, phonemes)) or "スカイ",
    )

    result = module.convert_target_words("空を見上げてsky", method="tinyllama-katakana-converter")

    assert result == "空を見上げてスカイ"
    assert calls == [("sky", "S K AY1")]


def test_convert_target_words_caches_conversion_result_across_calls(monkeypatch):
    import vocal_analysis.english_katakana as module

    # CMUdict参照・モデル推論はいずれもコストが大きいため、同じ語をまたがる複数回の
    # convert_target_words呼び出し(_g2pがrecognize()実行中に同じ内容へ繰り返し呼ばれる状況を想定)で
    # 再計算しない。2回目の呼び出しでは_lookup_cmudict_phonemes・_generate_katakana_tinyllamaの
    # どちらも呼ばれないことを確認する。
    lookup_calls = []
    monkeypatch.setattr(
        module, "_lookup_cmudict_phonemes",
        lambda word: lookup_calls.append(word) or "S K AY1",
    )
    generate_calls = []
    monkeypatch.setattr(
        module, "_generate_katakana_tinyllama",
        lambda word, phonemes, **kwargs: generate_calls.append(word) or "スカイ",
    )

    first = module.convert_target_words("空を見上げてsky", method="tinyllama-katakana-converter")
    second = module.convert_target_words("空を見上げてsky", method="tinyllama-katakana-converter")

    assert first == "空を見上げてスカイ"
    assert second == "空を見上げてスカイ"
    assert lookup_calls == ["sky"]
    assert generate_calls == ["sky"]


def test_convert_target_words_leaves_text_unchanged_when_cmudict_misses(monkeypatch):
    import vocal_analysis.english_katakana as module

    # CMUdict参照が失敗した場合(戻り値None)は変換せず元のまま残り、変換方式(arpakana・
    # tinyllama-katakana-converterのいずれも)は一切呼ばれない。この分岐そのものを検証したいので、
    # _lookup_cmudict_phonemesをモックして直接Noneを返させる(特定の実在語が現時点でCMUdictに
    # 収録されているか否かという外部データの事実には依存しない。その事実自体は
    # _lookup_cmudict_phonemes単体のテストで別途検証済み)。
    calls = []
    monkeypatch.setattr(module, "_lookup_cmudict_phonemes", lambda word: None)
    monkeypatch.setattr(
        module, "_generate_katakana_arpakana",
        lambda word, phonemes: calls.append(word) or "スカイ",
    )

    result = module.convert_target_words("空を見上げてsky")

    assert result == "空を見上げてsky"
    assert calls == []


def test_convert_target_words_leaves_text_unchanged_when_output_invalid(monkeypatch):
    import vocal_analysis.english_katakana as module

    # 生成結果がひらがな・カタカナ以外の文字(非かな文字)を含む場合は採用せず、変換せず元の
    # まま残る安全側フォールバックとする。対象語はフィラー・CMUdict収録済みの"sky"を使い、
    # 変換関数が実際に呼ばれた上でこのフォールバックが働くことを確認する。
    calls = []
    monkeypatch.setattr(
        module, "_generate_katakana_arpakana",
        lambda word, phonemes: calls.append(word) or "スカイER",
    )

    result = module.convert_target_words("空を見上げてsky")

    assert result == "空を見上げてsky"
    assert calls == ["sky"]


def test_convert_target_words_caches_separately_per_method(monkeypatch):
    import vocal_analysis.english_katakana as module

    # 同じ語でも変換方式が異なれば別の結果になりうるため、キャッシュは方式ごとに分離する
    # (arpakanaでの変換結果がtinyllama-katakana-converter指定時の呼び出しを妨げてはならない)。
    tinyllama_calls = []
    monkeypatch.setattr(
        module, "_generate_katakana_tinyllama",
        lambda word, phonemes, **kwargs: tinyllama_calls.append(word) or "スカイー",
    )

    arpakana_result = module.convert_target_words("空を見上げてsky")
    tinyllama_result = module.convert_target_words("空を見上げてsky", method="tinyllama-katakana-converter")

    assert arpakana_result == "空を見上げてスカイ"
    assert tinyllama_result == "空を見上げてスカイー"
    assert tinyllama_calls == ["sky"]


def test_uncached_target_words_filters_already_cached_words(monkeypatch):
    import vocal_analysis.english_katakana as module

    # skyはキャッシュ済み・glimmerは未変換 → 未変換のglimmerだけが返る。
    module._conversion_cache[("tinyllama-katakana-converter", "sky")] = "スカイ"

    words = module.uncached_target_words(
        "空を見上げてsky そしてglimmer", method="tinyllama-katakana-converter"
    )

    assert words == ["glimmer"]


def test_uncached_target_words_returns_empty_when_no_targets():
    import vocal_analysis.english_katakana as module

    assert module.uncached_target_words("空を見上げて", method="tinyllama-katakana-converter") == []


def test_convert_words_caches_results_and_releases_tinyllama_model(monkeypatch):
    import vocal_analysis.english_katakana as module

    monkeypatch.setattr(
        module, "_generate_katakana_tinyllama", lambda word, phonemes, **kwargs: "スカイ"
    )
    release_calls = []
    monkeypatch.setattr(
        module, "release_katakana_model", lambda: release_calls.append("release")
    )

    module.convert_words(["sky"], method="tinyllama-katakana-converter")

    assert module._conversion_cache[("tinyllama-katakana-converter", "sky")] == "スカイ"
    assert release_calls == ["release"]


def test_convert_words_forwards_on_progress_to_load_katakana_model(monkeypatch):
    # convert_words -> _convert_word -> _generate_katakana -> _generate_katakana_tinyllama ->
    # _load_katakana_model という配線チェーンのどこかで on_progress を落とす退行を検出するため、
    # _load_katakana_model が実際に受け取った on_progress の同一性を検証する
    # (**kwargs で黙って吸収するモックでは検出できない)。
    import vocal_analysis.english_katakana as module

    module._conversion_cache.clear()
    captured = {}

    def fake_load_katakana_model(on_progress=None):
        captured["on_progress"] = on_progress
        return (object(), object())

    monkeypatch.setattr(module, "_load_katakana_model", fake_load_katakana_model)
    monkeypatch.setattr(module, "release_katakana_model", lambda: None)

    sentinel = lambda note: None  # noqa: E731
    module.convert_words(["sky"], method="tinyllama-katakana-converter", on_progress=sentinel)

    assert captured["on_progress"] is sentinel


def test_convert_words_releases_tinyllama_model_even_when_conversion_fails(monkeypatch):
    import vocal_analysis.english_katakana as module

    def raising_generate(word, phonemes, **kwargs):
        raise OSError("モデル取得失敗")

    monkeypatch.setattr(module, "_generate_katakana_tinyllama", raising_generate)
    release_calls = []
    monkeypatch.setattr(
        module, "release_katakana_model", lambda: release_calls.append("release")
    )

    with pytest.raises(OSError):
        module.convert_words(["sky"], method="tinyllama-katakana-converter")
    assert release_calls == ["release"]


def test_convert_words_arpakana_does_not_release_model(monkeypatch):
    import vocal_analysis.english_katakana as module

    release_calls = []
    monkeypatch.setattr(
        module, "release_katakana_model", lambda: release_calls.append("release")
    )

    module.convert_words(["sky"], method="arpakana")

    assert ("arpakana", "sky") in module._conversion_cache
    assert release_calls == []


def test_release_katakana_model_clears_process_cache(monkeypatch):
    import vocal_analysis.english_katakana as module

    module._katakana_model_cache = (object(), object())
    module.release_katakana_model()
    assert module._katakana_model_cache is None


@pytest.mark.parametrize(
    "device, expected_dtype_name",
    [("cuda", "float16"), ("cpu", "float32")],
)
def test_load_katakana_model_selects_dtype_by_device(monkeypatch, device, expected_dtype_name):
    """変換モデルのdtypeは実行デバイスに連動する(GPU実行時はfp16で重みのVRAM使用量を半減、
    CPU実行時はfp16未対応のためfp32)。"""
    import torch
    import transformers

    import vocal_analysis.english_katakana as module

    captured = {}

    class _FakeModel:
        def to(self, target):
            captured["to_device"] = target
            return self

        def eval(self):
            return self

    def fake_model_from_pretrained(model_id, revision=None, dtype=None):
        captured["dtype"] = dtype
        return _FakeModel()

    monkeypatch.setattr(module, "_katakana_model_cache", None)
    monkeypatch.setattr(module, "_select_katakana_model_device", lambda: device)
    monkeypatch.setattr(
        transformers, "AutoTokenizer",
        type("_FakeTokenizerLoader", (), {"from_pretrained": staticmethod(lambda *a, **k: object())}),
    )
    monkeypatch.setattr(
        transformers, "AutoModelForCausalLM",
        type("_FakeModelLoader", (), {"from_pretrained": staticmethod(fake_model_from_pretrained)}),
    )

    module._load_katakana_model()

    assert captured["dtype"] == getattr(torch, expected_dtype_name)
    assert captured["to_device"] == device


def test_load_katakana_model_shows_loading_note_but_no_download_note_when_already_cached(monkeypatch):
    # ダウンロードが発生しなくても、ロード自体(ディスク読み込み・GPU転送)の間は
    # 「カタカナ生成モデル読み込み中」を示す(separator._load_model_with_progress・
    # recognizer._load_model_and_processor と同じ契約)。ダウンロード進捗通知(クリアの空文字列を
    # 含む)は出ない。
    import transformers

    import vocal_analysis.english_katakana as module
    from vocal_analysis.config import ENGLISH_KATAKANA_MODEL

    class _FakeModel:
        def to(self, target):
            return self

        def eval(self):
            return self

    monkeypatch.setattr(module, "_katakana_model_cache", None)
    monkeypatch.setattr(module, "_select_katakana_model_device", lambda: "cpu")
    monkeypatch.setattr(
        transformers, "AutoTokenizer",
        type("_FakeTokenizerLoader", (), {"from_pretrained": staticmethod(lambda *a, **k: object())}),
    )
    monkeypatch.setattr(
        transformers, "AutoModelForCausalLM",
        type("_FakeModelLoader", (), {"from_pretrained": staticmethod(lambda *a, **k: _FakeModel())}),
    )
    # 事前フェッチが tqdm を一切使わない = 実際のダウンロードが発生しないケースを模す。
    monkeypatch.setattr(module, "_hf_snapshot_download", lambda repo_id, *, revision=None, tqdm_class=None: None)

    notes = []
    module._load_katakana_model(on_progress=notes.append)

    assert notes == [f"カタカナ生成モデル読み込み中: {ENGLISH_KATAKANA_MODEL.model_id}"]
