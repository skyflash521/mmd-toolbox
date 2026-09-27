import pytest


@pytest.fixture(autouse=True)
def _reset_conversion_cache():
    import vocal_analysis.english_katakana as module

    module._conversion_cache.clear()
    yield
    module._conversion_cache.clear()


@pytest.mark.parametrize(
    "surface,pos,expected",
    [
        pytest.param("sky", "フィラー", True, id="ascii-filler"),
        pytest.param("sky", "名詞", False, id="already-read-as-noun"),
        pytest.param("sky", "記号", False, id="fragment-of-split-word"),
        pytest.param("sky3", "フィラー", False, id="contains-digit"),
        pytest.param("sky!", "フィラー", False, id="contains-symbol"),
        pytest.param("空", "フィラー", False, id="non-ascii"),
        pytest.param("", "フィラー", False, id="empty"),
    ],
)
def test_is_target_node(surface, pos, expected):
    from vocal_analysis.english_katakana import _is_target_node

    assert _is_target_node(surface, pos) is expected


def test_find_target_words_detects_filler_pos_ascii_word():
    from vocal_analysis.english_katakana import _find_target_words

    assert _find_target_words("空を見上げてsky") == ["sky"]


def test_find_target_words_excludes_word_already_read_as_noun():
    from vocal_analysis.english_katakana import _find_target_words

    assert _find_target_words("空を見上げてlove") == []


def test_find_target_words_excludes_non_target_when_no_ascii_word():
    from vocal_analysis.english_katakana import _find_target_words

    assert _find_target_words("空を見上げて雲をながめて") == []


def test_find_target_words_excludes_word_split_into_multiple_nodes():
    from vocal_analysis.english_katakana import _find_target_words

    assert _find_target_words("空を見上げてdog's") == []


def test_find_target_words_returns_each_occurrence_in_order():
    from vocal_analysis.english_katakana import _find_target_words

    assert _find_target_words("skyとsky") == ["sky", "sky"]


def test_lookup_cmudict_phonemes_known_word():
    from vocal_analysis.english_katakana import _lookup_cmudict_phonemes

    assert _lookup_cmudict_phonemes("sky") == "S K AY1"


def test_lookup_cmudict_phonemes_is_case_insensitive():
    from vocal_analysis.english_katakana import _lookup_cmudict_phonemes

    assert _lookup_cmudict_phonemes("Sky") == "S K AY1"


def test_lookup_cmudict_phonemes_unknown_word_returns_none():
    from vocal_analysis.english_katakana import _lookup_cmudict_phonemes

    assert _lookup_cmudict_phonemes("tiktok") is None


def test_lookup_cmudict_phonemes_uses_first_of_multiple_pronunciations(monkeypatch):
    import vocal_analysis.english_katakana as module

    monkeypatch.setattr(module, "_cmudict_cache", {"read": [["R", "IY1", "D"], ["R", "EH1", "D"]]})

    assert module._lookup_cmudict_phonemes("read") == "R IY1 D"


def test_get_cmudict_entries_auto_downloads_when_not_cached(monkeypatch):
    import nltk.corpus

    import vocal_analysis.english_katakana as module

    monkeypatch.setattr(module, "_cmudict_cache", None)
    download_calls = []
    attempts = {"n": 0}

    # nltk.corpus.cmudict は初回ロードで自身を別のオブジェクトへ置き換えるので、属性ではなく丸ごと差し替える。
    class _FakeCmudict:
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
def test_is_valid_katakana_accepts_only_hiragana_and_katakana(text, expected):
    from vocal_analysis.english_katakana import _is_valid_katakana

    assert _is_valid_katakana(text) is expected


def test_to_halfwidth_converts_fullwidth_latin():
    from vocal_analysis.english_katakana import _to_halfwidth

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

    result = _locate_and_replace("空を見上げてwordx", ["wordx"], {})

    assert result == "空を見上げてwordx"


def test_locate_and_replace_handles_multiple_occurrences_in_order():
    from vocal_analysis.english_katakana import _locate_and_replace

    result = _locate_and_replace("skyとsky", ["sky", "sky"], {"sky": "スカイ"})

    assert result == "スカイとスカイ"


def test_locate_and_replace_mixed_converted_and_unconverted():
    from vocal_analysis.english_katakana import _locate_and_replace

    result = _locate_and_replace(
        "wordxを見てskyを見上げる", ["wordx", "sky"], {"sky": "スカイ"}
    )

    assert result == "wordxを見てスカイを見上げる"


def test_locate_and_replace_matches_fullwidth_original_text():
    from vocal_analysis.english_katakana import _locate_and_replace

    result = _locate_and_replace("空を見上げてｓｋｙ", ["sky"], {"sky": "スカイ"})

    assert result == "空を見上げてスカイ"


def test_locate_and_replace_advances_cursor_past_unconverted_word():
    from vocal_analysis.english_katakana import _locate_and_replace

    result = _locate_and_replace("skyline sky", ["skyline", "sky"], {"sky": "スカイ"})

    assert result == "skyline スカイ"


def test_locate_and_replace_prefers_nearer_occurrence_when_widths_mixed():
    from vocal_analysis.english_katakana import _locate_and_replace

    result = _locate_and_replace("ｓｋｙ sky", ["sky", "sky"], {"sky": "スカイ"})

    assert result == "スカイ スカイ"


def test_generate_katakana_arpakana_converts_phonemes_to_kana():
    from vocal_analysis.english_katakana import _generate_katakana_arpakana

    assert _generate_katakana_arpakana("sky", "S K AY1") == "スカイ"


def test_generate_katakana_arpakana_returns_invalid_output_when_conversion_raises(monkeypatch):
    import arpakana

    import vocal_analysis.english_katakana as module

    def _raise(phonemes):
        raise ValueError("boom")

    monkeypatch.setattr(arpakana, "arpabet_to_kana", _raise)

    result = module._generate_katakana_arpakana("sky", "S K AY1")

    assert module._is_valid_katakana(result) is False


def test_generate_katakana_tinyllama_returns_invalid_output_when_generation_raises(monkeypatch):
    import vocal_analysis.english_katakana as module

    class _RaisingTokenizer:
        def __call__(self, *args, **kwargs):
            raise RuntimeError("boom")

    monkeypatch.setattr(
        module, "_load_katakana_model", lambda on_progress=None: (_RaisingTokenizer(), object()))

    result = module._generate_katakana_tinyllama("sky", "S K AY1")

    assert module._is_valid_katakana(result) is False


def test_generate_katakana_dispatches_to_arpakana_by_default():
    from vocal_analysis.english_katakana import _generate_katakana

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

    with pytest.raises(ValueError):
        _generate_katakana("sky", "S K AY1", method="unknown-method")


def test_convert_target_words_no_target_returns_text_unchanged():
    from vocal_analysis.english_katakana import convert_target_words

    assert convert_target_words("空を見上げて雲をながめて") == "空を見上げて雲をながめて"


def test_convert_target_words_uses_arpakana_by_default():
    from vocal_analysis.english_katakana import convert_target_words

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


def test_convert_target_words_leaves_text_unchanged_without_converting_when_cmudict_misses(monkeypatch):
    import vocal_analysis.english_katakana as module

    calls = []
    monkeypatch.setattr(module, "_lookup_cmudict_phonemes", lambda word: None)
    monkeypatch.setattr(
        module, "_generate_katakana_arpakana",
        lambda word, phonemes: calls.append(word) or "スカイ",
    )

    result = module.convert_target_words("空を見上げてsky")

    assert result == "空を見上げてsky"
    assert calls == []


def test_convert_target_words_leaves_text_unchanged_when_output_is_not_kana(monkeypatch):
    import vocal_analysis.english_katakana as module

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
    monkeypatch.setattr(module, "_hf_snapshot_download", lambda repo_id, *, revision=None, tqdm_class=None: None)

    notes = []
    module._load_katakana_model(on_progress=notes.append)

    assert notes == [f"カタカナ生成モデル読み込み中: {ENGLISH_KATAKANA_MODEL.model_id}"]
