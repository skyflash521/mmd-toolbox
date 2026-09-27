import builtins
import os
import subprocess
import sys

import pytest

from vocal_analysis.reading import to_kana_reading


def _fake_convert(captured, result=None, error=None):
    def convert_with_g2p(text, *, method=None, on_progress=None, kana=False):
        captured["text"] = text
        captured["kana"] = kana
        captured["method"] = method
        if error is not None:
            raise error
        return result

    return convert_with_g2p


def _install(monkeypatch, convert):
    from vocal_analysis import recognizer

    monkeypatch.setattr(recognizer, "convert_with_g2p", convert)


def test_converts_the_whole_text_in_one_call(monkeypatch):
    captured = {}
    _install(monkeypatch, _fake_convert(captured, result="キョーワヨイテンキ"))

    assert to_kana_reading("今日は良い天気") == "キョーワヨイテンキ"
    assert captured["text"] == "今日は良い天気"
    assert captured["kana"] is True


def test_conversion_method_is_left_at_default(monkeypatch):
    captured = {}
    _install(monkeypatch, _fake_convert(captured, result="ア"))

    to_kana_reading("あ")
    assert captured["method"] is None


def test_multiple_lines_are_passed_as_one_unit(monkeypatch):
    captured = {}
    _install(monkeypatch, _fake_convert(captured, result="アイ"))

    to_kana_reading("あ\nい")
    assert captured["text"] == "あ\nい"


def test_recognition_error_passes_through_unchanged(monkeypatch):
    from vocal_analysis.recognizer import RecognitionError

    error = RecognitionError("辞書を取得できません")
    _install(monkeypatch, _fake_convert({}, error=error))

    with pytest.raises(RecognitionError) as exc:
        to_kana_reading("あ")
    assert exc.value is error


@pytest.mark.parametrize("error", [ImportError("no module named pyopenjtalk"), RuntimeError("boom")])
def test_other_failures_are_reported_as_recognition_error(monkeypatch, error):
    from vocal_analysis.recognizer import RecognitionError

    _install(monkeypatch, _fake_convert({}, error=error))

    with pytest.raises(RecognitionError) as exc:
        to_kana_reading("あ")
    assert type(error).__name__ in str(exc.value)
    assert exc.value.__cause__ is error


def test_importing_the_module_does_not_pull_in_the_extra_dependencies():
    code = ("import sys; import vocal_analysis.reading; "
            "print('vocal_analysis.recognizer' in sys.modules)")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert completed.stdout.strip() == "False"


def test_missing_extra_dependency_at_import_time_is_reported_as_recognition_error(monkeypatch):
    from vocal_analysis.phonemes import RecognitionError

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name.endswith("recognizer") or ".recognizer" in name:
            raise ImportError("No module named 'soundfile'", name="soundfile")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(RecognitionError) as exc:
        to_kana_reading("あ")
    assert "ImportError" in str(exc.value)


def test_kana_reading_passes_english_katakana_fallback_result_to_g2p(monkeypatch):
    import pyopenjtalk

    from vocal_analysis import recognizer as recognizer_module

    convert_calls = []
    monkeypatch.setattr(
        recognizer_module, "convert_target_words",
        lambda text, method=None, **kwargs: convert_calls.append(text) or "スカイ",
    )

    g2p_calls = []

    def fake_g2p(text, kana=None, join=None):
        g2p_calls.append((text, kana, join))
        return "スカイ"

    monkeypatch.setattr(pyopenjtalk, "g2p", fake_g2p)

    assert to_kana_reading("空を見上げてsky") == "スカイ"
    assert convert_calls == ["空を見上げてsky"]
    assert g2p_calls == [("スカイ", True, None)]
