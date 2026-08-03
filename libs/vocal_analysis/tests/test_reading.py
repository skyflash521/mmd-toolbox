"""日本語テキストのかな読みのテスト。

読みの正しさそのものは G2P の辞書と形態素解析に依存し本モジュールの契約ではないので、ここでは
公開面の契約——入力の単位・変換方式の固定・失敗の分類・取り込みが追加依存を要さないこと——を検証する。
"""

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


def test_conversion_method_is_not_selectable(monkeypatch):
    # 方式の選択は認識の品質と実行コストを測って決めるもので、読みを得るだけの用途には持ち込まない。
    captured = {}
    _install(monkeypatch, _fake_convert(captured, result="ア"))

    to_kana_reading("あ")
    assert captured["method"] is None  # 既定のまま(呼び出し側から渡さない)


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
    # 追加依存の未導入も変換そのものの失敗も、呼び出し元が1つの型で引き取れるよう同じ例外にする。
    from vocal_analysis.recognizer import RecognitionError

    _install(monkeypatch, _fake_convert({}, error=error))

    with pytest.raises(RecognitionError) as exc:
        to_kana_reading("あ")
    assert type(error).__name__ in str(exc.value)
    assert exc.value.__cause__ is error


def test_importing_the_module_does_not_pull_in_the_extra_dependencies():
    # 取り込み自体は追加依存を要さない(未導入の環境でも取り込みは成立する)。追加依存を要する
    # モジュールを取り込んでいないことを、別プロセスの取り込み後の状態で確かめる。
    code = ("import sys; import vocal_analysis.reading; "
            "print('vocal_analysis.recognizer' in sys.modules)")
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)}
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True, env=env)
    assert completed.stdout.strip() == "False"


def test_missing_extra_dependency_at_import_time_is_reported_as_recognition_error(monkeypatch):
    # 追加依存が未導入の環境では取り込みの時点で失敗する。その失敗も同じ型で分類する。
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


def test_kana_reading_goes_through_the_english_katakana_fallback(monkeypatch):
    # かな読みも音素認識と同じ経路(カタカナ化フォールバックを適用してから G2P)を通る。読みの実体は
    # 規定しないので、フォールバックの結果がそのまま G2P へ渡ることと呼び出しの形だけを固定する。
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
