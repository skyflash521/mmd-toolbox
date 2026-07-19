"""S1 ボーカル抽出のテスト。

外部ライブラリ(audio-separator)を呼ぶ分離実行は、内部の分離器ファクトリ(_build_separator)を
差し替えてモックし、ネットワーク・実モデルを必須にしない。mode=never(分離なし)は純粋な
PCM→WAV書き出しであり、audio-separator が未導入でも動く(実ファイルで検証する)。
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def _make_pcm(sample_rate=8000, duration_sec=0.2, channels=2):
    from vocal_analysis.types import AudioPcm

    frames = int(sample_rate * duration_sec)
    rng = np.random.default_rng(0)
    samples = ((rng.random((frames, channels)) - 0.5) * 0.5).astype(np.float32)
    return AudioPcm(samples=samples, sample_rate=sample_rate)


def test_separate_never_mode_returns_input_unseparated():
    from vocal_analysis.separator import separate

    pcm = _make_pcm()

    result = separate(pcm, mode="never")

    assert isinstance(result, Path)
    assert result.exists()
    samples, sr = sf.read(result, dtype="float32", always_2d=True)
    assert sr == pcm.sample_rate
    assert np.allclose(samples, pcm.samples, atol=1e-4)


def test_separate_never_mode_does_not_call_separator_factory(monkeypatch):
    from vocal_analysis import separator as separator_module

    # never は分離せず入力をそのままボーカルとして扱う。分離器を呼んではならない
    # (audio-separator 未導入環境でも never モードだけは動くことを保証する)。
    def _fail(output_dir):
        raise AssertionError("never モードで分離器ファクトリを呼んではならない")

    monkeypatch.setattr(separator_module, "_build_separator", _fail)

    result = separator_module.separate(_make_pcm(), mode="never")

    assert result.exists()


def test_separate_invalid_mode_raises_value_error():
    from vocal_analysis.separator import separate

    with pytest.raises(ValueError):
        separate(_make_pcm(), mode="bogus")


class _FakeSeparator:
    def __init__(self, calls):
        self._calls = calls

    def load_model(self, model_filename):
        self._calls["model_filename"] = model_filename

    def separate(self, audio_file_path):
        self._calls["audio_file_path"] = audio_file_path
        out = Path(audio_file_path).parent / "vocals_output.wav"
        out.write_bytes(b"")  # ダミー出力(内容は検証対象外)
        # 実際の audio-separator(0.44.2)は output_dir 相対のファイル名だけを返す
        # (絶対パスではない)。この挙動を模して separate() 側の解決ロジックを検証する。
        return [out.name]


def test_separate_always_mode_calls_separator_factory(monkeypatch):
    from vocal_analysis import separator as separator_module

    calls = {}

    def fake_build_separator(output_dir):
        calls["output_dir"] = output_dir
        return _FakeSeparator(calls)

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)

    result = separator_module.separate(_make_pcm(), mode="always")

    # 出力は「ボーカルWAVのパス」を約束するため、返るパスが実在することも検証する。
    assert result == Path(calls["audio_file_path"]).parent / "vocals_output.wav"
    assert result.exists()
    assert calls["model_filename"] == "htdemucs_ft.yaml"


def test_separate_uses_pinned_separator_config(monkeypatch):
    from vocal_analysis import SEPARATOR_CONFIG
    from vocal_analysis import separator as separator_module

    calls = {}

    def fake_build_separator(output_dir):
        return _FakeSeparator(calls)

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)

    separator_module.separate(_make_pcm(), mode="always")

    assert calls["model_filename"] == SEPARATOR_CONFIG.model_filename


def test_separate_registers_work_dir_cleanup_that_removes_it(monkeypatch):
    import shutil

    from vocal_analysis import separator as separator_module

    registered = []
    monkeypatch.setattr(
        separator_module.atexit, "register",
        lambda fn, *args, **kwargs: registered.append((fn, args, kwargs)))

    result = separator_module.separate(_make_pcm(), mode="never")
    work_dir = result.parent
    assert work_dir.exists()

    assert len(registered) == 1
    fn, args, kwargs = registered[0]
    assert fn is shutil.rmtree
    assert args == (work_dir,)
    assert kwargs == {"ignore_errors": True}
    fn(*args, **kwargs)

    assert not work_dir.exists()


def test_separate_missing_library_raises_clear_error(monkeypatch):
    from vocal_analysis import separator as separator_module

    def fake_build_separator(output_dir):
        raise ImportError("no audio_separator")

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)

    with pytest.raises(separator_module.SeparationError):
        separator_module.separate(_make_pcm(), mode="always")


def test_build_separator_configures_real_separator_with_pinned_values():
    # audio-separator が実際に導入されている環境でのみ、_build_separator の実体を検証する
    # (vocal-analysis extra が無い最小環境では skip。実モデル・ネットワークは必須にしない)。
    pytest.importorskip("audio_separator")
    from vocal_analysis import SEPARATOR_CONFIG
    from vocal_analysis.separator import _build_separator

    with tempfile.TemporaryDirectory() as tmp_dir:
        sep = _build_separator(Path(tmp_dir))

    assert sep.output_single_stem == SEPARATOR_CONFIG.output_single_stem
    # Separator は demucs_params 引数を内部で arch_specific_params["Demucs"] へ格納する
    # (audio-separator 実装の実際の格納先。コンストラクタ引数名とインスタンス属性名は異なる)。
    assert sep.arch_specific_params["Demucs"]["shifts"] == SEPARATOR_CONFIG.shifts
