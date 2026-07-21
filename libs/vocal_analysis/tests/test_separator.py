"""S1 ボーカル抽出のテスト。

外部ライブラリ(audio-separator)を呼ぶ分離実行は、内部の分離器ファクトリ(_build_separator)を
差し替えてモックし、ネットワーク・実モデルを必須にしない。mode=never(分離なし)は純粋な
PCM→WAV書き出しであり、audio-separator が未導入でも動く(実ファイルで検証する)。
"""

import logging
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


def _fake_download_file_if_not_exists(self, url, output_path):
    """audio_separator.separator.separator.Separator.download_file_if_not_exists の代わり。

    実装(.venv 内 separator.py の download_file_if_not_exists)と同じく、ファイルサイズぶんの
    tqdm 進捗バーを生成して update する(URL への実アクセスはしない)。
    """
    import audio_separator.separator.separator as as_mod

    bar = as_mod.tqdm(total=100, unit="iB", unit_scale=True)
    bar.update(40)
    bar.update(60)
    bar.close()


def test_separate_relays_download_progress_when_on_progress_given(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.separator as as_mod

    from vocal_analysis import separator as separator_module

    calls = {}
    notes = []

    class _FakeSeparatorWithDownload(_FakeSeparator):
        def load_model(self, model_filename):
            as_mod.Separator.download_file_if_not_exists(
                self, "https://example.invalid/htdemucs_ft.yaml", "/models/htdemucs_ft.yaml")
            super().load_model(model_filename)

        def separate(self, audio_file_path):
            # クリア通知(空文字列)がロード完了直後に来ることを検証するため、実際の分離処理
            # (ダウンロードと無関係)の開始をマーカーとして記録する。
            notes.append("SEPARATE_STARTED")
            return super().separate(audio_file_path)

    def fake_build_separator(output_dir):
        return _FakeSeparatorWithDownload(calls)

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)
    monkeypatch.setattr(as_mod.Separator, "download_file_if_not_exists", _fake_download_file_if_not_exists)

    separator_module.separate(_make_pcm(), mode="always", on_progress=notes.append)

    assert any("40" in n for n in notes)
    assert any("100" in n for n in notes)
    # ファイル番号のような合成ラベルではなく、実際のファイル名(output_path のファイル名部分)を使う。
    assert any("htdemucs_ft.yaml" in n for n in notes)
    # クリア通知(空文字列)はロード完了直後に来る。実際の分離処理(ダウンロードと無関係)が
    # 始まるより前にクリアされていることを確認する(分離処理完了後まで遅延してはならない)。
    assert notes.index("") < notes.index("SEPARATE_STARTED")


def test_separate_download_progress_labels_multi_file_models_by_real_filename(monkeypatch):
    # htdemucs_ft は重み4分割+YAMLの計5ファイルで構成され、audio-separator は
    # download_file_if_not_exists をファイルごとに呼ぶ(バイト集約された単一の合計進捗が無い)。
    # 各ファイルで tqdm が 0%→100% を再スタートするため、通知に実際のファイル名を含めて
    # 「壊れて繰り返している」ように見えず、どのファイルの進捗かを正しく識別できることを検証する。
    pytest.importorskip("audio_separator")
    import audio_separator.separator.separator as as_mod

    from vocal_analysis import separator as separator_module

    calls = {}
    notes = []
    filenames = ["04573f0d-f3cf25b2.th", "htdemucs_ft.yaml"]

    class _FakeSeparatorWithMultiFileDownload(_FakeSeparator):
        def load_model(self, model_filename):
            for name in filenames:
                as_mod.Separator.download_file_if_not_exists(
                    self, f"https://example.invalid/{name}", f"/models/{name}")
            super().load_model(model_filename)

    def fake_build_separator(output_dir):
        return _FakeSeparatorWithMultiFileDownload(calls)

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)
    monkeypatch.setattr(as_mod.Separator, "download_file_if_not_exists", _fake_download_file_if_not_exists)

    separator_module.separate(_make_pcm(), mode="always", on_progress=notes.append)

    file1_notes = [n for n in notes if filenames[0] in n]
    file2_notes = [n for n in notes if filenames[1] in n]
    assert any("100" in n for n in file1_notes)
    assert any("100" in n for n in file2_notes)
    # 2ファイル目の通知は1ファイル目のファイル名を含まず、両者が混同されない。
    assert not any(filenames[1] in n for n in file1_notes)


def test_separate_shows_loading_note_but_no_download_note_when_no_download_happens(monkeypatch):
    pytest.importorskip("audio_separator")
    from vocal_analysis import SEPARATOR_CONFIG
    from vocal_analysis import separator as separator_module

    calls = {}
    notes = []

    def fake_build_separator(output_dir):
        # load_model が tqdm を一切使わない = 実際のダウンロードが発生しないケースを模す
        # (audio-separator が既にキャッシュ済みファイルを検出して download_file_if_not_exists を
        # 早期returnする経路に相当)。
        return _FakeSeparator(calls)

    monkeypatch.setattr(separator_module, "_build_separator", fake_build_separator)

    separator_module.separate(_make_pcm(), mode="always", on_progress=notes.append)

    # ダウンロードが発生しなくても、ロード自体(ディスク読み込み・GPU転送)の間は
    # 「モデル読み込み中」を示す。「ダウンロード中」やクリア(空文字列)は出ない
    # (ダウンロードが実際に発生した場合だけの通知のため)。
    assert notes == [f"モデル読み込み中: {SEPARATOR_CONFIG.model_filename}"]


def test_load_model_with_progress_restores_original_tqdm_and_download_method(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.separator as as_mod

    from vocal_analysis.separator import _load_model_with_progress

    original_tqdm = as_mod.tqdm
    original_download = as_mod.Separator.download_file_if_not_exists

    class _FakeSep:
        def load_model(self, model_filename):
            # ロード中だけ両方とも中継用に差し替わっている。
            assert as_mod.tqdm is not original_tqdm
            assert as_mod.Separator.download_file_if_not_exists is not original_download

    _load_model_with_progress(_FakeSep(), on_progress=lambda note: None)

    assert as_mod.tqdm is original_tqdm
    assert as_mod.Separator.download_file_if_not_exists is original_download


def test_load_model_with_progress_skips_relay_when_on_progress_is_none():
    pytest.importorskip("audio_separator")
    import audio_separator.separator.separator as as_mod

    from vocal_analysis import SEPARATOR_CONFIG
    from vocal_analysis.separator import _load_model_with_progress

    original = as_mod.tqdm
    calls = {}

    class _FakeSep:
        def load_model(self, model_filename):
            calls["model_filename"] = model_filename
            assert as_mod.tqdm is original  # 差し替えが起きていないこと

    downloaded = _load_model_with_progress(_FakeSep(), on_progress=None)

    assert downloaded is False
    assert calls["model_filename"] == SEPARATOR_CONFIG.model_filename


def test_separate_with_progress_relays_inference_progress(monkeypatch):
    # audio-separator の DemucsSeparator.demix_demucs は apply_model(set_progress_bar=None) を
    # 固定引数で呼ぶため、_separate_with_progress が demucs_separator モジュールの apply_model 名を
    # 差し替えてコールバックを注入する。ここでは apply_model 自体を fake 化し、注入された
    # set_progress_bar が実際に呼ばれ、その fraction(0-0.8を100%に正規化)が on_progress へ
    # 中継されることを検証する(実モデル・実推論は行わない)。
    pytest.importorskip("audio_separator")
    import audio_separator.separator.architectures.demucs_separator as demucs_mod

    from vocal_analysis.separator import _separate_with_progress

    def fake_apply_model(*args, set_progress_bar=None, **kwargs):
        set_progress_bar(0.1, 0.4)
        set_progress_bar(0.1, 0.8)
        return "dummy_source"

    monkeypatch.setattr(demucs_mod, "apply_model", fake_apply_model)

    class _FakeSep:
        def separate(self, audio_file_path):
            demucs_mod.apply_model(set_progress_bar=None)
            return ["vocals_output.wav"]

    notes = []
    _separate_with_progress(_FakeSep(), Path("input.wav"), on_progress=notes.append)

    assert notes == ["分離中: 50%", "分離中: 100%"]


def test_separate_with_progress_restores_original_apply_model(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.architectures.demucs_separator as demucs_mod

    from vocal_analysis.separator import _separate_with_progress

    original_apply_model = demucs_mod.apply_model

    class _FakeSep:
        def separate(self, audio_file_path):
            assert demucs_mod.apply_model is not original_apply_model
            return ["vocals_output.wav"]

    _separate_with_progress(_FakeSep(), Path("input.wav"), on_progress=lambda note: None)

    assert demucs_mod.apply_model is original_apply_model


def test_separate_with_progress_skips_relay_when_on_progress_is_none(monkeypatch):
    pytest.importorskip("audio_separator")
    import audio_separator.separator.architectures.demucs_separator as demucs_mod

    from vocal_analysis.separator import _separate_with_progress

    original_apply_model = demucs_mod.apply_model

    class _FakeSep:
        def separate(self, audio_file_path):
            assert demucs_mod.apply_model is original_apply_model  # 差し替えが起きていないこと
            return ["vocals_output.wav"]

    _separate_with_progress(_FakeSep(), Path("input.wav"), on_progress=None)


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
    assert sep.logger.level == logging.CRITICAL
