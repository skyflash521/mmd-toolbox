"""S2 音素認識の統合テスト(vocal_analysis.md §5・§5.1・§8.1・§8.3)。

実モデル(wav2vec2)の呼び出しはモック(_load_model_and_processor・_run_model_inference を差し替え)
して検証し、ネットワーク・実モデルを必須にしない。ダウンミックス・リサンプルは合成配列で決定論的に
検証する。recognize() の統合テストはモックしたモデル推論結果からセグメント列が正しく組み立てられる
ことを確認する(_merge_ctc_frames・_absorb_short_segments 自体の網羅的な検証は test_recognizer.py)。
"""

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def test_downmix_to_mono_averages_channels():
    from vocal_analysis.recognizer import _downmix_to_mono

    samples = np.array([[0.2, 0.6], [-0.4, 0.0]], dtype=np.float32)

    mono = _downmix_to_mono(samples)

    assert np.allclose(mono, [0.4, -0.2])


def test_downmix_to_mono_passthrough_for_already_mono():
    from vocal_analysis.recognizer import _downmix_to_mono

    samples = np.array([[0.3], [-0.1]], dtype=np.float32)

    mono = _downmix_to_mono(samples)

    assert np.allclose(mono, [0.3, -0.1])


def test_resample_to_target_same_rate_is_passthrough():
    from vocal_analysis.recognizer import _resample_to_target

    mono = np.linspace(-0.5, 0.5, 1000, dtype=np.float32)

    result = _resample_to_target(mono, sample_rate=16000, target_sample_rate=16000)

    assert np.array_equal(result, mono)


def test_resample_to_target_downsamples_to_expected_length():
    from vocal_analysis.recognizer import _resample_to_target

    # 48000Hz -> 16000Hz(整数比 3:1)。1秒(48000サンプル)は16000サンプルになる。
    mono = np.linspace(-0.5, 0.5, 48000, dtype=np.float32)

    result = _resample_to_target(mono, sample_rate=48000, target_sample_rate=16000)

    assert len(result) == 16000


def test_ids_to_symbols_maps_via_decoder_and_blank_to_none():
    from vocal_analysis.recognizer import _ids_to_symbols

    decoder = {0: "<pad>", 1: "a", 2: "n"}
    token_ids = [1, 1, 0, 2]

    symbols = _ids_to_symbols(token_ids, pad_token_id=0, decoder=decoder)

    assert symbols == ["a", "a", None, "n"]


class _FakeTokenizer:
    def __init__(self, decoder, pad_token_id=0):
        self.pad_token_id = pad_token_id
        self._decoder = decoder

    def get_vocab(self):
        return {v: k for k, v in self._decoder.items()}


class _FakeProcessor:
    def __init__(self, decoder, pad_token_id=0):
        self.tokenizer = _FakeTokenizer(decoder, pad_token_id)


def _write_wav(path: Path, samples: np.ndarray, sample_rate: int) -> Path:
    sf.write(path, samples, sample_rate)
    return path


def test_recognize_builds_segments_from_mocked_inference(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 16kHz・モノラルの短い合成音(実値は使わずダウンミックス/リサンプル経路を通すだけ)。
    # 960サンプル@16kHz = 60ms = 3フレーム分。§5.1の60ms吸収規則の閾値ちょうどにし、
    # 単一フレーム(20ms)だと非gap隣接を持たずgap化されてしまうため、3フレーム同一ラベルにする。
    wav_path = _write_wav(tmp_path / "vocal.wav", np.zeros((960, 1), dtype=np.float32), 16000)

    decoder = {0: "<pad>", 1: "a", 2: "n"}
    fake_token_ids = [1, 1, 1]

    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_run_model_inference", lambda processor, model, samples: fake_token_ids
    )

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "vowel"
    assert segments[0].phoneme == "a"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.06)


def test_recognize_missing_library_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", np.zeros((320, 1), dtype=np.float32), 16000)

    original_error = ImportError("no transformers")

    def fake_load(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fake_load)

    with pytest.raises(recognizer_module.RecognitionError, match="transformers") as excinfo:
        recognizer_module.recognize(wav_path)

    # transformers・torch は同じ import 節で読み込むため、原因が torch 側の欠落でも
    # メッセージが transformers だけを名指しして誤解を招かないよう、両方を案内する。
    assert "torch" in str(excinfo.value)
    # 「分かるエラー」であることを検証する: 案内メッセージだけでなく元の例外を握り潰さず
    # __cause__ として保持しているか確認する(原因追跡ができるように)。
    assert excinfo.value.__cause__ is original_error


def test_recognize_model_fetch_failure_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # §4: モデルが未キャッシュでネットワークからも取得できない場合、モデル取得が必要と分かる
    # エラーで失敗させる(黙って劣化させない)。from_pretrained 系はこの場合 OSError を送出する。
    wav_path = _write_wav(tmp_path / "vocal.wav", np.zeros((320, 1), dtype=np.float32), 16000)

    original_error = OSError("model not found in cache and offline")

    def fake_load(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fake_load)

    with pytest.raises(recognizer_module.RecognitionError, match="認識モデル") as excinfo:
        recognizer_module.recognize(wav_path)

    assert excinfo.value.__cause__ is original_error


def test_load_model_and_processor_passes_pinned_config(monkeypatch):
    # transformers・torch が実際に導入されている環境でのみ、_load_model_and_processor の実体を
    # 検証する(最小環境では skip)。from_pretrained 自体をモンキーパッチするためネットワーク・
    # 実モデルのダウンロードは発生しない。_load_model_and_processor は transformers より先に
    # torch を import するため、torch 単体の有無も個別に skip 判定する。
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from vocal_analysis import RECOGNIZER_CONFIG
    from vocal_analysis.recognizer import _load_model_and_processor

    captured = {}

    class _FakeModel:
        def to(self, device):
            captured["model_to_device"] = device
            return self

        def eval(self):
            return self

    def fake_processor_from_pretrained(model_id, revision=None, do_phonemize=None, **kwargs):
        captured["processor_model_id"] = model_id
        captured["processor_revision"] = revision
        captured["do_phonemize"] = do_phonemize
        return _FakeProcessor({0: "<pad>"})

    def fake_model_from_pretrained(model_id, revision=None, torch_dtype=None, **kwargs):
        captured["model_model_id"] = model_id
        captured["model_revision"] = revision
        captured["model_torch_dtype"] = torch_dtype
        return _FakeModel()

    def fake_manual_seed(seed):
        captured["manual_seed"] = seed

    def fake_set_num_threads(num_threads):
        captured["num_threads"] = num_threads

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_processor_from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCTC, "from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr(torch, "manual_seed", fake_manual_seed)
    monkeypatch.setattr(torch, "set_num_threads", fake_set_num_threads)

    _load_model_and_processor()

    # §5.1: wav2vec2-espeak のトークナイザは既定で espeak ネイティブバイナリ(phonemizer)を要求
    # するため do_phonemize=False を渡してこの依存を回避する(音素IDのデコードのみが必要で
    # 音素へのエンコードは不要なため)。
    assert captured["do_phonemize"] is False
    # §8.3: モデル id・revision を固定する。
    assert captured["processor_model_id"] == RECOGNIZER_CONFIG.model_id
    assert captured["processor_revision"] == RECOGNIZER_CONFIG.model_revision
    assert captured["model_model_id"] == RECOGNIZER_CONFIG.model_id
    assert captured["model_revision"] == RECOGNIZER_CONFIG.model_revision
    # §5.1: 実行デバイス・dtype・スレッド数・乱数シードを固定条件どおりに適用する。
    assert captured["model_torch_dtype"] == getattr(torch, RECOGNIZER_CONFIG.dtype)
    assert captured["model_to_device"] == RECOGNIZER_CONFIG.device
    assert captured["num_threads"] == RECOGNIZER_CONFIG.num_threads
    assert captured["manual_seed"] == RECOGNIZER_CONFIG.random_seed
