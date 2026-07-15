"""S2 音素認識の統合テスト(vocal_analysis.md §5・§5.2・§8.1・§8.3)。

無音検出による区間分割(§5.2手順1・2)は合成音声(実RMS)でそのまま検証し、外部呼び出し(内容認識器の
書き起こし・G2P・音素モデル推論)はモック(_transcribe_segment・_load_content_recognizer_pipeline・
_g2p・_load_model_and_processor・_compute_log_probs を差し替え)して検証する(ネットワーク・実モデルを
必須にしない)。ダウンミックス・
リサンプルは合成配列で決定論的に検証する。recognize() の統合テストは、実RMSによる区間分割・無音判定と、
モックした書き起こし・G2P・推論結果から複合構成の純関数群(_assemble_phoneme_sequence・
_assemble_with_word_windows・_g2p_symbols_to_token_ids・_forced_align・_forced_align_windowed・
_path_to_segments。各関数自体の網羅的な検証は test_recognizer.py)を経て正しくセグメント列が
組み立てられることを確認する。
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


def _loud_samples(num_samples: int, sample_rate: int = 16000, amplitude: float = 0.5) -> np.ndarray:
    """無音判定のRMSしきい値を確実に上回る、振幅一定の正弦波(§5.2手順1・2の無音検出を通過させる)。"""
    t = np.arange(num_samples) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32).reshape(-1, 1)


def test_recognize_builds_segments_from_mocked_pipeline(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 0.12秒(モックする対数確率行列の6フレーム分ちょうど)・振幅一定の音声。内部に無音区間が
    # 無いため単一の区間として扱われ、区間の終端が対数確率行列のフレーム数と一致する。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    # 区間のテキスト「あ」-> G2P「a」-> 音素列 ["pau","a","pau"]。語彙は blank=0・a=1 の2記号のみ。
    # 最小滞在制約(§5.2手順7)により a は (6-2)//1=4 個のサブ状態へ展開され、6フレームに
    # サブ状態6個がちょうど収まるため経路は [0,1,1,1,1,2] に一意に定まる(a はフレーム1〜4)。
    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [
            [5.0, -5.0],
            [5.0, -5.0],
            [-5.0, 5.0],
            [-5.0, 5.0],
            [5.0, -5.0],
            [5.0, -5.0],
        ]
    )

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 3
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.02)
    assert segments[1].type == "vowel"
    assert segments[1].phoneme == "a"
    assert segments[1].start_sec == pytest.approx(0.02)
    assert segments[1].end_sec == pytest.approx(0.10)
    assert segments[2].type == "gap"
    assert segments[2].phoneme is None
    assert segments[2].start_sec == pytest.approx(0.10)
    assert segments[2].end_sec == pytest.approx(0.12)


def test_recognize_skips_content_recognition_for_silent_segment(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 先頭2秒は大音量、続く3秒は完全な無音(RMS=0)。無音区間は内容認識呼び出し自体をスキップする
    # (§5.2手順2)。分割点は無音区間[2.0,5.0)の中点3.5秒に立ち、区間[0,3.5)は大音量を含むため
    # 無音でなく、区間[3.5,5.0)は完全に無音になる。
    loud = _loud_samples(32000)
    silence = np.zeros((48000, 1), dtype=np.float32)
    wav_path = _write_wav(tmp_path / "vocal.wav", np.concatenate([loud, silence], axis=0), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )
    call_count = {"transcribe": 0}

    def fake_transcribe(pipeline, samples):
        call_count["transcribe"] += 1
        return "あ", None

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    assert call_count["transcribe"] == 1  # 無音区間では内容認識を呼ばない
    assert segments[-1].type == "gap"
    assert segments[-1].phoneme is None
    assert segments[-1].end_sec == pytest.approx(5.0)  # 音声全体の終端まで被覆する


def test_recognize_loads_content_recognizer_pipeline_once_for_multiple_segments(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 大音量2.0秒 - 無音1.0秒(分割点を作る) - 大音量2.0秒。分割点は無音区間[2.0,3.0)の中点2.5秒に
    # 立ち、区間[0,2.5)・[2.5,5.0)はいずれも大音量部分を含み無音でないため、内容認識を2回呼ぶ。
    # パイプラインは区間ごとに再ロードせず、無音でない最初の区間で一度だけロードして使い回すべき
    # である(区間ごとの再ロードはモデル転送コスト、特にGPU使用時に致命的な浪費になる)。
    loud1 = _loud_samples(32000)
    silence = np.zeros((16000, 1), dtype=np.float32)
    loud2 = _loud_samples(32000)
    wav_path = _write_wav(
        tmp_path / "vocal.wav", np.concatenate([loud1, silence, loud2], axis=0), 16000
    )

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )
    load_calls = {"count": 0}

    def fake_load_pipeline(content_recognizer_model):
        load_calls["count"] += 1
        return object()

    def fake_transcribe(pipeline, samples):
        return "あ", None

    monkeypatch.setattr(recognizer_module, "_load_content_recognizer_pipeline", fake_load_pipeline)
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    recognizer_module.recognize(wav_path)

    assert load_calls["count"] == 1  # 2区間とも内容認識を呼ぶが、パイプラインは使い回す


def test_recognize_trims_leading_silence_and_offsets_segments(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 先頭1.5秒は完全な無音(RMS=0)、続く2.0秒は大音量。先頭の無音区間[0,1.5)は分割点(中点0.75秒)を
    # 立てるが、区間[0,0.75)は最小長1.5秒未満のため次と結合され、全体が1つの非無音区間[0,3.5)になる。
    # §5.2手順2のトリムにより、有声スパン(1.5秒から。100msフレーム単位)の外側余白100msを残した
    # 1.4秒より前は gap として直接確定され、アライメント結果のセグメントは絶対時刻1.4秒起点で
    # オフセットされるべきである(先頭無音上にトークンを置かない)。
    silence = np.zeros((24000, 1), dtype=np.float32)
    loud = _loud_samples(32000)
    wav_path = _write_wav(tmp_path / "vocal.wav", np.concatenate([silence, loud], axis=0), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    # 最小滞在制約により a は4サブ状態へ展開され、6フレームのモック行列では経路が
    # [0,1,1,1,1,2](a はローカル[0.02,0.10))に一意に定まる。トリムgap[0,1.4) と
    # ローカル先頭pau由来gap[1.4,1.42) は結合されて1本の先頭gapになる。
    assert len(segments) == 3
    assert segments[0].type == "gap"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(1.42)
    assert segments[1].type == "vowel"
    assert segments[1].phoneme == "a"
    assert segments[1].start_sec == pytest.approx(1.42)
    assert segments[1].end_sec == pytest.approx(1.50)
    assert segments[2].type == "gap"
    assert segments[2].start_sec == pytest.approx(1.50)
    assert segments[2].end_sec == pytest.approx(3.5)  # 末尾トリム無し: 区間終端まで被覆


def test_recognize_treats_high_phoneme_density_chunk_as_gap(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 2.0秒の有声区間に対しG2Pが100音素(密度50/秒。§5.2手順4のしきい値20/秒を大幅に超える)を
    # 返すケース(内容認識の反復幻覚を模す。リトライも同じ結果を返し、テキストに末尾反復が無いため
    # 反復救済も適用されない)。誤った音素列で強制アライメントを試みず、区間全体をgapとして確定し、
    # 音素モデルも一度もロードしない。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)  # 2.0秒・無音区間なし

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment",
        lambda pipeline, samples: ("あいうえおかきくけこさしすせそ", None))
    monkeypatch.setattr(
        recognizer_module, "_transcribe_text_only",
        lambda samples, content_recognizer_model: "あいうえおかきくけこさしすせそ")
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: ["a"] * 100)

    def fail_if_called():
        raise AssertionError("音素密度が高い区間で音素モデルをロードしてはならない")

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fail_if_called)

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(2.0)


def test_recognize_empty_transcription_confirms_gap_without_g2p_or_model(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 書き起こしが空文字列(空白のみを含む)の区間は、§5.2手順3によりG2P・強制アライメントを
    # 試みず区間全体をgapとして直接確定する(手順2の無音確定・手順4の音素密度超過確定と同様)。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)  # 2.0秒・無音区間なし

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment",
        lambda pipeline, samples: ("  ", None))

    def fail_g2p(text, method=None):
        raise AssertionError("空文字列の区間でG2Pを呼んではならない")

    def fail_load_model():
        raise AssertionError("空文字列の区間で音素モデルをロードしてはならない")

    monkeypatch.setattr(recognizer_module, "_g2p", fail_g2p)
    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fail_load_model)

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(2.0)


def test_recognize_computes_and_passes_word_windows(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 単語タイムスタンプが取得できた場合、§5.2手順5の窓を計算して単語窓制約Viterbi
    # (_forced_align_windowed)へ渡す(位置バンド制限へのフォールバックではない)。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)  # 0.12秒・6フレーム

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )
    captured = {}

    def fake_transcribe(pipeline, samples):
        return "あ", [("あ", 0.02, 0.06)]

    def fake_forced_align_windowed(log_probs_arg, token_ids, windows_sec):
        captured["windows_sec"] = windows_sec
        # 実際の経路組み立ては既存のバンド制限Viterbiへ委譲する(窓の受け渡しだけを検証する)。
        return recognizer_module._forced_align(log_probs_arg, token_ids)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )
    monkeypatch.setattr(recognizer_module, "_forced_align_windowed", fake_forced_align_windowed)

    segments = recognizer_module.recognize(wav_path)

    assert "windows_sec" in captured  # 単語窓制約経路が呼ばれた
    margin = recognizer_module._WORD_WINDOW_MARGIN_SEC
    # seq=["pau","a","pau"]。最小滞在は単語ごとの局所適応(§5.2手順7)で、「あ」の実時間
    # 0.02s-0.06s(2フレーム)を音素数1で割った2フレームへ展開されるため、窓は
    # [先頭pau, a, a, 末尾pau] の4個(log_probsの6フレームのうち残り2フレームはViterbi自身が
    # stay遷移で埋める)。
    windows_sec = captured["windows_sec"]
    assert len(windows_sec) == 4
    assert windows_sec[0] == pytest.approx((0.0, 0.02 + margin))
    assert all(w == pytest.approx((0.02 - margin, 0.06 + margin)) for w in windows_sec[1:3])
    assert windows_sec[3] == pytest.approx((0.06 - margin, 0.12))
    assert segments[1].phoneme == "a"


def test_recognize_falls_back_to_band_alignment_when_word_window_infeasible(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 単語窓制約Viterbiが末尾トークンへ到達できず RecognitionError を送出した場合、
    # §5.2手順7のフォールバック(位置バンド制限)へ切り替えて正常にセグメントを組み立てる
    # (エラーを外へ伝播させない)。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )

    def fake_transcribe(pipeline, samples):
        return "あ", [("あ", 0.02, 0.06)]

    def fail_windowed(log_probs_arg, token_ids, windows_sec):
        raise recognizer_module.RecognitionError("単語窓制約下で強制アライメントが末尾トークンへ到達できませんでした")

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )
    monkeypatch.setattr(recognizer_module, "_forced_align_windowed", fail_windowed)

    segments = recognizer_module.recognize(wav_path)

    assert segments[1].type == "vowel"
    assert segments[1].phoneme == "a"


def test_recognize_falls_back_to_global_min_stay_when_windowed_alignment_fails(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 単語窓制約Viterbiが失敗した場合、フォールバック(位置バンド制限)は単語ごとの局所適応
    # (_expand_min_stay_local)ではなく、チャンク全体の最小滞在(_expand_min_stay)を計算し
    # 直したサブ状態を使う(§5.2手順7)。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )
    calls = {"local": 0, "global": 0}

    def fake_transcribe(pipeline, samples):
        return "あ", [("あ", 0.02, 0.06)]

    def fail_windowed(log_probs_arg, token_ids, windows_sec):
        raise recognizer_module.RecognitionError("単語窓制約下で強制アライメントが末尾トークンへ到達できませんでした")

    original_local = recognizer_module._expand_min_stay_local
    original_global = recognizer_module._expand_min_stay

    def spy_local(*args, **kwargs):
        calls["local"] += 1
        return original_local(*args, **kwargs)

    def spy_global(*args, **kwargs):
        calls["global"] += 1
        return original_global(*args, **kwargs)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )
    monkeypatch.setattr(recognizer_module, "_forced_align_windowed", fail_windowed)
    monkeypatch.setattr(recognizer_module, "_expand_min_stay_local", spy_local)
    monkeypatch.setattr(recognizer_module, "_expand_min_stay", spy_global)

    segments = recognizer_module.recognize(wav_path)

    assert calls["local"] == 1  # 主経路でまず局所適応を試みる
    assert calls["global"] == 1  # 窓制約の失敗を受けてチャンク全体の最小滞在を計算し直す
    assert segments[1].type == "vowel"
    assert segments[1].phoneme == "a"


def test_recognize_places_multiple_words_via_per_word_g2p_and_inter_word_window(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 単語2個("あ"→G2P"a"、"い"→G2P"i")。単語ごとに個別G2Pされ(辞書引きモックのため、結合
    # テキストを1回で変換する誤実装ならKeyErrorになる)、母音が正しい順序で配置されることを
    # 検証する(§5.2手順4・5)。窓・単語間pauの拘束力そのものは他のテスト
    # (test_recognize_computes_and_passes_word_windows・test_forced_align_windowed_*)が担う
    # (この音声長0.14秒は余白0.75秒に対し極めて短く、ここでは窓は実質非拘束になる)。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(2240), 16000)  # 0.14秒・7フレーム

    decoder = {0: "<pad>", 1: "a", 2: "i"}
    log_probs = np.array(
        [
            [5.0, -5.0, -5.0],   # frame0: 先頭pau
            [-5.0, 5.0, -5.0],   # frame1-2: a
            [-5.0, 5.0, -5.0],
            [5.0, -5.0, -5.0],   # frame3: 単語間pau
            [-5.0, -5.0, 5.0],   # frame4-5: i
            [-5.0, -5.0, 5.0],
            [5.0, -5.0, -5.0],   # frame6: 末尾pau
        ]
    )

    def fake_transcribe(pipeline, samples):
        return "あ い", [("あ", 0.02, 0.04), ("い", 0.06, 0.08)]

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    # 全文("あ い"。トリガ判定の密度算出に使う)と単語ごとの両方の呼び出しに応える。
    monkeypatch.setattr(
        recognizer_module, "_g2p",
        lambda text, method=None: {"あ": ["a"], "い": ["i"], "あ い": ["a", "i"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    vowels = [(s.phoneme, s.start_sec, s.end_sec) for s in segments if s.type == "vowel"]
    assert [p for p, _, _ in vowels] == ["a", "i"]
    assert vowels[0][2] <= vowels[1][1]  # 「あ」は「い」より前に置かれる


def test_recognize_hallucination_density_sums_across_words(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 単語ごとにG2Pする場合の音素密度(§5.2手順4)は全単語分の音素数の合計で判定する。
    # 1.0秒の区間に単語2個・各15音素を割り当てる: 単語1個分だけ(15音素/秒)ならしきい値
    # (20音素/秒)未満で見逃すが、2単語の合計(30音素/秒)なら明確に超過する。単語1個分しか
    # 数えない誤実装ならこの区間をgap確定できず音素モデルをロードしてしまい、fail_if_calledで
    # 検出できる。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(16000), 16000)  # 1.0秒・無音区間なし

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment",
        lambda pipeline, samples: ("あ い", [("あ", 0.1, 0.4), ("い", 0.5, 0.9)]))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: ["a"] * 15)

    def fail_if_called():
        raise AssertionError("音素密度が高い区間で音素モデルをロードしてはならない")

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fail_if_called)

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "gap"


def test_recognize_empty_word_list_with_nonempty_text_falls_back_like_no_timestamps(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 書き起こしは非空だが単語タイムスタンプが空リスト(chunksはあったが単語として使える要素が
    # 無かった場合)は、Noneと同様に「単語タイムスタンプが取得できない場合」として扱い、
    # 単語ごとのG2P・単語窓制約を使わず、区間の書き起こし全体を1回で変換して位置バンド制限で
    # 整列する(§5.2手順3・4・5)。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )

    def fail_windowed(*args, **kwargs):
        raise AssertionError("単語タイムスタンプ非取得扱いでは単語窓制約経路を呼んではならない")

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", []))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )
    monkeypatch.setattr(recognizer_module, "_forced_align_windowed", fail_windowed)

    segments = recognizer_module.recognize(wav_path)

    assert segments[1].phoneme == "a"


def test_recognize_fully_silent_input_never_loads_phoneme_model(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 音声全体が無音(RMS=0)の場合、§5.2手順2により全区間が内容認識呼び出し無しでgap確定され、
    # 音素モデル(_load_model_and_processor)は一度も必要にならない(不要な依存失敗を避ける)。
    wav_path = _write_wav(tmp_path / "vocal.wav", np.zeros((32000, 1), dtype=np.float32), 16000)

    def fail_if_called():
        raise AssertionError("無音のみの入力で音素モデルをロードしてはならない")

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fail_if_called)

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "gap"
    assert segments[0].phoneme is None
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(2.0)


def test_recognize_last_local_segment_extends_exactly_to_segment_boundary(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 2秒(=100フレーム@20ms)の区間に対し、モックする対数確率行列はわざと6フレームしか無い
    # (実モデルでも丸め等でフレーム数が区間の秒数からずれ得ることを模する)。区間の最後の区切りは
    # フレーム数由来の終端ではなく、区間自身の終端(2.0秒)まで強制的に延ばされるべきである。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    assert segments[-1].type == "gap"
    assert segments[-1].end_sec == pytest.approx(2.0)


def test_recognize_content_recognizer_missing_library_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    original_error = ImportError("no transformers")
    decoder = {0: "<pad>"}
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )

    def fake_transcribe(pipeline, samples):
        raise original_error

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)

    with pytest.raises(recognizer_module.RecognitionError, match="transformers") as excinfo:
        recognizer_module.recognize(wav_path)

    # transformers・torch は同じ import 節で読み込むため、原因が torch 側の欠落でも
    # メッセージが transformers だけを名指しして誤解を招かないよう、両方を案内する。
    assert "torch" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_error


def test_recognize_content_recognizer_model_fetch_failure_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # §4: モデルが未キャッシュでネットワークからも取得できない場合、モデル取得が必要と分かる
    # エラーで失敗させる(黙って劣化させない)。from_pretrained 系はこの場合 OSError を送出する。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    original_error = OSError("model not found in cache and offline")
    decoder = {0: "<pad>"}
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )

    def fake_transcribe(pipeline, samples):
        raise original_error

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)

    with pytest.raises(recognizer_module.RecognitionError, match="内容認識モデル") as excinfo:
        recognizer_module.recognize(wav_path)

    assert excinfo.value.__cause__ is original_error


def test_recognize_g2p_missing_library_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    original_error = ImportError("no pyopenjtalk")
    decoder = {0: "<pad>"}
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))

    def fake_g2p(text, method=None):
        raise original_error

    monkeypatch.setattr(recognizer_module, "_g2p", fake_g2p)

    with pytest.raises(recognizer_module.RecognitionError, match="pyopenjtalk-plus") as excinfo:
        recognizer_module.recognize(wav_path)

    assert excinfo.value.__cause__ is original_error


def test_recognize_phoneme_model_missing_library_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 音素モデルは 内容認識・G2P が両方成功した後(初回の非無音区間)に遅延ロードされるため、
    # 実モデルを呼ばずにそこへ到達させるには _transcribe_segment・_g2p もモックする必要がある。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    original_error = ImportError("no transformers")
    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: ["a"])

    def fake_load(*args, **kwargs):
        raise original_error

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fake_load)

    with pytest.raises(recognizer_module.RecognitionError, match="transformers") as excinfo:
        recognizer_module.recognize(wav_path)

    assert "torch" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_error


def test_recognize_phoneme_model_fetch_failure_raises_clear_error(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)

    original_error = OSError("model not found in cache and offline")
    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: ["a"])

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

    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", fake_processor_from_pretrained)
    monkeypatch.setattr(transformers.AutoModelForCTC, "from_pretrained", fake_model_from_pretrained)
    monkeypatch.setattr(torch, "manual_seed", fake_manual_seed)

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
    # §5.1: 実行デバイス・dtype・乱数シードを固定条件どおりに適用する(スレッド数は推論時に
    # _compute_log_probs が局所適用する。下記 test_compute_log_probs_scopes_num_threads_...)。
    assert captured["model_torch_dtype"] == getattr(torch, RECOGNIZER_CONFIG.dtype)
    assert captured["model_to_device"] == RECOGNIZER_CONFIG.device
    assert captured["manual_seed"] == RECOGNIZER_CONFIG.random_seed


def test_compute_log_probs_scopes_num_threads_to_phoneme_model_and_restores(monkeypatch):
    """_compute_log_probs はスレッド数を推論の直前だけ RECOGNIZER_CONFIG.num_threads へ設定し、
    呼び出し前の値へ復元する(§5.1)。内容認識(Whisper系)のCPU実行をこの制約に道連れにしない
    ための局所化(§5.2「決定論」)。"""
    torch = pytest.importorskip("torch")
    from vocal_analysis import RECOGNIZER_CONFIG
    from vocal_analysis.recognizer import _compute_log_probs

    calls = []
    monkeypatch.setattr(torch, "get_num_threads", lambda: 8)
    monkeypatch.setattr(torch, "set_num_threads", lambda n: calls.append(n))

    class _FakeOutputs:
        logits = torch.zeros(1, 2, 2)

    class _FakeModel:
        def __call__(self, input_values):
            # 推論の時点では num_threads へ設定済みで、まだ復元されていない。
            assert calls == [RECOGNIZER_CONFIG.num_threads]
            return _FakeOutputs()

    class _FakeInputValues:
        def to(self, device):
            return self

    class _FakeInputs:
        input_values = _FakeInputValues()

    class _FakeProcessor:
        def __call__(self, samples, sampling_rate, return_tensors):
            return _FakeInputs()

    _compute_log_probs(_FakeProcessor(), _FakeModel(), np.zeros(16000, dtype=np.float32))

    assert calls == [RECOGNIZER_CONFIG.num_threads, 8]  # 設定→復元の順


def test_load_content_recognizer_pipeline_uses_cpu_when_gpu_unavailable(monkeypatch):
    # transformers・torch が実際に導入されている環境でのみ検証する(最小環境では skip)。
    # transformers.pipeline・torch.cuda.is_available をモンキーパッチするためネットワーク・
    # 実モデルのダウンロード・実GPUは不要。
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.recognizer import _load_content_recognizer_pipeline

    captured = {}

    def fake_pipeline(task, model=None, revision=None, device=None, dtype=None, **kwargs):
        captured["task"] = task
        captured["model"] = model
        captured["revision"] = revision
        captured["device"] = device
        captured["dtype"] = dtype
        return object()

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # プロセス内キャッシュを空に差し替え、他テストのロード結果を再利用させない(必ずロードさせる)。
    monkeypatch.setattr(recognizer_module, "_content_recognizer_pipeline_cache", None)

    _load_content_recognizer_pipeline(DEFAULT_CONTENT_RECOGNIZER_MODEL)

    assert captured["task"] == "automatic-speech-recognition"
    # §5.2・§8.3: content_recognizer_model が指すモデル・revisionをそのままロードに渡す。
    assert captured["model"] == DEFAULT_CONTENT_RECOGNIZER_MODEL.model_id
    assert captured["revision"] == DEFAULT_CONTENT_RECOGNIZER_MODEL.model_revision
    # §5.1: GPU不在時はCPUを使う(強制アライメント用音素モデルとは独立の自動選択)。
    assert captured["device"] == "cpu"
    assert captured["dtype"] == torch.float32


def test_load_content_recognizer_pipeline_uses_gpu_when_available(monkeypatch):
    transformers = pytest.importorskip("transformers")
    torch = pytest.importorskip("torch")
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.recognizer import _load_content_recognizer_pipeline

    captured = {}

    def fake_pipeline(task, model=None, revision=None, device=None, dtype=None, **kwargs):
        captured["device"] = device
        captured["dtype"] = dtype
        return object()

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # プロセス内キャッシュを空に差し替え、他テストのロード結果を再利用させない(必ずロードさせる)。
    monkeypatch.setattr(recognizer_module, "_content_recognizer_pipeline_cache", None)

    _load_content_recognizer_pipeline(DEFAULT_CONTENT_RECOGNIZER_MODEL)

    # §5.1: GPUが利用可能なら自動的にGPUを使う(強制アライメント用音素モデルはCPU固定のまま)。
    assert captured["device"] == "cuda"
    # GPU実行時はfp16でロードし、重み・アクティベーションのメモリ使用量を半減させる。
    assert captured["dtype"] == torch.float16


def test_load_content_recognizer_pipeline_evicts_previous_model_before_loading_next(monkeypatch):
    """異なるモデルへ切り替える際、新モデルのロードを始める時点で直前のパイプラインが実際に
    解放可能(参照を一切保持していない)になっている(グローバル変数がNoneというだけでなく、
    関数内のローカル変数に旧パイプラインへの強参照が残っていないことを、弱参照で確認する)。"""
    transformers = pytest.importorskip("transformers")
    import gc
    import weakref

    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.recognizer import _load_content_recognizer_pipeline

    model_a = ContentRecognizerModel(model_id="org/model-a")
    model_b = ContentRecognizerModel(model_id="org/model-b")
    observed = {}

    class _Pipeline:
        """object() は弱参照を作れないため、素の object の代わりに使う。"""

    def fake_pipeline(task, model=None, revision=None, device=None, **kwargs):
        if model == model_b.model_id:
            gc.collect()
            observed["model_a_pipeline_alive"] = observed["model_a_pipeline_ref"]() is not None
        return _Pipeline()

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)
    monkeypatch.setattr(recognizer_module, "_select_content_recognizer_device", lambda: "cpu")
    monkeypatch.setattr(recognizer_module, "_content_recognizer_pipeline_cache", None)

    model_a_pipeline = _load_content_recognizer_pipeline(model_a)
    observed["model_a_pipeline_ref"] = weakref.ref(model_a_pipeline)
    del model_a_pipeline

    _load_content_recognizer_pipeline(model_b)

    assert observed["model_a_pipeline_alive"] is False


def test_transcribe_segment_extracts_word_timestamps_from_chunks(monkeypatch):
    """パイプラインが chunks(単語ごとのテキスト・タイムスタンプ)を返す場合、_transcribe_segment は
    それを単調化した単語タイムスタンプ列として書き起こしテキストと共に返す(§5.2手順3)。"""
    from vocal_analysis import recognizer as recognizer_module

    class _FakePromptIds:
        def to(self, device):
            return "PROMPT_IDS"

    class _FakePipeline:
        device = "cpu"

        class tokenizer:
            @staticmethod
            def get_prompt_ids(prompt, return_tensors):
                return _FakePromptIds()

        def __call__(self, samples, return_timestamps, generate_kwargs):
            return {
                "text": "あ い",
                "chunks": [
                    {"text": "あ", "timestamp": (0.0, 0.5)},
                    {"text": "い", "timestamp": (0.5, 1.0)},
                ],
            }

    samples = np.zeros(16000, dtype=np.float32)  # 1.0秒@16kHz
    text, words = recognizer_module._transcribe_segment(_FakePipeline(), samples)

    assert text == "あ い"
    assert words == [("あ", 0.0, 0.5), ("い", 0.5, 1.0)]


def test_transcribe_segment_returns_none_words_when_pipeline_has_no_chunks(monkeypatch):
    """パイプラインが chunks を返さない(単語タイムスタンプ非対応の)場合、_transcribe_segment は
    words に None を返す(§5.2手順7のフォールバックに帰着)。"""
    from vocal_analysis import recognizer as recognizer_module

    class _FakePromptIds:
        def to(self, device):
            return "PROMPT_IDS"

    class _FakePipeline:
        device = "cpu"

        class tokenizer:
            @staticmethod
            def get_prompt_ids(prompt, return_tensors):
                return _FakePromptIds()

        def __call__(self, samples, return_timestamps, generate_kwargs):
            return {"text": "あ"}  # chunksキー自体が無い

    text, words = recognizer_module._transcribe_segment(
        _FakePipeline(), np.zeros(16000, dtype=np.float32)
    )

    assert text == "あ"
    assert words is None


def test_transcribe_segment_bounds_generation_length(monkeypatch):
    """_transcribe_segment は反復ハルシネーション時の生成時間を有界化するため、生成トークン数の
    上限をgenerate_kwargsへ指定してパイプラインへ渡す(§5.2手順3)。"""
    from vocal_analysis import recognizer as recognizer_module

    class _FakePromptIds:
        def to(self, device):
            return "PROMPT_IDS"

    captured = {}

    class _FakePipeline:
        device = "cpu"

        class tokenizer:
            @staticmethod
            def get_prompt_ids(prompt, return_tensors):
                return _FakePromptIds()

        def __call__(self, samples, return_timestamps, generate_kwargs):
            captured["generate_kwargs"] = generate_kwargs
            return {"text": "あ"}

    recognizer_module._transcribe_segment(_FakePipeline(), np.zeros(16000, dtype=np.float32))

    assert captured["generate_kwargs"]["max_new_tokens"] == 380


def test_transcribe_text_only_bounds_generation_length(monkeypatch):
    """_transcribe_text_only(トリガ式リトライの再認識用)も同様に生成トークン数の上限を
    generate_kwargsへ指定する(§5.2手順3)。"""
    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module

    captured = {}

    class _FakePipeline:
        def __call__(self, samples, generate_kwargs):
            captured["generate_kwargs"] = generate_kwargs
            return {"text": "あ"}

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline", lambda model: _FakePipeline()
    )

    recognizer_module._transcribe_text_only(
        np.zeros(16000, dtype=np.float32), ContentRecognizerModel(model_id="org/model")
    )

    assert captured["generate_kwargs"]["max_new_tokens"] == 420


def test_recognize_default_content_recognizer_model_loads_pinned_pipeline(tmp_path, monkeypatch):
    """既定値(DEFAULT_CONTENT_RECOGNIZER_MODEL)で呼び出すと、そのmodel_id・revisionで
    パイプラインをロードし、かな限定プロンプト(KANA_PROMPT)をprompt_idsとして渡す(§5.2)。"""
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL, KANA_PROMPT
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )
    calls = {}

    class _FakePromptIds:
        def to(self, device):
            calls["prompt_ids_device"] = device
            return "PROMPT_IDS"

    class _FakePipeline:
        device = "cpu"

        class tokenizer:
            @staticmethod
            def get_prompt_ids(prompt, return_tensors):
                calls["prompt_text"] = prompt
                return _FakePromptIds()

        def __call__(self, samples, return_timestamps, generate_kwargs):
            calls["generate_kwargs"] = generate_kwargs
            calls["return_timestamps"] = return_timestamps
            return {"text": "あ"}

    def fake_load_pipeline(content_recognizer_model):
        calls["loaded_model"] = content_recognizer_model
        return _FakePipeline()

    monkeypatch.setattr(recognizer_module, "_load_content_recognizer_pipeline", fake_load_pipeline)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    segments = recognizer_module.recognize(wav_path)

    assert calls["loaded_model"] == DEFAULT_CONTENT_RECOGNIZER_MODEL
    assert calls["prompt_text"] == KANA_PROMPT
    assert calls["generate_kwargs"]["prompt_ids"] == "PROMPT_IDS"
    assert calls["return_timestamps"] == "word"  # §5.2手順3: 単語タイムスタンプ付きで呼び出す
    assert segments[1].phoneme == "a"


def test_recognize_custom_content_recognizer_model_is_passed_through(tmp_path, monkeypatch):
    """content_recognizer_model に既定値以外(例: 候補値 KANA_WHISPER_MODEL)を渡すと、
    そのモデルでパイプラインをロードする(モデルによる分岐は無い。§5.2)。"""
    from vocal_analysis import KANA_WHISPER_MODEL
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )
    calls = {}

    class _FakePromptIds:
        def to(self, device):
            return "PROMPT_IDS"

    class _FakePipeline:
        device = "cpu"

        class tokenizer:
            @staticmethod
            def get_prompt_ids(prompt, return_tensors):
                return _FakePromptIds()

        def __call__(self, samples, return_timestamps, generate_kwargs):
            return {"text": "あ"}

    def fake_load_pipeline(content_recognizer_model):
        calls["loaded_model"] = content_recognizer_model
        return _FakePipeline()

    monkeypatch.setattr(recognizer_module, "_load_content_recognizer_pipeline", fake_load_pipeline)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    recognizer_module.recognize(wav_path, content_recognizer_model=KANA_WHISPER_MODEL)

    assert calls["loaded_model"] == KANA_WHISPER_MODEL


def test_recognize_english_oov_katakana_method_is_passed_through_to_g2p(tmp_path, monkeypatch):
    """english_oov_katakana_method に既定値(arpakana)以外を渡すと、_g2p(ひいてはconvert_oov_words)
    へその方式が渡る(製品経路recognize()からTinyLlama方式を実際に選択できることの検証)。"""
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(1920), 16000)

    decoder = {0: "<pad>", 1: "a"}
    log_probs = np.array(
        [[5.0, -5.0], [5.0, -5.0], [-5.0, 5.0], [-5.0, 5.0], [5.0, -5.0], [5.0, -5.0]]
    )
    g2p_calls = []

    def fake_g2p(text, method=None):
        g2p_calls.append(method)
        return {"あ": ["a"]}[text]

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", fake_g2p)
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    recognizer_module.recognize(
        wav_path, english_oov_katakana_method="tinyllama-katakana-converter")

    # _resolve_transcription内の音素密度判定(_text_phoneme_density)も含め、_g2pへの全呼び出しが
    # 同じ指定方式を受け取る(呼び出し回数自体は密度判定の有無に依存するため、回数を1件に固定しない)。
    assert len(g2p_calls) >= 1
    assert all(m == "tinyllama-katakana-converter" for m in g2p_calls)


def test_recognize_english_oov_katakana_method_reaches_per_word_g2p(tmp_path, monkeypatch):
    """単語タイムスタンプが取得できた場合の単語ごとのG2P分岐にも、english_oov_katakana_methodが
    渡る(全文G2P分岐だけでなく単語別G2P分岐も配線されていることの検証)。
    """
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(2240), 16000)  # 0.14秒・7フレーム

    decoder = {0: "<pad>", 1: "a", 2: "i"}
    log_probs = np.array(
        [
            [5.0, -5.0, -5.0], [-5.0, 5.0, -5.0], [-5.0, 5.0, -5.0], [5.0, -5.0, -5.0],
            [-5.0, -5.0, 5.0], [-5.0, -5.0, 5.0], [5.0, -5.0, -5.0],
        ]
    )
    g2p_calls = []

    def fake_g2p(text, method=None):
        g2p_calls.append((text, method))
        return {"あ": ["a"], "い": ["i"], "あ い": ["a", "i"]}[text]

    def fake_transcribe(pipeline, samples):
        return "あ い", [("あ", 0.02, 0.04), ("い", 0.06, 0.08)]

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", fake_g2p)
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    recognizer_module.recognize(
        wav_path, english_oov_katakana_method="tinyllama-katakana-converter")

    # 単語ごとの呼び出し("あ"・"い")が実際に発生し、いずれも指定方式を受け取っている。
    per_word_calls = [c for c in g2p_calls if c[0] in ("あ", "い")]
    assert len(per_word_calls) == 2
    assert all(method == "tinyllama-katakana-converter" for _, method in per_word_calls)


def test_recognize_content_recognizer_model_fetch_failure_names_that_model(tmp_path, monkeypatch):
    """内容認識モデルの取得失敗時、エラーメッセージに指定したcontent_recognizer_modelの
    model_id・revisionが含まれる(既定値以外を渡した場合でも正しく名指しする)。"""
    from vocal_analysis import ContentRecognizerModel
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)
    custom_model = ContentRecognizerModel(model_id="openai/whisper-large-v3", model_revision="deadbeef")

    original_error = OSError("model not found in cache and offline")

    def fail_load_pipeline(content_recognizer_model):
        raise original_error

    monkeypatch.setattr(recognizer_module, "_load_content_recognizer_pipeline", fail_load_pipeline)

    with pytest.raises(recognizer_module.RecognitionError, match="openai/whisper-large-v3") as excinfo:
        recognizer_module.recognize(wav_path, content_recognizer_model=custom_model)

    assert "deadbeef" in str(excinfo.value)
    assert excinfo.value.__cause__ is original_error


def _make_sofa_config(tmp_path):
    from vocal_analysis import SofaAlignerConfig

    return SofaAlignerConfig(
        sofa_python=tmp_path / "sofa-venv" / "python",
        sofa_root=tmp_path / "SOFA",
        checkpoint_path=tmp_path / "checkpoint.ckpt",
    )


def test_recognize_sofa_without_config_raises_recognition_error(tmp_path):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(16000), 16000)

    with pytest.raises(recognizer_module.RecognitionError):
        recognizer_module.recognize(wav_path, forced_aligner="sofa-forcedalign", sofa_aligner=None)


def test_recognize_unknown_forced_aligner_raises_recognition_error(tmp_path):
    from vocal_analysis import recognizer as recognizer_module

    # ForcedAlignerId(Literal)の静的検査をすり抜けた未知の値(例: 誤字)を渡しても、黙って
    # SOFA経路へ落ちたりwav2vec2経路を使ったりせず、明示的にRecognitionErrorにする。関数入口
    # (音声読み込みより前)で検証するため、音声を無音・全区間gap確定にしても検出できる
    # (実在しないパスでも到達前に失敗することを、あえてダミーの存在しないパスで確認する)。
    with pytest.raises(recognizer_module.RecognitionError):
        recognizer_module.recognize(tmp_path / "does_not_exist.wav", forced_aligner="unknown-aligner")


def test_recognize_default_forced_aligner_does_not_call_sofa(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module
    from vocal_analysis.config import DEFAULT_FORCED_ALIGNER

    assert DEFAULT_FORCED_ALIGNER == "wav2vec2-ctc-forcedalign"

    wav_path = _write_wav(tmp_path / "vocal.wav", np.zeros((16000, 1), dtype=np.float32), 16000)

    def fail_if_called(targets, config):
        raise AssertionError("既定のforced_alignerでSOFAを呼び出してはならない")

    monkeypatch.setattr(recognizer_module.sofa_align, "_align_batch", fail_if_called)

    segments = recognizer_module.recognize(wav_path)

    assert len(segments) == 1
    assert segments[0].type == "gap"


def test_recognize_sofa_path_splits_words_and_reassembles_segments(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 0.4秒・全区間有声(トリムなし)。単語2件「か」[0.1,0.2)・「き」[0.25,0.35)を内容認識が
    # 返す(モック)。有効な単語列確定(cursorクランプ)・gap確定・SOFAへの一括バッチ呼び出し・
    # Segment契約検証・IPA写像・絶対時刻への復元、のすべてを実コードで通す(SOFAサブプロセス
    # 自体だけをモックする)。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(6400), 16000)
    config = _make_sofa_config(tmp_path)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment",
        lambda pipeline, samples: ("かき", [("か", 0.1, 0.2), ("き", 0.25, 0.35)]),
    )
    # 全文("かき"。トリガ判定の密度算出に使う)と単語ごとの両方の呼び出しに応える。
    monkeypatch.setattr(
        recognizer_module, "_g2p",
        lambda text, method=None: {"か": ["k", "a"], "き": ["k", "i"], "かき": ["k", "a", "k", "i"]}[text]
    )

    align_batch_calls = []

    def fake_align_batch(targets, sofa_aligner_config):
        align_batch_calls.append((targets, sofa_aligner_config))
        return {
            "segment_0000": [(0.0, 0.05, "k"), (0.05, 0.1, "a")],
            "segment_0001": [(0.0, 0.04, "k"), (0.04, 0.1, "i")],
        }

    monkeypatch.setattr(recognizer_module.sofa_align, "_align_batch", fake_align_batch)

    segments = recognizer_module.recognize(
        wav_path, forced_aligner="sofa-forcedalign", sofa_aligner=config
    )

    # SOFAへは1回のrecognize()呼び出しにつき1回だけ、2単語分をまとめて渡す(バッチ単位。§5.3)。
    assert len(align_batch_calls) == 1
    targets, passed_config = align_batch_calls[0]
    assert passed_config is config
    assert len(targets) == 2
    assert targets[0][2] == ["k", "a"]
    assert targets[1][2] == ["k", "i"]

    expected = [
        ("gap", None, 0.0, 0.1),
        ("consonant", "k", 0.1, 0.15),
        ("vowel", "a", 0.15, 0.2),
        ("gap", None, 0.2, 0.25),
        ("consonant", "k", 0.25, 0.29),
        ("vowel", "i", 0.29, 0.35),
        ("gap", None, 0.35, 0.4),
    ]
    assert len(segments) == len(expected)
    for seg, (exp_type, exp_phoneme, exp_start, exp_end) in zip(segments, expected):
        assert seg.type == exp_type
        assert seg.phoneme == exp_phoneme
        assert seg.start_sec == pytest.approx(exp_start)
        assert seg.end_sec == pytest.approx(exp_end)


def test_recognize_sofa_path_uses_whole_region_when_no_word_timestamps(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 単語タイムスタンプが取得できない場合(§5.2手順3のフォールバック対象)は、区間全体を
    # 1つのSOFA対象とする(単語単位分割をしない)。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(3200), 16000)
    config = _make_sofa_config(tmp_path)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])

    align_batch_calls = []

    def fake_align_batch(targets, sofa_aligner_config):
        align_batch_calls.append(targets)
        return {"segment_0000": [(0.0, 0.2, "a")]}

    monkeypatch.setattr(recognizer_module.sofa_align, "_align_batch", fake_align_batch)

    segments = recognizer_module.recognize(
        wav_path, forced_aligner="sofa-forcedalign", sofa_aligner=config
    )

    assert len(align_batch_calls) == 1
    assert len(align_batch_calls[0]) == 1
    assert align_batch_calls[0][0][2] == ["a"]

    assert len(segments) == 1
    assert segments[0].type == "vowel"
    assert segments[0].phoneme == "a"
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(0.2)


def test_recognize_sofa_path_never_loads_wav2vec2_model(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(3200), 16000)
    config = _make_sofa_config(tmp_path)

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: object(),
    )
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda pipeline, samples: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text, method=None: {"あ": ["a"]}[text])
    monkeypatch.setattr(recognizer_module.sofa_align, "_align_batch", lambda targets, cfg: {"segment_0000": [(0.0, 0.2, "a")]})

    def fail_if_called():
        raise AssertionError("SOFA経路でwav2vec2の音素モデルをロードしてはならない")

    monkeypatch.setattr(recognizer_module, "_load_model_and_processor", fail_if_called)

    recognizer_module.recognize(wav_path, forced_aligner="sofa-forcedalign", sofa_aligner=config)


def test_recognize_sofa_path_silent_input_does_not_call_align_batch_with_targets(tmp_path, monkeypatch):
    from vocal_analysis import recognizer as recognizer_module

    # 音声全体が無音の場合、SOFA対象は1件も積まれない。sofa_align._align_batch自体は
    # (targets=[]で)呼ばれてもよいが、空リストならサブプロセスを起動しないのはsofa_align側の
    # 責務であり、ここではrecognize()が空targetsで呼ぶことだけを確認する。
    wav_path = _write_wav(tmp_path / "vocal.wav", np.zeros((16000, 1), dtype=np.float32), 16000)
    config = _make_sofa_config(tmp_path)

    calls = []

    def fake_align_batch(targets, cfg):
        calls.append(targets)
        return {}

    monkeypatch.setattr(recognizer_module.sofa_align, "_align_batch", fake_align_batch)

    segments = recognizer_module.recognize(wav_path, forced_aligner="sofa-forcedalign", sofa_aligner=config)

    assert len(calls) == 1
    assert calls[0] == []
    assert len(segments) == 1
    assert segments[0].type == "gap"
