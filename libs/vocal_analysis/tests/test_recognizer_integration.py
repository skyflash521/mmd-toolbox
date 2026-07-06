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

    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda samples, content_recognizer_model: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: {"あ": ["a"]}[text])
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

    def fake_transcribe(samples, content_recognizer_model):
        call_count["transcribe"] += 1
        return "あ", None

    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: {"あ": ["a"]}[text])
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

    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda samples, content_recognizer_model: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: {"あ": ["a"]}[text])
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
    # 返すケース(内容認識の反復幻覚を模す)。誤った音素列で強制アライメントを試みず、区間全体を
    # gapとして確定し、音素モデルも一度もロードしない。
    wav_path = _write_wav(tmp_path / "vocal.wav", _loud_samples(32000), 16000)  # 2.0秒・無音区間なし

    monkeypatch.setattr(
        recognizer_module, "_transcribe_segment",
        lambda samples, content_recognizer_model: ("かんじは つかわないでください。" * 20, None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: ["a"] * 100)

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
        recognizer_module, "_transcribe_segment",
        lambda samples, content_recognizer_model: ("  ", None))

    def fail_g2p(text):
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

    def fake_transcribe(samples, content_recognizer_model):
        return "あ", [("あ", 0.02, 0.06)]

    def fake_forced_align_windowed(log_probs_arg, token_ids, windows_sec):
        captured["windows_sec"] = windows_sec
        # 実際の経路組み立ては既存のバンド制限Viterbiへ委譲する(窓の受け渡しだけを検証する)。
        return recognizer_module._forced_align(log_probs_arg, token_ids)

    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: {"あ": ["a"]}[text])
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
    # seq=["pau","a","pau"]。aは最小滞在で4サブ状態へ展開されるため、窓は
    # [先頭pau, a, a, a, a, 末尾pau] の6個(log_probsの6フレームに対応)。
    windows_sec = captured["windows_sec"]
    assert windows_sec[0] == pytest.approx((0.0, 0.02 + margin))
    assert all(w == pytest.approx((0.02 - margin, 0.06 + margin)) for w in windows_sec[1:5])
    assert windows_sec[5] == pytest.approx((0.06 - margin, 0.12))
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

    def fake_transcribe(samples, content_recognizer_model):
        return "あ", [("あ", 0.02, 0.06)]

    def fail_windowed(log_probs_arg, token_ids, windows_sec):
        raise recognizer_module.RecognitionError("単語窓制約下で強制アライメントが末尾トークンへ到達できませんでした")

    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: {"あ": ["a"]}[text])
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

    def fake_transcribe(samples, content_recognizer_model):
        return "あ い", [("あ", 0.02, 0.04), ("い", 0.06, 0.08)]

    monkeypatch.setattr(recognizer_module, "_transcribe_segment", fake_transcribe)
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: {"あ": ["a"], "い": ["i"]}[text])
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
        recognizer_module, "_transcribe_segment",
        lambda samples, content_recognizer_model: ("あ い", [("あ", 0.1, 0.4), ("い", 0.5, 0.9)]))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: ["a"] * 15)

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
        recognizer_module, "_transcribe_segment", lambda samples, content_recognizer_model: ("あ", []))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: {"あ": ["a"]}[text])
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

    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda samples, content_recognizer_model: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: {"あ": ["a"]}[text])
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

    def fake_transcribe(samples, content_recognizer_model):
        raise original_error

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

    def fake_transcribe(samples, content_recognizer_model):
        raise original_error

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
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda samples, content_recognizer_model: ("あ", None))

    def fake_g2p(text):
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
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda samples, content_recognizer_model: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: ["a"])

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
    monkeypatch.setattr(recognizer_module, "_transcribe_segment", lambda samples, content_recognizer_model: ("あ", None))
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: ["a"])

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


def test_load_content_recognizer_pipeline_passes_pinned_config(monkeypatch):
    # transformers が実際に導入されている環境でのみ、_load_content_recognizer_pipeline の実体を
    # 検証する(最小環境では skip)。transformers.pipeline 自体をモンキーパッチするためネットワーク・
    # 実モデルのダウンロードは発生しない。
    transformers = pytest.importorskip("transformers")
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL, RECOGNIZER_CONFIG
    from vocal_analysis.recognizer import _load_content_recognizer_pipeline

    captured = {}

    def fake_pipeline(task, model=None, revision=None, device=None, **kwargs):
        captured["task"] = task
        captured["model"] = model
        captured["revision"] = revision
        captured["device"] = device
        return object()

    monkeypatch.setattr(transformers, "pipeline", fake_pipeline)

    _load_content_recognizer_pipeline(DEFAULT_CONTENT_RECOGNIZER_MODEL)

    assert captured["task"] == "automatic-speech-recognition"
    # §5.2・§8.3: content_recognizer_model が指すモデル・revisionをそのままロードに渡す。
    assert captured["model"] == DEFAULT_CONTENT_RECOGNIZER_MODEL.model_id
    assert captured["revision"] == DEFAULT_CONTENT_RECOGNIZER_MODEL.model_revision
    # §5.1: 実行デバイスを固定条件どおりに適用する。
    assert captured["device"] == RECOGNIZER_CONFIG.device


def test_transcribe_segment_extracts_word_timestamps_from_chunks(monkeypatch):
    """パイプラインが chunks(単語ごとのテキスト・タイムスタンプ)を返す場合、_transcribe_segment は
    それを単調化した単語タイムスタンプ列として書き起こしテキストと共に返す(§5.2手順3)。"""
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL
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

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: _FakePipeline(),
    )

    samples = np.zeros(16000, dtype=np.float32)  # 1.0秒@16kHz
    text, words = recognizer_module._transcribe_segment(samples, DEFAULT_CONTENT_RECOGNIZER_MODEL)

    assert text == "あ い"
    assert words == [("あ", 0.0, 0.5), ("い", 0.5, 1.0)]


def test_transcribe_segment_returns_none_words_when_pipeline_has_no_chunks(monkeypatch):
    """パイプラインが chunks を返さない(単語タイムスタンプ非対応の)場合、_transcribe_segment は
    words に None を返す(§5.2手順7のフォールバックに帰着)。"""
    from vocal_analysis import DEFAULT_CONTENT_RECOGNIZER_MODEL
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

    monkeypatch.setattr(
        recognizer_module, "_load_content_recognizer_pipeline",
        lambda content_recognizer_model: _FakePipeline(),
    )

    text, words = recognizer_module._transcribe_segment(
        np.zeros(16000, dtype=np.float32), DEFAULT_CONTENT_RECOGNIZER_MODEL
    )

    assert text == "あ"
    assert words is None


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
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: {"あ": ["a"]}[text])
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
    monkeypatch.setattr(recognizer_module, "_g2p", lambda text: {"あ": ["a"]}[text])
    monkeypatch.setattr(
        recognizer_module, "_load_model_and_processor", lambda: (_FakeProcessor(decoder), object())
    )
    monkeypatch.setattr(
        recognizer_module, "_compute_log_probs", lambda processor, model, samples: log_probs
    )

    recognizer_module.recognize(wav_path, content_recognizer_model=KANA_WHISPER_MODEL)

    assert calls["loaded_model"] == KANA_WHISPER_MODEL


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
