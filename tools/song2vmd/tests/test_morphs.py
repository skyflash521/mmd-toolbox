"""song2vmd のモーフ生成・VMD組み立てのテスト(song2vmd.md §6.5・9章)。

口形イベント列(MouthEvent)を共有モジュール lipsync へ渡してモーフキーを生成し、モーフキーのみの
VmdDocument を組み立てる morphs.build_vmd_document を検証する。共有モーフ生成コア(lipsync)自体の
正しさ・回帰は lipsync 仕様 §7 の既知値フィクスチャが担保するため、ここでは song2vmd 側の接続
(lipsync 呼び出し・VmdDocument 組み立て・0F中立キー・正規化・cp932エンコード)だけを検証する。
"""

import pytest

from lipsync import ConsonantClass, GenerationParams, MouthEvent, MouthShape
from vmd import MorphKey, io

from song2vmd import morphs


def ev(shape, start, end, open_amount=0.0, consonant_class=ConsonantClass.NONE):
    return MouthEvent(shape=shape, start=start, end=end, open_amount=open_amount,
                       consonant_class=consonant_class)


def test_build_vmd_document_calls_lipsync_generate_morph_keys(monkeypatch):
    """lipsyncへの接続を、実際の母音合成結果に依存せず直接検証する(song2vmd.md 6.5)。

    generate_morph_keys を差し替え、同一の events・params が渡ること、返した MorphKey が
    そのまま VmdDocument.morph へ入ることを確認する(共有モーフ生成コア自体の正しさは対象外)。
    """
    events = [ev(MouthShape.A, 0, 10, open_amount=0.5)]
    params = GenerationParams()
    captured = {}
    fake_keys = [MorphKey("Fake".encode("cp932").ljust(15, b"\x00"), 3, 0.42)]

    def fake_generate_morph_keys(passed_events, passed_params):
        captured["events"] = passed_events
        captured["params"] = passed_params
        return fake_keys

    monkeypatch.setattr(morphs, "generate_morph_keys", fake_generate_morph_keys)
    document = morphs.build_vmd_document(events, params, model_name="")
    assert captured["events"] == events
    assert captured["params"] is params
    assert any(key.name == "Fake" and key.frame == 3 and key.weight == pytest.approx(0.42)
               for key in document.morph)


def test_build_vmd_document_contains_generated_morph_keys():
    events = [
        ev(MouthShape.SILENCE, 0, 5),
        ev(MouthShape.A, 5, 20, open_amount=0.6),
        ev(MouthShape.SILENCE, 20, 25),
    ]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="")
    names = {key.name for key in document.morph}
    assert "あ" in names


def test_model_name_is_cp932_encoded_and_padded_to_20_bytes():
    events = [ev(MouthShape.A, 0, 10, open_amount=0.5)]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="Model")
    assert len(document.model_name_raw) == 20
    assert document.model_name == "Model"
    assert document.model_name_raw == b"Model".ljust(20, b"\x00")


def test_empty_model_name_produces_all_zero_padding():
    events = [ev(MouthShape.A, 0, 10, open_amount=0.5)]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="")
    assert document.model_name_raw == b"\x00" * 20


def test_frame0_neutral_key_exists_for_every_used_morph():
    # 最初の実キーがフレーム0より後にある場合でも、使用モーフには0Fの中立キーが補われる
    # (vmd.ensure_frame0_neutral_keys。song2vmd.md 9章)。
    events = [
        ev(MouthShape.SILENCE, 0, 10),
        ev(MouthShape.A, 10, 30, open_amount=0.6),
    ]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="")
    names_used = {key.name for key in document.morph}
    for name in names_used:
        frames = [key.frame for key in document.morph if key.name == name]
        assert 0 in frames


def test_morph_keys_are_sorted_and_deduplicated_per_name():
    events = [
        ev(MouthShape.A, 0, 15, open_amount=0.5),
        ev(MouthShape.I, 15, 30, open_amount=0.5),
        ev(MouthShape.A, 30, 45, open_amount=0.7),
    ]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="")
    by_name = {}
    for key in document.morph:
        by_name.setdefault(key.name, []).append(key.frame)
    for name, frames in by_name.items():
        assert frames == sorted(frames)
        assert len(frames) == len(set(frames))


def test_only_morph_section_is_populated():
    events = [ev(MouthShape.A, 0, 10, open_amount=0.5)]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="")
    assert document.bone == []
    assert document.camera == []
    assert document.light == []
    assert document.self_shadow == []
    assert document.ik_property == []


def test_empty_events_produce_empty_morph_document():
    document = morphs.build_vmd_document([], GenerationParams(), model_name="")
    assert document.morph == []


def test_round_trip_write_and_read(tmp_path):
    events = [
        ev(MouthShape.SILENCE, 0, 5),
        ev(MouthShape.A, 5, 20, open_amount=0.6),
        ev(MouthShape.N, 20, 26, open_amount=0.3),
        ev(MouthShape.SILENCE, 26, 30),
    ]
    document = morphs.build_vmd_document(events, GenerationParams(), model_name="テスト")
    out = tmp_path / "out.vmd"
    io.write_file(document, str(out))
    read_back, warnings = io.read(str(out))
    assert warnings == []
    assert read_back.model_name == "テスト"
    # weight はファイル格納が float32 なので、往復で float64 と厳密一致しないことがある(丸め誤差)。
    read_morphs = sorted((key.name, key.frame) for key in read_back.morph)
    expected_names_frames = sorted((key.name, key.frame) for key in document.morph)
    assert read_morphs == expected_names_frames
    expected_weight = {(key.name, key.frame): key.weight for key in document.morph}
    for key in read_back.morph:
        assert key.weight == pytest.approx(expected_weight[(key.name, key.frame)], abs=1e-6)
