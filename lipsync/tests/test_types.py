"""雛形・データ型と mmd_toolbox 連携のテスト(lipsync.md §3/§4)。

公開データ型(MouthShape / MouthEvent / GenerationParams)とコア関数
generate_morph_keys の契約、および出力モーフキーが mmd_toolbox.vmd の
I/O をラウンドトリップすることを検証する。
"""

import pytest

import lipsync
from mmd_toolbox.vmd import MorphKey, VmdDocument, read, write


def test_mouth_shape_members():
    """MouthShape は母音5種＋両唇閉鎖＋無音を持つ。"""
    names = {m.name for m in lipsync.MouthShape}
    assert names == {"A", "I", "U", "E", "O", "BILABIAL", "SILENCE"}


def test_mouth_event_defaults():
    """MouthEvent は shape/start/end を取り、open_amount 既定 0.0。"""
    ev = lipsync.MouthEvent(shape=lipsync.MouthShape.A, start=0.0, end=10.0)
    assert ev.shape is lipsync.MouthShape.A
    assert ev.start == 0.0
    assert ev.end == 10.0
    assert ev.open_amount == 0.0


def test_generation_params_defaults():
    """GenerationParams の既定値が初期目安と一致する。"""
    p = lipsync.GenerationParams()
    assert p.open_cap == 0.8
    assert p.vowel_scale == (1.0, 1.0, 1.0, 1.0, 1.0)
    assert p.attack_frames == 2
    assert p.release_frames == 2
    assert p.min_hold_frames == 3
    assert p.coartic_overlap_max == 2
    assert p.anticipation_frames == 1
    assert p.exaggeration == 1.0
    assert p.vibrato_threshold == 18
    assert p.vibrato_amp == 0.05
    assert p.vibrato_period == 15


STANDARD_MORPHS = {"あ", "い", "う", "え", "お"}


def test_generate_morph_keys_output_contract():
    """generate_morph_keys は MouthEvent 列と GenerationParams を受け取り、
    出力契約を満たす MorphKey 列を返す: 母音イベントに対し非空、
    各キーは cp932・15バイト固定の name_raw・標準5モーフ名・整数 frame で、
    時間順に並ぶ。"""
    events = [
        lipsync.MouthEvent(lipsync.MouthShape.A, 0.0, 10.0, 0.5),
        lipsync.MouthEvent(lipsync.MouthShape.I, 10.0, 20.0, 0.5),
    ]
    keys = lipsync.generate_morph_keys(events, lipsync.GenerationParams())
    assert isinstance(keys, list)
    assert keys, "母音イベントに対しモーフキーが生成されるべき"
    for k in keys:
        assert isinstance(k, MorphKey)
        assert len(k.name_raw) == 15
        assert k.name in STANDARD_MORPHS
        assert isinstance(k.frame, int)
    frames = [k.frame for k in keys]
    assert frames == sorted(frames)


def test_generated_keys_roundtrip_through_vmd():
    """generate_morph_keys の出力を VmdDocument に組み立て、mmd_toolbox.vmd の
    write/read でラウンドトリップしてもモーフ名・フレーム・ウェイトが保たれる
    (受入条件「VMD出力がラウンドトリップする」)。"""
    events = [
        lipsync.MouthEvent(lipsync.MouthShape.A, 0.0, 10.0, 0.5),
        lipsync.MouthEvent(lipsync.MouthShape.I, 10.0, 20.0, 0.5),
    ]
    keys = lipsync.generate_morph_keys(events, lipsync.GenerationParams())
    assert keys
    restored, _warnings = read(write(VmdDocument(morph=keys)))
    assert len(restored.morph) == len(keys)
    for src, dst in zip(keys, restored.morph):
        assert dst.name == src.name
        assert dst.frame == src.frame
        assert dst.weight == pytest.approx(src.weight)
