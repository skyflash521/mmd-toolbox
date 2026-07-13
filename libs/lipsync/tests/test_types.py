"""雛形・データ型と vmd 連携のテスト。

公開データ型(MouthShape / MouthEvent / GenerationParams)とコア関数
generate_morph_keys の契約、および出力モーフキーが vmd の
I/O をラウンドトリップすることを検証する。
"""

import pytest

import lipsync
from vmd import MorphKey, VmdDocument, read, write


def test_mouth_shape_members():
    """MouthShape は母音5種＋撥音「ん」(N、列挙値 "n")＋両唇閉鎖＋無音＋レガート間隙を持つ。

    「ん」は閉口でなく母音と同じ機構を通る母音的口形。LEGATO_GAP は SILENCE と同じく非発音だが、
    生成時は完全閉口でなく谷として描く。
    """
    names = {m.name for m in lipsync.MouthShape}
    assert names == {"A", "I", "U", "E", "O", "N", "BILABIAL", "SILENCE", "LEGATO_GAP"}
    assert lipsync.MouthShape.N.value == "n"
    assert lipsync.MouthShape.LEGATO_GAP.value == "legato_gap"


def test_consonant_class_members():
    """ConsonantClass は NONE/NEUTRAL/ROUNDED/SPREAD を持つ。

    NONE=子音なし、NEUTRAL=唇を動かさない子音、ROUNDED=唇を丸める子音、SPREAD=い 方向へ寄せる子音。
    両唇閉鎖は MouthShape.BILABIAL で表しここには含めない。
    """
    names = {c.name for c in lipsync.ConsonantClass}
    assert names == {"NONE", "NEUTRAL", "ROUNDED", "SPREAD"}


@pytest.mark.xfail(reason="impl pending: ApertureClass", strict=True)
def test_aperture_class_members():
    """ApertureClass は NONE/FIRM_CLOSURE/NARROW_CHANNEL/SLIGHT_CLOSURE を持つ。

    ConsonantClass(唇の丸め・横引き方向)とは独立な軸で、母音合成の各モーフ最終重みを一律に
    減衰させる。両唇閉鎖は MouthShape.BILABIAL で表しここには含めない。
    """
    names = {c.name for c in lipsync.ApertureClass}
    assert names == {"NONE", "FIRM_CLOSURE", "NARROW_CHANNEL", "SLIGHT_CLOSURE"}


def test_mouth_event_defaults():
    """MouthEvent は shape/start/end を取り、open_amount 既定 0.0・consonant_class 既定 NONE。"""
    ev = lipsync.MouthEvent(shape=lipsync.MouthShape.A, start=0.0, end=10.0)
    assert ev.shape is lipsync.MouthShape.A
    assert ev.start == 0.0
    assert ev.end == 10.0
    assert ev.open_amount == 0.0
    assert ev.consonant_class is lipsync.ConsonantClass.NONE


def test_mouth_event_consonant_class_set():
    """consonant_class は位置引数(shape,start,end,open_amount,consonant_class)で渡せる。"""
    ev = lipsync.MouthEvent(
        lipsync.MouthShape.A, 0.0, 10.0, 0.5, lipsync.ConsonantClass.ROUNDED
    )
    assert ev.consonant_class is lipsync.ConsonantClass.ROUNDED


@pytest.mark.xfail(reason="impl pending: ApertureClass", strict=True)
def test_mouth_event_aperture_class_default():
    """MouthEvent は aperture_class 既定 ApertureClass.NONE を持つ。"""
    ev = lipsync.MouthEvent(shape=lipsync.MouthShape.A, start=0.0, end=10.0)
    assert ev.aperture_class is lipsync.ApertureClass.NONE


@pytest.mark.xfail(reason="impl pending: ApertureClass", strict=True)
def test_mouth_event_aperture_class_set():
    """aperture_class は consonant_class の後に位置引数(既存フィールドの末尾に追加)で渡せる。"""
    ev = lipsync.MouthEvent(
        lipsync.MouthShape.A,
        0.0,
        10.0,
        0.5,
        lipsync.ConsonantClass.ROUNDED,
        lipsync.ApertureClass.FIRM_CLOSURE,
    )
    assert ev.aperture_class is lipsync.ApertureClass.FIRM_CLOSURE


def test_generation_params_defaults():
    """GenerationParams の既定値が初期目安と一致する。

    vowel_scale は母音的口形別(a,i,u,e,o,n)の6要素で既定は全口形 1 倍。
    """
    p = lipsync.GenerationParams()
    assert p.open_cap == 0.8
    assert p.vowel_scale == (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    assert p.attack_frames == 2
    assert p.release_frames == 2
    assert p.min_hold_frames == 3
    assert p.triangle_min_frames == 2.0
    assert p.coartic_overlap_max == 2
    assert p.anticipation_frames == 1
    assert p.legato_valley_shallow == 0.4
    assert p.legato_valley_deep == 0.2
    assert p.legato_valley_slope == 0.025
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
    """generate_morph_keys の出力を VmdDocument に組み立て、vmd の
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
