"""frame-0 中立キー補完のテスト。

出力VMDの編集・互換のため、名前付きセクション(morph/bone)の参照名へ frame=0 の中立キーを補う
`ensure_frame0_neutral_keys` を検証する。既存の frame-0 は尊重し、無いものだけ中立値で挿入する。
write/normalize とは独立した明示ステップで、CLI が出力前に呼ぶ。
"""

import pytest

from vmd import BoneKey, MorphKey, VmdDocument, ensure_frame0_neutral_keys


def _morph(name, frame, weight=0.5):
    return MorphKey(name.encode("cp932").ljust(15, b"\x00"), frame, weight)


def _bone(name, frame):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, (1.0, 2.0, 3.0),
                   (0.1, 0.2, 0.3, 0.9), b"\x01" * 64)


def test_morph_frame0_added_when_missing():
    # frame-0 を持たないモーフへ中立キー (name, 0, 0.0) を補う。
    doc = VmdDocument(morph=[_morph("あ", 10), _morph("あ", 20)])
    out = ensure_frame0_neutral_keys(doc)
    zero = [k for k in out.morph if k.frame == 0]
    assert len(zero) == 1
    assert zero[0].name == "あ"
    assert zero[0].weight == pytest.approx(0.0)
    assert zero[0].name_raw == "あ".encode("cp932").ljust(15, b"\x00")


def test_existing_frame0_respected():
    # 既に frame-0 を持つモーフは中立キーを足さず既存を尊重する(重複させない)。
    doc = VmdDocument(morph=[_morph("い", 0, 0.7), _morph("い", 10)])
    out = ensure_frame0_neutral_keys(doc)
    zero = [k for k in out.morph if k.name == "い" and k.frame == 0]
    assert len(zero) == 1
    assert zero[0].weight == pytest.approx(0.7)  # 既存値を保つ


def test_each_referenced_morph_gets_frame0():
    # 参照される各モーフ名に frame-0 が付く。
    doc = VmdDocument(morph=[_morph("あ", 5), _morph("う", 8), _morph("お", 0, 0.3)])
    out = ensure_frame0_neutral_keys(doc)
    zero_names = {k.name for k in out.morph if k.frame == 0}
    assert zero_names == {"あ", "う", "お"}


def test_no_morph_no_change():
    # キーが無ければ何も足さない。
    doc = VmdDocument()
    out = ensure_frame0_neutral_keys(doc)
    assert out.morph == []


def test_bone_frame0_neutral_when_generalized():
    # 一般化: sections に bone を指定すると、ボーンへ identity 回転・ゼロ位置・既定リニア補間の中立キーを補う。
    doc = VmdDocument(bone=[_bone("センター", 10)])
    out = ensure_frame0_neutral_keys(doc, sections=("morph", "bone"))
    zero = [k for k in out.bone if k.frame == 0]
    assert len(zero) == 1
    assert zero[0].name == "センター"
    assert zero[0].position == pytest.approx((0.0, 0.0, 0.0))
    assert zero[0].rotation == pytest.approx((0.0, 0.0, 0.0, 1.0))
    # 既定リニア補間: 全チャンネル (20, 20, 107, 107)。
    for cp in zero[0].control_points().values():
        assert cp == (20, 20, 107, 107)


def test_unnamed_section_rejected():
    # camera/light/self_shadow 等は名前付き使用集合でないので対象外(指定したらエラー)。
    doc = VmdDocument()
    with pytest.raises(ValueError):
        ensure_frame0_neutral_keys(doc, sections=("camera",))
