import pytest

from vmd import BoneKey, MorphKey, VmdDocument, ensure_frame0_neutral_keys


def _morph(name, frame, weight=0.5):
    return MorphKey(name.encode("cp932").ljust(15, b"\x00"), frame, weight)


def _bone(name, frame):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, (1.0, 2.0, 3.0),
                   (0.1, 0.2, 0.3, 0.9), b"\x01" * 64)


def test_morph_frame0_added_with_zero_weight_when_missing():
    doc = VmdDocument(morph=[_morph("あ", 10), _morph("あ", 20)])
    out = ensure_frame0_neutral_keys(doc)
    zero = [k for k in out.morph if k.frame == 0]
    assert len(zero) == 1
    assert zero[0].name == "あ"
    assert zero[0].weight == pytest.approx(0.0)
    assert zero[0].name_raw == "あ".encode("cp932").ljust(15, b"\x00")


def test_existing_frame0_kept_without_duplicate():
    doc = VmdDocument(morph=[_morph("い", 0, 0.7), _morph("い", 10)])
    out = ensure_frame0_neutral_keys(doc)
    zero = [k for k in out.morph if k.name == "い" and k.frame == 0]
    assert len(zero) == 1
    assert zero[0].weight == pytest.approx(0.7)


def test_each_referenced_morph_gets_frame0():
    doc = VmdDocument(morph=[_morph("あ", 5), _morph("う", 8), _morph("お", 0, 0.3)])
    out = ensure_frame0_neutral_keys(doc)
    zero_names = {k.name for k in out.morph if k.frame == 0}
    assert zero_names == {"あ", "う", "お"}


def test_neutral_keys_appended_after_existing_keys():
    existing = [_morph("あ", 10), _morph("う", 8)]
    out = ensure_frame0_neutral_keys(VmdDocument(morph=list(existing)))
    assert out.morph[:2] == existing
    assert [k.frame for k in out.morph[2:]] == [0, 0]


def test_no_morph_no_change():
    doc = VmdDocument()
    out = ensure_frame0_neutral_keys(doc)
    assert out.morph == []


def test_bone_section_gets_identity_zero_position_linear_interp_key():
    doc = VmdDocument(bone=[_bone("センター", 10)])
    out = ensure_frame0_neutral_keys(doc, sections=("morph", "bone"))
    zero = [k for k in out.bone if k.frame == 0]
    assert len(zero) == 1
    assert zero[0].name == "センター"
    assert zero[0].position == pytest.approx((0.0, 0.0, 0.0))
    assert zero[0].rotation == pytest.approx((0.0, 0.0, 0.0, 1.0))
    for cp in zero[0].control_points().values():
        assert cp == (20, 20, 107, 107)


@pytest.mark.parametrize("section", ["camera", "light", "self_shadow", "ik_property"])
def test_unnamed_section_rejected(section):
    doc = VmdDocument()
    with pytest.raises(ValueError):
        ensure_frame0_neutral_keys(doc, sections=(section,))
