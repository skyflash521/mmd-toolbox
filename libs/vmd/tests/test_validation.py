import math

import pytest

from vmd import (
    BoneKey,
    CameraKey,
    IkBone,
    IkPropertyKey,
    LightKey,
    MorphKey,
    SelfShadowKey,
    VmdDocument,
    VmdFormatError,
    read,
    write,
)
from vmd.fit import BoneRotationChannel
from vmd.interp import bake_bone_track, sample
from vmd.reduce import BONE_LINEAR_INTERP as LINEAR_BONE_INTERP
from vmd.reduce import CAMERA_LINEAR_INTERP as LINEAR_CAMERA_INTERP

BONE_NAME = b"center".ljust(15, b"\x00")
IDENTITY = (0.0, 0.0, 0.0, 1.0)
ZERO_QUAT = (0.0, 0.0, 0.0, 0.0)


def bone(frame=0, position=(0.0, 0.0, 0.0), rotation=IDENTITY) -> BoneKey:
    return BoneKey(BONE_NAME, frame, position, rotation, LINEAR_BONE_INTERP)


def cam(
    frame=0,
    distance=-30.0,
    position=(0.0, 0.0, 0.0),
    rotation=(0.0, 0.0, 0.0),
    fov=30,
    perspective=0,
) -> CameraKey:
    return CameraKey(frame, distance, position, rotation, LINEAR_CAMERA_INTERP, fov, perspective)


def _import_validate():
    from vmd import VmdValueError, validate_values

    return validate_values, VmdValueError


def _base_bytes() -> bytes:
    return write(VmdDocument(bone=[bone(3)], camera=[cam(5)]))


class TestTrailingBytes:
    def test_read_keeps_trailing_bytes_with_warning(self):
        extra = b"\x01\x02\xffabc"
        doc, warnings = read(_base_bytes() + extra)
        assert doc.trailing_raw == extra
        assert isinstance(doc.trailing_raw, bytes)
        assert [w.code for w in warnings if w.code == "trailing-bytes"] == ["trailing-bytes"]

    def test_unmodified_write_round_trips_byte_exact(self):
        data = _base_bytes() + b"\x00\x7f\x80tail"
        doc, _ = read(data)
        assert write(doc) == data

    def test_no_trailing_bytes_gives_empty_and_no_warning(self):
        doc, warnings = read(_base_bytes())
        assert doc.trailing_raw == b""
        assert not [w for w in warnings if w.code == "trailing-bytes"]

    def test_default_document_has_empty_trailing_raw(self):
        assert VmdDocument().trailing_raw == b""


def _format_error(doc: VmdDocument) -> VmdFormatError:
    with pytest.raises(VmdFormatError) as ei:
        write(doc)
    return ei.value


class TestWriteValidation:
    def test_self_shadow_without_section_flag(self):
        doc = VmdDocument(
            has_self_shadow_section=False,
            has_ik_section=False,
            self_shadow=[SelfShadowKey(0, 1, 0.5)],
        )
        err = _format_error(doc)
        assert (err.section, err.key_index) == ("self_shadow", 0)

    def test_ik_flag_true_with_ik_keys_but_no_self_shadow_section(self):
        doc = VmdDocument(
            has_self_shadow_section=False,
            has_ik_section=True,
            ik_property=[IkPropertyKey(0, 1)],
        )
        err = _format_error(doc)
        assert (err.section, err.key_index) == ("ik_property", 0)

    def test_ik_without_section_flag_but_self_shadow_section_present(self):
        doc = VmdDocument(
            has_self_shadow_section=True,
            has_ik_section=False,
            ik_property=[IkPropertyKey(0, 1)],
        )
        err = _format_error(doc)
        assert (err.section, err.key_index) == ("ik_property", 0)

    def test_ik_flag_true_without_self_shadow_section_is_rejected_even_when_ik_is_empty(self):
        doc = VmdDocument(has_self_shadow_section=False, has_ik_section=True)
        err = _format_error(doc)
        assert (err.section, err.key_index) == ("ik_property", 0)

    def test_ik_flag_true_with_empty_ik_and_self_shadow_section_is_accepted(self):
        doc = VmdDocument(has_self_shadow_section=True, has_ik_section=True)
        read(write(doc))

    def test_negative_bone_frame_points_at_key(self):
        err = _format_error(VmdDocument(bone=[bone(0), bone(-1)]))
        assert (err.section, err.key_index) == ("bone", 1)

    def test_frame_over_32bit_points_at_key(self):
        err = _format_error(VmdDocument(camera=[cam(0), cam(1), cam(2**32)]))
        assert (err.section, err.key_index) == ("camera", 2)

    def test_negative_fov_points_at_key(self):
        err = _format_error(VmdDocument(camera=[cam(0, fov=-1)]))
        assert (err.section, err.key_index) == ("camera", 0)

    def test_fov_over_32bit_points_at_key(self):
        err = _format_error(VmdDocument(camera=[cam(0), cam(1, fov=2**32)]))
        assert (err.section, err.key_index) == ("camera", 1)

    @pytest.mark.parametrize("perspective", [-1, 256])
    def test_perspective_out_of_byte_range(self, perspective):
        err = _format_error(VmdDocument(camera=[cam(0, perspective=perspective)]))
        assert (err.section, err.key_index) == ("camera", 0)

    def test_self_shadow_mode_out_of_byte_range(self):
        doc = VmdDocument(self_shadow=[SelfShadowKey(0, 0, 0.0), SelfShadowKey(1, 256, 0.0)])
        err = _format_error(doc)
        assert (err.section, err.key_index) == ("self_shadow", 1)

    def test_ik_display_out_of_byte_range(self):
        err = _format_error(VmdDocument(ik_property=[IkPropertyKey(0, 256)]))
        assert (err.section, err.key_index) == ("ik_property", 0)

    @pytest.mark.parametrize("enable", [-1, 256])
    def test_ik_enable_out_of_byte_range(self, enable):
        ik_bone = IkBone(b"ik".ljust(20, b"\0"), enable)
        doc = VmdDocument(ik_property=[IkPropertyKey(0, 1), IkPropertyKey(1, 1, [ik_bone])])
        err = _format_error(doc)
        assert (err.section, err.key_index) == ("ik_property", 1)

    def test_bone_position_overflowing_float32(self):
        err = _format_error(VmdDocument(bone=[bone(0), bone(1, position=(0.0, 1e40, 0.0))]))
        assert (err.section, err.key_index) == ("bone", 1)

    def test_camera_distance_overflowing_float32(self):
        err = _format_error(VmdDocument(camera=[cam(0), cam(1, distance=1e40)]))
        assert (err.section, err.key_index) == ("camera", 1)

    def test_morph_weight_overflowing_float32(self):
        morphs = [MorphKey(BONE_NAME, 0, 0.0), MorphKey(BONE_NAME, 1, 1e40)]
        err = _format_error(VmdDocument(morph=morphs))
        assert (err.section, err.key_index) == ("morph", 1)


class TestValidateValues:
    def test_import(self):
        validate_values, value_error = _import_validate()
        assert callable(validate_values)
        assert issubclass(value_error, Exception)

    def test_valid_document_passes(self):
        validate_values, _ = _import_validate()
        doc = VmdDocument(
            bone=[bone(0), bone(5, (1.0, 2.0, 3.0), (0.0, 0.0, 1.0, 0.0))],
            camera=[cam(0), cam(10, rotation=(0.5, -0.25, 1.0))],
        )
        validate_values(doc)

    def test_camera_zero_angle_is_accepted(self):
        validate_values, _ = _import_validate()
        validate_values(VmdDocument(camera=[cam(0, rotation=(0.0, 0.0, 0.0))]))

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize(
        ("field", "kwargs"),
        [
            ("distance", lambda b: {"distance": b}),
            ("position", lambda b: {"position": (0.0, b, 0.0)}),
            ("rotation", lambda b: {"rotation": (0.0, 0.0, b)}),
        ],
    )
    def test_camera_non_finite(self, bad, field, kwargs):
        validate_values, value_error = _import_validate()
        doc = VmdDocument(camera=[cam(0), cam(1, **kwargs(bad))])
        with pytest.raises(value_error) as ei:
            validate_values(doc)
        assert (ei.value.section, ei.value.index, ei.value.field) == ("camera", 1, field)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize(
        ("field", "kwargs"),
        [
            ("position", lambda b: {"position": (b, 0.0, 0.0)}),
            ("rotation", lambda b: {"rotation": (0.0, b, 0.0, 1.0)}),
        ],
    )
    def test_bone_non_finite(self, bad, field, kwargs):
        validate_values, value_error = _import_validate()
        doc = VmdDocument(bone=[bone(0), bone(1), bone(2, **kwargs(bad))])
        with pytest.raises(value_error) as ei:
            validate_values(doc)
        assert (ei.value.section, ei.value.index, ei.value.field) == ("bone", 2, field)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_morph_weight_non_finite(self, bad):
        validate_values, value_error = _import_validate()
        doc = VmdDocument(morph=[MorphKey(BONE_NAME, 0, 0.0), MorphKey(BONE_NAME, 1, bad)])
        with pytest.raises(value_error) as ei:
            validate_values(doc)
        assert (ei.value.section, ei.value.index, ei.value.field) == ("morph", 1, "weight")

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    @pytest.mark.parametrize(
        ("field", "make"),
        [
            ("color", lambda b: LightKey(1, (1.0, b, 1.0), (0.0, 0.0, 0.0))),
            ("position", lambda b: LightKey(1, (1.0, 1.0, 1.0), (0.0, 0.0, b))),
        ],
    )
    def test_light_non_finite(self, bad, field, make):
        validate_values, value_error = _import_validate()
        doc = VmdDocument(light=[LightKey(0, (1.0, 1.0, 1.0), (0.0, 0.0, 0.0)), make(bad)])
        with pytest.raises(value_error) as ei:
            validate_values(doc)
        assert (ei.value.section, ei.value.index, ei.value.field) == ("light", 1, field)

    @pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
    def test_self_shadow_distance_non_finite(self, bad):
        validate_values, value_error = _import_validate()
        doc = VmdDocument(self_shadow=[SelfShadowKey(0, 1, 0.5), SelfShadowKey(1, 1, bad)])
        with pytest.raises(value_error) as ei:
            validate_values(doc)
        assert (ei.value.section, ei.value.index, ei.value.field) == ("self_shadow", 1, "distance")

    def test_bone_zero_norm_rotation(self):
        validate_values, value_error = _import_validate()
        doc = VmdDocument(bone=[bone(0), bone(1, rotation=ZERO_QUAT)])
        with pytest.raises(value_error) as ei:
            validate_values(doc)
        assert (ei.value.section, ei.value.index, ei.value.field) == ("bone", 1, "rotation")


def _axis_quat(deg):
    a = math.radians(deg) / 2.0
    return (0.0, 0.0, math.sin(a), math.cos(a))


class TestZeroNormRotation:
    def test_sample_treats_zero_quat_as_identity(self):
        keys = [bone(0, rotation=ZERO_QUAT), bone(10, rotation=_axis_quat(90.0))]
        assert sample(keys, "rot", 0) == pytest.approx(IDENTITY)
        mid = sample(keys, "rot", 5)
        assert mid == pytest.approx(_axis_quat(45.0), abs=1e-3)

    def test_sample_zero_quat_at_end_key(self):
        keys = [bone(0, rotation=_axis_quat(90.0)), bone(10, rotation=ZERO_QUAT)]
        assert sample(keys, "rot", 10) == pytest.approx(IDENTITY)

    def test_sample_single_zero_key(self):
        assert sample([bone(0, rotation=ZERO_QUAT)], "rot", 3) == pytest.approx(IDENTITY)

    def test_bake_bone_track_treats_zero_quat_as_identity(self):
        keys = [bone(0, rotation=ZERO_QUAT), bone(10, rotation=_axis_quat(90.0))]
        _, rotations = bake_bone_track(keys, 0, 10)
        assert rotations[0] == pytest.approx(IDENTITY)
        assert rotations[5] == pytest.approx(_axis_quat(45.0), abs=1e-3)

    def test_bake_bone_track_single_zero_key(self):
        _, rotations = bake_bone_track([bone(0, rotation=ZERO_QUAT)], 0, 2)
        assert rotations == [pytest.approx(IDENTITY)] * 3

    def test_fit_channel_treats_zero_quat_as_identity(self):
        quats = [ZERO_QUAT] + [_axis_quat(9.0 * i) for i in range(1, 11)]
        ch = BoneRotationChannel(0, quats, tol=0.1)
        assert ch.quats[0] == pytest.approx(IDENTITY)
        err, _ = ch.residual(0, 10)
        assert err == pytest.approx(0.0, abs=1e-6)
