from vmd import reduce
from vmd.reduce import Tolerances


def test_build_bone_tolerances_sets_bone_fields_and_zeroes_camera():
    t = reduce.build_bone_tolerances(0.02, 0.20)
    assert isinstance(t, Tolerances)
    assert t.bone_pos == 0.02
    assert t.bone_rot == 0.20
    assert t.camera_pos == 0.0
    assert t.camera_rot == 0.0
    assert t.camera_distance == 0.0
    assert t.camera_fov == 0.0
