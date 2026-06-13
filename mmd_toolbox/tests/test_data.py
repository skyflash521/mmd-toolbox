"""MMD産テストデータが tests/data/README.md の存在条件を満たすことの検査。

1条件 = 1テスト。失敗した場合はテストデータの作り直しが必要
(条件と作成手順は tests/data/README.md)。
"""

import pytest

from mmd_toolbox.vmd import read

DEFAULT_CAMERA_INTERP = bytes([20, 107, 20, 107]) * 6
REMAKE = "テストデータの作り直しが必要(mmd_toolbox/tests/data/README.md): "


@pytest.fixture
def camera_basic(camera_basic_bytes):
    doc, _ = read(camera_basic_bytes)
    return doc


@pytest.fixture
def model_motion(model_motion_bytes):
    doc, _ = read(model_motion_bytes)
    return doc


class TestCameraBasicConditions:
    def test_camera_keys_at_least_3(self, camera_basic):
        assert len(camera_basic.camera) >= 3, REMAKE + "カメラキーが3個以上必要"

    def test_has_default_interp_key(self, camera_basic):
        # 区間の補間曲線は到達側(後側)キーに格納されるため(vmd-interp.md §2)、
        # 先頭キーの補間曲線はどの区間にも使われず、判定の対象外
        interps = [k.interpolation for k in camera_basic.camera[1:]]
        assert any(i == DEFAULT_CAMERA_INTERP for i in interps), (
            REMAKE + "補間曲線がデフォルト値のキー(2個目以降)が1個以上必要"
        )

    def test_has_non_default_interp_key(self, camera_basic):
        interps = [k.interpolation for k in camera_basic.camera[1:]]
        assert any(i != DEFAULT_CAMERA_INTERP for i in interps), (
            REMAKE + "補間曲線がデフォルト値でないキー(2個目以降)が1個以上必要"
        )

    def test_distance_has_multiple_values(self, camera_basic):
        assert len({k.distance for k in camera_basic.camera}) > 1, (
            REMAKE + "距離の値が2種類以上必要"
        )

    def test_target_position_has_multiple_values(self, camera_basic):
        assert len({k.position for k in camera_basic.camera}) > 1, (
            REMAKE + "カメラ中心の座標(X・Y・Zの組)が2種類以上必要"
        )

    def test_rotation_has_multiple_values(self, camera_basic):
        assert len({k.rotation for k in camera_basic.camera}) > 1, (
            REMAKE + "角度(X・Y・Zの組)が2種類以上必要"
        )

    def test_fov_has_multiple_values(self, camera_basic):
        assert len({k.fov for k in camera_basic.camera}) > 1, (
            REMAKE + "視野角の値が2種類以上必要"
        )

    def test_perspective_has_both_values(self, camera_basic):
        assert {k.perspective for k in camera_basic.camera} == {0, 1}, (
            REMAKE + "パースONのキーとOFFのキーの両方が必要"
        )

    def test_has_light_key(self, camera_basic):
        assert len(camera_basic.light) >= 1, REMAKE + "照明キーが1個以上必要"

    def test_has_self_shadow_key(self, camera_basic):
        assert len(camera_basic.self_shadow) >= 1, REMAKE + "セルフ影キーが1個以上必要"


class TestModelMotionConditions:
    def test_has_bone_key(self, model_motion):
        assert len(model_motion.bone) >= 1, REMAKE + "ボーンキーが1個以上必要"

    def test_has_morph_key(self, model_motion):
        assert len(model_motion.morph) >= 1, REMAKE + "モーフキーが1個以上必要"
