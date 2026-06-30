"""ボーン専用トレランス構築ヘルパのテスト(mmd_toolbox.md vmd.reduce)。

build_bone_tolerances は bone_pos / bone_rot だけを受け取り、未使用のカメラフィールドを 0.0 で
埋めた Tolerances を返す。ボーンのみを扱うツール(mocapvmd)がカメラ許容値を指定せずに済むようにする。
"""

from vmd import reduce
from vmd.reduce import Tolerances, reduce_bone_track
from vmd.types import BoneKey

LINEAR = reduce.BONE_LINEAR_INTERP


def test_build_bone_tolerances_sets_bone_fields_and_zeroes_camera():
    t = reduce.build_bone_tolerances(0.02, 0.20)
    assert isinstance(t, Tolerances)
    assert t.bone_pos == 0.02
    assert t.bone_rot == 0.20
    assert t.camera_pos == 0.0
    assert t.camera_rot == 0.0
    assert t.camera_distance == 0.0
    assert t.camera_fov == 0.0


def test_build_bone_tolerances_usable_by_reduce_bone_track():
    # 返した Tolerances がそのまま reduce_bone_track に渡せる(ボーン疎化が成立する)。
    # 線形に並ぶ密キーは中間キーが省かれ、端の2キーへ削減される。
    keys = [
        BoneKey(b"\x00" * 15, f, (float(f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), LINEAR)
        for f in range(5)
    ]
    tols = reduce.build_bone_tolerances(0.01, 0.10)
    out = reduce_bone_track(
        keys, [(0, 4)], tols,
        cut_thresholds=(1.0, 30.0), keep_frames=[], no_cut_detect=False,
        min_seg=1, max_seg=180, strict=False, curve_mode="linear",
    )
    assert [k.frame for k in out] == [0, 4]
