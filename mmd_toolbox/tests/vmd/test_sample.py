"""perspective 直近ホールドのテスト(sparsevmd.md §4.2)。

perspective_series は各フレームの perspective を直近キー値で保持して返す。
当該フレーム以前にキーが無い場合は先頭キーの値を用いる(vmd-interp の境界規約に倣う)。
"""

import pytest

from mmd_toolbox.vmd.types import CameraKey
from mmd_toolbox.vmd.sample import perspective_series


# 真の線形補間になる制御点(各チャンネル x1==y1, x2==y2 → y=x)。
CAM_LINEAR = bytes([20, 107, 20, 107]) * 6  # 24バイト


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, persp)


def test_perspective_series_holds_nearest_prior():
    # 0..30 が persp=0、30 で persp=1 に切替。
    keys = [cam(0, persp=0), cam(30, persp=1)]
    series = perspective_series(keys, 0, 31)
    assert series[0] == 0
    assert series[29] == 0  # 30より前は直近(0)
    assert series[30] == 1  # 切替フレーム
    assert series[31] == 1  # 以降ホールド


def test_perspective_series_empty_raises():
    # 空キー列は interp に倣い ValueError(直接呼び出し時の防御)。
    with pytest.raises(ValueError):
        perspective_series([], 0, 10)


def test_perspective_before_first_key_uses_first():
    # 当該フレーム以前にキーが無い場合は先頭キーの値を用いる。これは §4.2 の
    # 直近ホールドの境界補完であり、vmd-interp の境界規約(最初のキー以前は端キーの
    # 値で一定)に倣う。実パイプラインの抽出はトラック extent 内なので通常は発生しない。
    keys = [cam(10, persp=1), cam(20, persp=0)]
    series = perspective_series(keys, 0, 20)
    assert series[0] == 1  # 先頭キー以前は先頭値
    assert series[20] == 0
