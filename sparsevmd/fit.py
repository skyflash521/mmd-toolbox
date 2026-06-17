"""sparsevmd.fit: 後方互換のための re-export。

補間曲線フィット・誤差評価の実体は共通ライブラリ mmd_toolbox.vmd.fit へ移送した
(リファクタ refactor-plan-direct-reduce.md Step 1)。本モジュールは旧 import パス
(`sparsevmd.fit`)を維持するための薄い再公開層で、公開・非公開シンボルを同名で再輸出する。
"""

from mmd_toolbox.vmd.fit import (  # noqa: F401
    _BEZIER_INITS,
    _BEZIER_LINEAR_CP,
    _ZERO_EPS,
    BoneRotationChannel,
    CameraRotationChannel,
    EuclideanVectorChannel,
    FovChannel,
    LinearScalarChannel,
    _axis_curve,
    _bez,
    _bezier_axis_pred,
    _bezier_y_at,
    _clip01,
    _fit_coeff_curve,
    _normalize,
    _quantize_cp,
    _quat_angle_deg,
    _quat_conj,
    _quat_dot,
    _quat_mul,
    _quat_normalize,
    _quat_slerp,
    _round_half_up,
    _select_worst,
    fit_bezier_curve,
)
