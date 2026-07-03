"""sparsevmd.reduce: 後方互換のための re-export。

区間分割・キー削減・出力後検証・継ぎ目処理・誤差測定の実体は共通ライブラリ
vmd.reduce にある。
本モジュールは旧 import パス(`sparsevmd.reduce`)を維持するための薄い再公開層。
"""

from vmd.reduce import (  # noqa: F401
    BONE_LINEAR_INTERP,
    CAMERA_LINEAR_INTERP,
    StrictError,
    Tolerances,
    bone_interp_bytes,
    build_bone_keys,
    build_camera_keys,
    camera_interp_bytes,
    measure_bone_errors,
    measure_camera_errors,
    reduce_bone_track,
    reduce_camera_track,
    reduce_track,
    verify_bone_track,
    verify_camera_track,
)
