import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIBS_DIR = _REPO_ROOT / "libs"

_BEZIER_REDUCE_DRIVER = r"""
import math
from vmd import reduce
from vmd.types import BoneKey

n = 30
keys = []
for f in range(n):
    t = f / 10.0
    x = math.sin(t) * 3.0 + 0.1 * math.sin(7.3 * t)
    y = math.cos(t * 0.7) * 2.0 + 0.08 * math.sin(11.1 * t)
    z = math.sin(t * 1.3) * 1.5
    ang = math.radians(30.0 * math.sin(t * 0.5))
    q = (0.0, 0.0, math.sin(ang / 2), math.cos(ang / 2))
    keys.append(BoneKey(b"\x00" * 15, f, (x, y, z), q, reduce.BONE_LINEAR_INTERP))

tols = reduce.build_bone_tolerances(0.02, 0.20)
out = reduce.reduce_bone_track(
    keys, [(0, n - 1)], tols,
    cut_thresholds=(1.0, 30.0), keep_frames=[], no_cut_detect=True,
    min_seg=1, max_seg=180, strict=False, curve_mode="bezier",
)
assert out, "reduce produced no keys"
"""

_TIMEOUT_SEC = 30


def _env_importing_this_tree_libs_first() -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_LIBS_DIR), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(_LIBS_DIR)]
    )
    return env


def test_bezier_reduce_does_not_hang():
    try:
        # signal.SIGALRM は Windows に無い。
        proc = subprocess.run(
            [sys.executable, "-c", _BEZIER_REDUCE_DRIVER],
            cwd=_REPO_ROOT,
            env=_env_importing_this_tree_libs_first(),
            timeout=_TIMEOUT_SEC,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(f"bezier 疎化が {_TIMEOUT_SEC}s 以内に完了しなかった")
    assert proc.returncode == 0, f"サブプロセスが異常終了: {proc.stderr}"
