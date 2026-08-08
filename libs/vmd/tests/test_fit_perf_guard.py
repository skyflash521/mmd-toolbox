"""疎化フィットの性能ガード。

性能チューニングの実装はミスるとフィットが収束せず止まりうる。本テストは代表的な bezier 疎化
ワークロード(曲線的＋微小ゆらぎの密ボーン動作=多数の小区間フィットを踏む)をサブプロセスで
実行し、明示の上限時間で確実に停止させることで、ハング・破滅的な性能回帰を「失敗」として
検出する(遅い/止まるフィット1つで pytest スイート全体を巻き込まない)。`signal.SIGALRM` は
Windows に無いので使わず、子プロセスごと確実に殺せるサブプロセス＋タイムアウトで止める。
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

# libs/vmd/tests/<this> から見たリポジトリルート。サブプロセスには PYTHONPATH に libs/ を渡し、
# editable install や site-packages の古いコピーでなく、このツリーのソース vmd を import させる。
_REPO_ROOT = Path(__file__).resolve().parents[3]
_LIBS_DIR = _REPO_ROOT / "libs"

# 代表ワークロード。外部ファイルを要求せずテスト内で組み立てる。出力が空でないことも確認。
_DRIVER = r"""
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

# 通常 ~2 秒。上限はハング・破滅的回帰(十数倍以上)だけを捕らえる余裕値で、通常の機械差では
# 発火しない(タイトな性能閾値ではなく、止まらないことの保証)。
_TIMEOUT_SEC = 30


def test_bezier_reduce_does_not_hang():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(_LIBS_DIR), env["PYTHONPATH"]] if env.get("PYTHONPATH") else [str(_LIBS_DIR)]
    )
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _DRIVER],
            cwd=_REPO_ROOT,
            env=env,
            timeout=_TIMEOUT_SEC,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"bezier 疎化が {_TIMEOUT_SEC}s 以内に完了しなかった(ハング/破滅的な性能回帰の疑い)。"
            " 曲線評価のベクトル化など棄却済み最適化を踏んでいないか確認。"
        )
    assert proc.returncode == 0, f"サブプロセスが異常終了: {proc.stderr}"
