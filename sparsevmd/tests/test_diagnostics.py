"""削減診断の surface とレポート出力のテスト(sparsevmd.md §2.7, §6.3)。

§2.7 はレポートに「不連続検出位置」(dry-run)・「分割理由」(JSON)を求める。§6.3 は
継ぎ目の補間曲線書き換えを「レポートに明示する」と求める。reduce_*_track は diagnostics
out-param(dict)を受け取り、cuts(検出カット位置)・splits(分割フレームと駆動チャンネル)・
seam_rewrites(継ぎ目で曲線を書き換えたフレーム)を埋める。build_report はこれを各トラック
エントリに載せ、dry-run と JSON に出す。
"""

import json

from mmd_toolbox.vmd import interp
from mmd_toolbox.vmd.types import BoneKey, CameraKey
from sparsevmd import presets, report
from sparsevmd.reduce import camera_interp_bytes, reduce_camera_track

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
EASE = (96, 0, 96, 30)
LIN = (20, 20, 107, 107)
TOLS = presets.resolve_tolerances("balanced")
CAM_CUT = (5.0, 20.0, 5.0)


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, interp_block=None):
    return CameraKey(frame, dist, center, rot, interp_block or CAM_LINEAR, fov, 0)


def _eased(v0, v1, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*EASE, f / span) for f in range(n)]


def camera_track(source, ranges, **kw):
    opts = dict(
        cut_thresholds=CAM_CUT,
        keep_frames=[],
        no_cut_detect=True,
        min_seg=1,
        max_seg=180,
        strict=False,
        curve_mode="bezier",
    )
    opts.update(kw)
    return reduce_camera_track(source, ranges, TOLS, **opts)


# --- reduce out-param -------------------------------------------------------


def test_diagnostics_captures_cut_positions():
    # frame15 以降に +50 のオフセット(frame14=14 → frame15=65、差分51)→ 不連続検出。
    src = [
        cam(f, center=((float(f) if f < 15 else float(f) + 50.0), 0.0, 0.0))
        for f in range(31)
    ]
    d = {}
    camera_track(src, [(0, 30)], diagnostics=d, no_cut_detect=False)
    assert 15 in d["cuts"]


def test_diagnostics_captures_split_with_channel():
    # Y がイージング → linear では位置チャンネル駆動で分割が起きる。
    ys = _eased(0.0, 100.0)
    src = [cam(f, center=(0.0, ys[f], 0.0)) for f in range(11)]
    d = {}
    camera_track(src, [(0, 10)], diagnostics=d, curve_mode="linear")
    assert d["splits"]
    s = d["splits"][0]
    assert "frame" in s
    assert s["channel"] == "position"
    assert s["norm_error"] > 1.0


def test_diagnostics_captures_seam_rewrites():
    # key10 の pos_x が [0,10] eased。range[10,20] で frame0 が範囲外 → 下側継ぎ目で key10 書換え。
    blk = camera_interp_bytes(EASE, LIN, LIN, LIN, LIN, LIN)
    src = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(100.0, 0.0, 0.0), interp_block=blk),
           cam(20, center=(100.0, 0.0, 0.0))]
    d = {}
    camera_track(src, [(10, 20)], diagnostics=d, curve_mode="bezier")
    assert 10 in d["seam_rewrites"]


def test_diagnostics_optional_no_dict():
    # diagnostics 未指定でも従来どおり動く。
    src = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(11)]
    keys = camera_track(src, [(0, 10)])
    assert [k.frame for k in keys] == [0, 10]


# --- report 統合 ------------------------------------------------------------


def _diag():
    return {
        "cuts": [15],
        "splits": [{"frame": 7, "channel": "position", "norm_error": 2.0}],
        "seam_rewrites": [10],
    }


def test_build_report_includes_camera_diagnostics():
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_diag=_diag(),
    )
    assert rep["camera"]["diagnostics"] == _diag()


def test_build_report_no_diagnostics_when_omitted():
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[],
    )
    assert "diagnostics" not in rep["camera"]


def test_format_dry_run_shows_cut_positions():
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_diag=_diag(),
    )
    text = report.format_dry_run(rep)
    assert "cuts" in text.lower() or "不連続" in text
    assert "15" in text


def test_write_json_includes_splits_and_seams(tmp_path):
    rep = report.build_report(
        target="camera", camera=(31, 5), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_diag=_diag(),
    )
    p = tmp_path / "r.json"
    report.write_json(rep, str(p))
    data = json.loads(p.read_text(encoding="utf-8"))
    diag = data["camera"]["diagnostics"]
    assert diag["splits"][0]["channel"] == "position"
    assert diag["seam_rewrites"] == [10]
    assert diag["cuts"] == [15]
