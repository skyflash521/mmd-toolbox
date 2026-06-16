"""レポートの誤差診断テスト(sparsevmd.md §7.2, §2.7)。

§7.2「最終レポートには軸ごとの最大絶対誤差も出す」に対応し、出力を再サンプリングして
元サンプルとのチャンネル別・軸別の最大絶対誤差を測り、dry-run / JSON / CSV に載せる。

- measure_camera_errors: pos_x/pos_y/pos_z(軸別)・distance・fov・rot_deg(軸別角度の最大)
- measure_bone_errors: pos_x/pos_y/pos_z・rot_deg(quaternion 角度距離)
"""

import csv as _csv
import json

import pytest

from mmd_toolbox.vmd import interp
from mmd_toolbox.vmd.types import BoneKey, CameraKey
from sparsevmd import presets, report
from sparsevmd.reduce import measure_bone_errors, measure_camera_errors

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
EASE = (96, 0, 96, 30)


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30):
    return CameraKey(frame, dist, center, rot, CAM_LINEAR, fov, 0)


def bone(name, frame, pos=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0, 1.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, rot, BL)


def _eased(v0, v1, n=11):
    span = n - 1
    return [v0 + (v1 - v0) * interp._solve_factor(*EASE, f / span) for f in range(n)]


# --- measure_camera_errors --------------------------------------------------


def test_measure_camera_errors_zero_for_faithful_output():
    src = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(11)]
    out = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(10.0, 0.0, 0.0))]
    m = measure_camera_errors(src, out, [(0, 10)])
    # 線形再サンプリングの浮動小数ドリフトを許容(量子化下限相当)。
    assert m["pos_x"] == pytest.approx(0.0, abs=1e-6)
    assert m["pos_y"] == 0.0
    assert m["distance"] == 0.0
    assert m["rot_deg"] == pytest.approx(0.0, abs=1e-6)


def test_measure_camera_errors_per_axis_position():
    # Y 軸だけ曲線を2キー線形で表す → pos_y に誤差、pos_x/pos_z は0。
    ys = _eased(0.0, 100.0)
    src = [cam(f, center=(0.0, ys[f], 0.0)) for f in range(11)]
    out = [cam(0, center=(0.0, 0.0, 0.0)), cam(10, center=(0.0, 100.0, 0.0))]
    m = measure_camera_errors(src, out, [(0, 10)])
    assert m["pos_y"] > 5.0
    assert m["pos_x"] == pytest.approx(0.0, abs=1e-9)
    assert m["pos_z"] == pytest.approx(0.0, abs=1e-9)


# --- measure_bone_errors ----------------------------------------------------


def test_measure_bone_errors_position_and_rotation():
    xs = _eased(0.0, 100.0)
    src = [bone("c", f, pos=(xs[f], 0.0, 0.0)) for f in range(11)]
    out = [bone("c", 0, pos=(0.0, 0.0, 0.0)), bone("c", 10, pos=(100.0, 0.0, 0.0))]
    m = measure_bone_errors(src, out, [(0, 10)])
    assert m["pos_x"] > 5.0
    assert m["pos_y"] == pytest.approx(0.0, abs=1e-9)
    assert m["rot_deg"] == pytest.approx(0.0, abs=1e-6)


# --- レポート統合 -----------------------------------------------------------


def _cam_errors():
    return {"pos_x": 0.5, "pos_y": 0.0, "pos_z": 0.0, "distance": 0.1, "fov": 0.0, "rot_deg": 0.3}


def test_build_report_includes_camera_errors():
    rep = report.build_report(
        target="camera",
        camera=(31, 2),
        bones=None,
        selected_bones=set(),
        ranges=[(0, 30)],
        keep_frames=[],
        camera_errors=_cam_errors(),
    )
    assert rep["camera"]["errors"] == _cam_errors()


def test_build_report_includes_bone_errors_per_name():
    berr = {"センター": {"pos_x": 0.2, "pos_y": 0.0, "pos_z": 0.0, "rot_deg": 0.1}}
    rep = report.build_report(
        target="bone",
        camera=None,
        bones={"センター": (31, 3)},
        selected_bones={"センター"},
        ranges=[(0, 30)],
        keep_frames=[],
        bone_errors=berr,
    )
    entry = rep["bones"][0]
    assert entry["errors"] == berr["センター"]


def test_format_dry_run_shows_camera_max_error():
    rep = report.build_report(
        target="camera", camera=(31, 2), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_errors=_cam_errors(),
    )
    text = report.format_dry_run(rep)
    assert "max error" in text or "最大誤差" in text
    assert "0.5" in text  # pos_x の最大誤差が表示される


def test_write_json_contains_errors(tmp_path):
    rep = report.build_report(
        target="camera", camera=(31, 2), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[], camera_errors=_cam_errors(),
    )
    p = tmp_path / "r.json"
    report.write_json(rep, str(p))
    data = json.loads(p.read_text(encoding="utf-8"))
    assert data["camera"]["errors"]["pos_x"] == 0.5


def test_cli_preview_csv_is_per_frame(tmp_path):
    # §2.7 --preview-csv はフレーム毎の入力/出力サンプル値と誤差。トラック要約ではない。
    from mmd_toolbox.vmd import io
    from mmd_toolbox.vmd.types import VmdDocument
    from sparsevmd import cli

    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    prev = tmp_path / "preview.csv"
    keys = [cam(f, center=(float(f), 0.0, 0.0)) for f in range(11)]
    io.write_file(VmdDocument(camera=keys), str(src))
    code = cli.main(
        [str(src), "-o", str(out), "--target", "camera", "--preview-csv", str(prev)]
    )
    assert code == 0
    rows = list(_csv.reader(prev.read_text(encoding="utf-8").splitlines()))
    header = rows[0]
    assert header == ["track", "frame", "channel", "input", "output", "error"]
    frames = {r[header.index("frame")] for r in rows[1:]}
    channels = {r[header.index("channel")] for r in rows[1:]}
    assert len(frames) > 2  # フレーム毎(要約でない)
    assert "pos_x" in channels
    # center.x=f は完全な線形。削減後のキー数に関わらず線形補間で各フレーム完全再現される
    # ため、frame5 の pos_x は入力出力一致・誤差0。
    row5 = [
        r for r in rows[1:]
        if r[header.index("frame")] == "5" and r[header.index("channel")] == "pos_x"
    ][0]
    assert row5[header.index("input")] == "5.000000"
    assert row5[header.index("output")] == "5.000000"
    assert row5[header.index("error")] == "0.000000"


def test_cli_report_json_includes_camera_errors(tmp_path):
    # CLI 経由で --report-json が軸別最大誤差を含むことをエンドツーエンドで確認する。
    from mmd_toolbox.vmd import io
    from mmd_toolbox.vmd.types import VmdDocument
    from sparsevmd import cli

    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    rep = tmp_path / "r.json"
    keys = [
        cam(f, dist=-10.0 - 100.0 * interp._solve_factor(*EASE, f / 30.0))
        for f in range(31)
    ]
    io.write_file(VmdDocument(camera=keys), str(src))
    code = cli.main([str(src), "-o", str(out), "--target", "camera", "--report-json", str(rep)])
    assert code == 0
    data = json.loads(rep.read_text(encoding="utf-8"))
    assert "errors" in data["camera"]
    for k in ("pos_x", "pos_y", "pos_z", "distance", "fov", "rot_deg"):
        assert k in data["camera"]["errors"]
        assert data["camera"]["errors"][k] >= 0.0


def test_build_report_no_errors_when_omitted():
    rep = report.build_report(
        target="camera", camera=(31, 2), bones=None, selected_bones=set(),
        ranges=[(0, 30)], keep_frames=[],
    )
    assert "errors" not in rep["camera"]
