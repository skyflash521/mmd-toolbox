import json

from sparsevmd import cli
from vmd import io
from vmd.types import BoneKey, CameraKey, VmdDocument, VmdWarning

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6
_CP932_LONE_LEAD_BYTE = b"\x81"


def _bone_linear():
    b = bytearray(64)
    for i in (0, 1, 2, 3, 4, 5, 6, 7, 17, 18):
        b[i] = 20
    for i in (8, 9, 10, 11, 12, 13, 14, 15):
        b[i] = 107
    return bytes(b)


BL = _bone_linear()


def cam(frame, center=(0.0, 0.0, 0.0)):
    return CameraKey(frame, -30.0, center, (0.0, 0.0, 0.0), CAM_LINEAR, 30, 0)


def bone(name, frame, pos=(0.0, 0.0, 0.0)):
    return BoneKey(name.encode("cp932").ljust(15, b"\x00"), frame, pos, (0.0, 0.0, 0.0, 1.0), BL)


def write_vmd(path, **sections):
    io.write_file(VmdDocument(**sections), str(path))


def linear_camera_doc():
    return [cam(f, center=(float(f), 0.0, 0.0)) for f in range(31)]


def ramp_bone_doc(name="センター"):
    return [bone(name, f, pos=(0.0, float(f), 0.0)) for f in range(31)]


def machine_events(capsysbinary):
    out = capsysbinary.readouterr().out
    return [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]


def terminal(events):
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    assert events[-1]["type"] in ("result", "error")
    return events[-1]


def test_machine_emits_reduce_result(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    rc = cli.main([str(src), "-o", str(out), "--curve-mode", "linear", "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = terminal(events)
    assert r["type"] == "result" and r["mode"] == "reduce"
    assert set(r) == {"type", "mode", "output", "target", "camera", "bone", "reduced"}
    assert r["output"] == str(out)
    assert r["target"] == "all"
    assert r["camera"] == {"input_keys": 31, "output_keys": len(io.read(str(out))[0].camera)}
    assert r["bone"] == {"input_keys": 31, "output_keys": len(io.read(str(out))[0].bone)}
    assert r["reduced"] is True
    assert out.exists()


def test_machine_reduce_result_target_camera_null_bone(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera",
                   "--curve-mode", "linear", "--machine"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    assert r["mode"] == "reduce" and r["target"] == "camera"
    assert r["camera"] is not None and r["bone"] is None


def test_machine_reduce_result_target_bone_null_camera(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "bone",
                   "--curve-mode", "linear", "--machine"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    assert r["mode"] == "reduce" and r["target"] == "bone"
    assert r["bone"] is not None and r["camera"] is None


def _assert_json_lines_lf(raw):
    assert raw.endswith(b"\n") and b"\r" not in raw
    for ln in raw.decode("utf-8").split("\n")[:-1]:
        assert ln != ""
        assert "type" in json.loads(ln)


def test_machine_stdout_json_lines_lf_all_modes(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    assert cli.main([str(src), "-o", str(tmp_path / "o.vmd"), "--curve-mode", "linear", "--machine"]) == 0
    _assert_json_lines_lf(capsysbinary.readouterr().out)
    assert cli.main([str(src), "--curve-mode", "linear", "--machine", "--dry-run"]) == 0
    _assert_json_lines_lf(capsysbinary.readouterr().out)
    assert cli.main([str(src), "--machine", "--list-bones"]) == 0
    _assert_json_lines_lf(capsysbinary.readouterr().out)


def test_machine_no_human_text_on_stdout_with_verbose(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera",
              "--curve-mode", "linear", "--machine", "--verbose"])
    text = capsysbinary.readouterr().out.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln]
    assert lines
    for ln in lines:
        json.loads(ln)


def test_machine_verbose_diagnostics_go_to_stderr_through_main(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    keys = [cam(f, center=((float(f) if f < 15 else float(f) + 50.0), 0.0, 0.0)) for f in range(31)]
    write_vmd(src, camera=keys)
    cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera",
              "--curve-mode", "linear", "--machine", "--verbose"])
    cap = capsysbinary.readouterr()
    out_text = cap.out.decode("utf-8")
    err_text = cap.err.decode("utf-8")
    lines = [ln for ln in out_text.split("\n") if ln]
    assert lines
    for ln in lines:
        json.loads(ln)
    assert "不連続検出位置" in err_text


def test_machine_emits_progress_camera_and_bone(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--curve-mode", "linear", "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    progress = [e for e in events if e["type"] == "progress"]
    assert progress
    for p in progress:
        assert set(p) == {"type", "stage", "done", "total", "note", "elapsed"}
        assert p["stage"] in ("camera", "bone")
        assert isinstance(p["elapsed"], float) and p["elapsed"] >= 0.0
    stages = {p["stage"] for p in progress}
    assert stages == {"camera", "bone"}
    for stage in ("camera", "bone"):
        starts = [p for p in progress if p["stage"] == stage and p["done"] == 0 and p["total"] is None]
        assert len(starts) == 1, f"{stage} 段の開始イベントは 1 本"
        assert starts[0]["note"] == "" and starts[0]["elapsed"] == 0.0
    bone_done = [p for p in progress if p["stage"] == "bone" and p["total"] is not None]
    assert bone_done and all(p["note"] == "センター" for p in bone_done)
    assert {p["total"] for p in bone_done} == {1}
    cam_progress = [p for p in progress if p["stage"] == "camera" and p["total"] is not None]
    assert cam_progress and all(isinstance(p["total"], int) and p["total"] > 0 for p in cam_progress)
    assert any(p["note"] for p in progress if p["stage"] == "camera")


def test_machine_camera_one_key_still_emits_camera_start(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=[cam(7, center=(3.0, 0.0, 0.0))])
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera", "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = terminal(events)
    assert r["mode"] == "reduce" and r["camera"] == {"input_keys": 1, "output_keys": 1}
    camera_starts = [e for e in events if e["type"] == "progress" and e["stage"] == "camera"
                     and e["done"] == 0 and e["total"] is None]
    assert len(camera_starts) == 1


def test_machine_progress_only_for_processed_stages(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera",
                   "--curve-mode", "linear", "--machine"])
    assert rc == 0
    stages = {e["stage"] for e in machine_events(capsysbinary) if e["type"] == "progress"}
    assert "camera" in stages and "bone" not in stages


def test_machine_emits_decode_error_warning_with_bone_section(tmp_path, capsysbinary):
    bad_name = (_CP932_LONE_LEAD_BYTE + b"\x20name").ljust(15, b"\x00")
    keys = [BoneKey(bad_name, f, (0.0, float(f), 0.0), (0.0, 0.0, 0.0, 1.0), BL) for f in range(4)]
    io.write_file(VmdDocument(bone=keys), str(tmp_path / "in.vmd"))
    rc = cli.main([str(tmp_path / "in.vmd"), "-o", str(tmp_path / "out.vmd"),
                   "--target", "bone", "--curve-mode", "linear", "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert terminal(events)["type"] == "result"
    decode = [e for e in events if e["type"] == "warning" and e["code"] == "decode-error"]
    assert len(decode) == 1
    assert set(decode[0]) == {"type", "code", "message", "section"}
    assert decode[0]["section"] == ["bone"]
    assert isinstance(decode[0]["message"], str) and decode[0]["message"]


def test_machine_selector_unmatched_warning_with_other_include_matching(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=ramp_bone_doc("頭"))
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "bone",
                   "--curve-mode", "linear", "--machine", "--bone", "頭", "--bone-glob", "幻*"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert terminal(events)["type"] == "result"
    unmatched = [e for e in events if e["type"] == "warning" and e["code"] == "selector_unmatched"]
    assert unmatched and set(unmatched[0]) == {"type", "code", "message", "section"}
    assert unmatched[0]["section"] is None and "幻*" in unmatched[0]["message"]


def test_machine_keep_frame_ignored_warning_before_result(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera",
                   "--curve-mode", "linear", "--machine", "--range", "0:10", "--keep-frame", "50"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = terminal(events)
    assert r["type"] == "result"
    ignored = [e for e in events if e["type"] == "warning" and e["code"] == "keep_frame_ignored"]
    assert ignored and set(ignored[0]) == {"type", "code", "message", "section"}
    assert ignored[0]["section"] is None and "50" in ignored[0]["message"]
    assert events.index(ignored[0]) < events.index(r)


def test_machine_dry_run_emits_inspect(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear",
                   "--machine", "--dry-run"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = terminal(events)
    assert r["type"] == "result" and r["mode"] == "inspect"
    assert set(r) == {"type", "mode", "output", "target", "sections", "keys", "frame_range",
                      "duration_sec", "ranges", "keep_frames", "reduced", "camera", "bones"}
    assert r["output"] is None and not out.exists()
    assert r["target"] == "camera"
    assert isinstance(r["sections"], list) and "camera" in r["sections"]
    assert r["keys"] == {"camera": 31, "bone": 0}
    assert r["frame_range"] == [0, 30]
    assert r["duration_sec"] == 30 / 30.0
    assert r["ranges"] == [[0, 30]]
    assert r["keep_frames"] == []
    assert r["reduced"] is True
    cam_r = r["camera"]
    assert set(cam_r) == {"input_keys", "output_keys", "errors", "cuts"}
    assert cam_r["input_keys"] == 31 and isinstance(cam_r["output_keys"], int)
    assert set(cam_r["errors"]) == {"pos_x", "pos_y", "pos_z", "rot_deg", "distance", "fov"}
    assert isinstance(cam_r["cuts"], list)
    assert r["bones"] is None


def test_machine_dry_run_inspect_target_all(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    rc = cli.main([str(src), "--curve-mode", "linear", "--machine", "--dry-run"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    assert r["mode"] == "inspect" and r["target"] == "all"
    assert r["camera"] is not None and set(r["camera"]) == {"input_keys", "output_keys", "errors", "cuts"}
    assert isinstance(r["bones"], list) and r["bones"]
    assert r["keys"] == {"camera": 31, "bone": 31}


def test_machine_dry_run_inspect_bones(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=ramp_bone_doc("センター"))
    rc = cli.main([str(src), "--target", "bone", "--curve-mode", "linear", "--machine", "--dry-run"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    assert r["mode"] == "inspect" and r["camera"] is None
    assert isinstance(r["bones"], list) and r["bones"]
    b0 = r["bones"][0]
    assert set(b0) == {"name", "selected", "input_keys", "output_keys", "errors", "cuts"}
    assert b0["name"] == "センター" and b0["selected"] is True
    assert set(b0["errors"]) == {"pos_x", "pos_y", "pos_z", "rot_deg"}
    assert isinstance(b0["cuts"], list)


def test_machine_dry_run_inspect_null_errors_nonselected_and_nonreducible(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=ramp_bone_doc("センター") + ramp_bone_doc("頭") + [bone("肩", 5)])
    rc = cli.main([str(src), "--target", "bone", "--curve-mode", "linear", "--machine", "--dry-run",
                   "--bone", "センター", "--bone", "肩"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    by_name = {b["name"]: b for b in r["bones"]}
    assert by_name["センター"]["selected"] is True and by_name["センター"]["errors"] is not None
    assert by_name["肩"]["selected"] is True
    assert by_name["肩"]["errors"] is None and by_name["肩"]["cuts"] is None
    assert by_name["頭"]["selected"] is False
    assert by_name["頭"]["errors"] is None and by_name["頭"]["cuts"] is None


def test_machine_dry_run_inspect_keep_frames_sorted_and_deduplicated(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "--target", "camera", "--curve-mode", "linear", "--machine", "--dry-run",
                   "--keep-frame", "20", "--keep-frame", "5", "--keep-frame", "20"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    assert r["keep_frames"] == [5, 20]


def test_machine_list_bones_result(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30), bone("頭", 0)])
    rc = cli.main([str(src), "-o", str(out), "--machine", "--list-bones", "--bone", "センター"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = terminal(events)
    assert r["type"] == "result" and r["mode"] == "list_bones"
    assert set(r) == {"type", "mode", "bones"}
    assert not out.exists()
    assert r["bones"] == [
        {"name": "センター", "keys": 2, "selected": True},
        {"name": "頭", "keys": 1, "selected": False},
    ]


def test_machine_list_bones_unresolved_warns_unmatched_then_unresolved_then_result(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("頭", 0), bone("頭", 30)])
    rc = cli.main([str(src), "--machine", "--list-bones", "--bone-glob", "幻*"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = terminal(events)
    assert r["type"] == "result" and r["mode"] == "list_bones"
    warns = [e for e in events if e["type"] == "warning"]
    codes = [w["code"] for w in warns]
    assert "selector_unmatched" in codes and "selection_unresolved" in codes
    assert codes.index("selector_unmatched") < codes.index("selection_unresolved")
    unresolved = next(w for w in warns if w["code"] == "selection_unresolved")
    assert set(unresolved) == {"type", "code", "message", "section"}
    assert unresolved["section"] is None
    assert isinstance(unresolved["message"], str) and unresolved["message"]
    result_idx = events.index(r)
    assert all(i < result_idx for i, e in enumerate(events) if e["type"] == "warning")
    assert all(b["selected"] is False for b in r["bones"])


def test_machine_reduce_result_bone_input_keys_count_non_selected_bones(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=ramp_bone_doc("センター") + ramp_bone_doc("頭"))
    rc = cli.main([str(src), "-o", str(out), "--target", "bone", "--curve-mode", "linear", "--machine",
                   "--bone", "センター"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    assert r["bone"] == {"input_keys": 62, "output_keys": 2 + 31}


def test_machine_reduce_result_not_reduced_when_range_misses_every_track(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera", "--curve-mode", "linear",
                   "--machine", "--range", "100:200"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    assert r["reduced"] is False
    assert r["camera"] == {"input_keys": 31, "output_keys": 31}


def test_machine_range_open_end_expands_to_selected_tracks_extent(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    long_head = [bone("頭", f, pos=(0.0, float(f), 0.0)) for f in range(101)]
    write_vmd(src, bone=ramp_bone_doc("センター") + long_head)
    rc = cli.main([str(src), "--target", "bone", "--curve-mode", "linear", "--machine", "--dry-run",
                   "--bone", "センター", "--range", "10:"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    assert r["ranges"] == [[10, 30]]


def test_machine_bone_selection_invalid_preceded_by_selector_unmatched(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=ramp_bone_doc("頭"))
    rc = cli.main([str(src), "--machine", "--target", "bone", "--bone-glob", "幻*"])
    assert rc == 2
    events = machine_events(capsysbinary)
    e = terminal(events)
    assert e["code"] == "bone_selection_invalid"
    unmatched = [w for w in events if w["type"] == "warning" and w["code"] == "selector_unmatched"]
    assert unmatched and events.index(unmatched[-1]) < events.index(e)


def test_machine_read_warning_without_section_has_null_section(tmp_path, capsysbinary, monkeypatch):
    cam_keys = linear_camera_doc()
    warning = VmdWarning(code="sections-missing", message="missing")
    monkeypatch.setattr(cli.io, "read", lambda path: (VmdDocument(camera=cam_keys), [warning]))
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=cam_keys)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera", "--curve-mode", "linear",
                   "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    missing = [e for e in events if e["type"] == "warning" and e["code"] == "sections-missing"]
    assert len(missing) == 1 and missing[0]["section"] is None


def test_machine_non_ascii_paths(tmp_path, capsysbinary):
    src = tmp_path / "入力モーション.vmd"
    out = tmp_path / "出力_疎.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "--machine"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    assert r["mode"] == "reduce" and r["output"] == str(out)
    assert out.exists()
