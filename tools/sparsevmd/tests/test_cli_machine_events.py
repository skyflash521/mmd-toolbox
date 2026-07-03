"""sparsevmd CLI 機械モードの成功経路イベントのテスト(sparsevmd.md §12.2)。

機械モードの成功経路: progress(camera/bone 2 段)・warning(読み込み透過・選択不一致・keep-frame 無視・
選択不能)・result(reduce / inspect / list_bones)。ストリームは result のちょうど 1 つで終端する。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。テスト方針は
../../../libs/vmd/vmd.md §4 に準ずる(決定論的・外部依存なし)。
"""

import json

from vmd import io
from vmd.types import BoneKey, CameraKey, VmdDocument
from sparsevmd import cli

CAM_LINEAR = bytes([20, 107, 20, 107]) * 6


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


# --- result(reduce)通常実行(§12.2)-----------------------------------------


def test_machine_emits_reduce_result(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    rc = cli.main([str(src), "-o", str(out), "--curve-mode", "linear", "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = terminal(events)
    assert r["type"] == "result" and r["mode"] == "reduce"
    # reduce の result は {output, target, camera, bone, reduced} のみ(§12.2)。
    assert set(r) == {"type", "mode", "output", "target", "camera", "bone", "reduced"}
    assert r["output"] == str(out)
    assert r["target"] == "all"
    assert r["camera"] == {"input_keys": 31, "output_keys": len(io.read(str(out))[0].camera)}
    assert r["bone"] == {"input_keys": 31, "output_keys": len(io.read(str(out))[0].bone)}
    assert r["reduced"] is True
    assert out.exists()


def test_machine_reduce_result_target_camera_null_bone(tmp_path, capsysbinary):
    # --target camera → bone は null。
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


# --- チャネル固定(JSON Lines・LF・人間向けテキスト非混入)---------------------


def _assert_json_lines_lf(raw):
    # 末尾 LF あり・\r 不在・末尾 LF より前の各行は空でない単一 JSON(§9 項目1)。
    assert raw.endswith(b"\n") and b"\r" not in raw
    for ln in raw.decode("utf-8").split("\n")[:-1]:
        assert ln != ""
        assert "type" in json.loads(ln)


def test_machine_stdout_json_lines_lf_all_modes(tmp_path, capsysbinary):
    # 通常実行・dry-run(inspect)・list-bones のどのモードでも stdout は LF のみの有効な JSON Lines(§9)。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    assert cli.main([str(src), "-o", str(tmp_path / "o.vmd"), "--curve-mode", "linear", "--machine"]) == 0
    _assert_json_lines_lf(capsysbinary.readouterr().out)
    assert cli.main([str(src), "--curve-mode", "linear", "--machine", "--dry-run"]) == 0
    _assert_json_lines_lf(capsysbinary.readouterr().out)
    assert cli.main([str(src), "--machine", "--list-bones"]) == 0
    _assert_json_lines_lf(capsysbinary.readouterr().out)


def test_machine_no_human_text_on_stdout_with_verbose(tmp_path, capsysbinary):
    # --machine --verbose でも標準出力はイベント専用(人間向けレポートは混入しない)。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera",
              "--curve-mode", "linear", "--machine", "--verbose"])
    text = capsysbinary.readouterr().out.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln]
    assert lines
    for ln in lines:
        json.loads(ln)


# --- progress(camera/bone 2 段)(§12.2)------------------------------------


def test_machine_emits_progress_camera_and_bone(tmp_path, capsysbinary):
    # camera 段・bone 段の progress が出る。各段は開始時に done=0/total=null/note=""/elapsed=0.0 を 1 本。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--curve-mode", "linear", "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    progress = [e for e in events if e["type"] == "progress"]
    assert progress
    for p in progress:
        # progress は {type, stage, done, total, note, elapsed} のちょうど 6 キー(§12.2)。
        assert set(p) == {"type", "stage", "done", "total", "note", "elapsed"}
        assert p["stage"] in ("camera", "bone")
        assert isinstance(p["elapsed"], float) and p["elapsed"] >= 0.0
    stages = {p["stage"] for p in progress}
    assert stages == {"camera", "bone"}  # camera/bone の 2 段のみ
    for stage in ("camera", "bone"):
        starts = [p for p in progress if p["stage"] == stage and p["done"] == 0 and p["total"] is None]
        assert len(starts) == 1, f"{stage} 段の開始イベントは 1 本"
        assert starts[0]["note"] == "" and starts[0]["elapsed"] == 0.0
    # bone 段の完了イベントは note=ボーン名・total=対象トラック数。
    bone_done = [p for p in progress if p["stage"] == "bone" and p["total"] is not None]
    assert bone_done and all(p["note"] for p in bone_done)
    assert {p["total"] for p in bone_done} == {1}  # 対象トラック 1 本(センター)
    # camera 段は開始イベントだけでなくフレーム進捗(total=全範囲フレーム数≠null)も出す(§12.2)。
    cam_progress = [p for p in progress if p["stage"] == "camera" and p["total"] is not None]
    assert cam_progress and all(isinstance(p["total"], int) and p["total"] > 0 for p in cam_progress)
    # camera 段は出力後検証区間で補足文字列(note)を付ける(§12.2 / §9 項目15)。note が落ちていないこと。
    assert any(p["note"] for p in progress if p["stage"] == "camera")


def test_machine_camera_one_key_still_emits_camera_start(tmp_path, capsysbinary):
    # カメラ1キー(削減不能)でも --target camera なら camera 段は処理対象なので start イベントを 1 本出す(§12.2)。
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


def test_machine_progress_only_processed_stages(tmp_path, capsysbinary):
    # --target camera はカメラ段のみ。bone 段の progress は出さない(処理しない段はイベントを出さない)。
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=ramp_bone_doc())
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--target", "camera",
                   "--curve-mode", "linear", "--machine"])
    assert rc == 0
    stages = {e["stage"] for e in machine_events(capsysbinary) if e["type"] == "progress"}
    assert "camera" in stages and "bone" not in stages


# --- warning 透過・選択不一致・keep-frame 無視(§12.2)------------------------


def test_machine_emits_decode_error_warning(tmp_path, capsysbinary):
    # デコード不能なボーン名 → vmd.io の decode-error を warning へ透過(section 単一要素配列)。
    bad_name = b"\x81\x20name".ljust(15, b"\x00")
    keys = [BoneKey(bad_name, f, (0.0, float(f), 0.0), (0.0, 0.0, 0.0, 1.0), BL) for f in range(4)]
    io.write_file(VmdDocument(bone=keys), str(tmp_path / "in.vmd"))
    rc = cli.main([str(tmp_path / "in.vmd"), "-o", str(tmp_path / "out.vmd"),
                   "--target", "bone", "--curve-mode", "linear", "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert terminal(events)["type"] == "result"
    decode = [e for e in events if e["type"] == "warning" and e["code"] == "decode-error"]
    assert len(decode) == 1
    assert set(decode[0]) == {"type", "code", "message", "section"}  # warning の形(§12.2)
    # bone 名のデコードエラーなので section は単一要素 ["bone"](VmdWarning.section 透過。§12.2)。
    assert decode[0]["section"] == ["bone"]
    assert isinstance(decode[0]["message"], str) and decode[0]["message"]


def test_machine_selector_unmatched_warning(tmp_path, capsysbinary):
    # 一致する include があり、別 include の glob が不一致 → selector_unmatched 警告 + 正常終了。
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


def test_machine_keep_frame_ignored_warning(tmp_path, capsysbinary):
    # 全削減範囲外の keep-frame → keep_frame_ignored 警告。
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
    assert events.index(ignored[0]) < events.index(r)  # warning は result より前


# --- inspect(--machine --dry-run)(§12.2)----------------------------------


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
    # inspect の result のトップレベル形を固定(§12.2)。
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
    assert r["bones"] is None  # target camera


def test_machine_dry_run_inspect_target_all(tmp_path, capsysbinary):
    # --target all の inspect は camera と bones の両方を載せる(§9 項目11)。
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
    # --target bone の inspect は bones 配列(name/selected/input_keys/output_keys/errors/cuts)。camera は null。
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
    # errors/cuts が null になるのは「非選択」と「選択でも削減不能(1キー)」の両方(§12.2)。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=ramp_bone_doc("センター") + ramp_bone_doc("頭") + [bone("肩", 5)])
    rc = cli.main([str(src), "--target", "bone", "--curve-mode", "linear", "--machine", "--dry-run",
                   "--bone", "センター", "--bone", "肩"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    by_name = {b["name"]: b for b in r["bones"]}
    # 選択+削減可能: errors 非 null。
    assert by_name["センター"]["selected"] is True and by_name["センター"]["errors"] is not None
    # 選択でも 1 キー(削減不能): errors/cuts null。
    assert by_name["肩"]["selected"] is True
    assert by_name["肩"]["errors"] is None and by_name["肩"]["cuts"] is None
    # 非選択: errors/cuts null。
    assert by_name["頭"]["selected"] is False
    assert by_name["頭"]["errors"] is None and by_name["頭"]["cuts"] is None


# --- list_bones(--machine --list-bones)(§12.2)----------------------------


def test_machine_list_bones_result(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30), bone("頭", 0)])
    rc = cli.main([str(src), "-o", str(out), "--machine", "--list-bones", "--bone", "センター"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = terminal(events)
    assert r["type"] == "result" and r["mode"] == "list_bones"
    assert set(r) == {"type", "mode", "bones"}  # list_bones の result は bones のみ(§12.2)
    assert not out.exists()
    assert r["bones"] == [
        {"name": "センター", "keys": 2, "selected": True},
        {"name": "頭", "keys": 1, "selected": False},
    ]


def test_machine_list_bones_unresolved_warns_then_result(tmp_path, capsysbinary):
    # 選択子解決不能(唯一の include が不一致 glob)でも一覧(全件非選択)で終了コード 0。
    # selector_unmatched(蓄積分)と selection_unresolved(理由)の warning を result より先に出す。
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
    # selector_unmatched(蓄積分)は selection_unresolved(理由)より先(§12.2)。
    assert codes.index("selector_unmatched") < codes.index("selection_unresolved")
    # selection_unresolved の形は {code, message, section}、section は null(§12.2)。
    unresolved = next(w for w in warns if w["code"] == "selection_unresolved")
    assert set(unresolved) == {"type", "code", "message", "section"}
    assert unresolved["section"] is None
    assert isinstance(unresolved["message"], str) and unresolved["message"]
    # warning はすべて result より前に出る。
    result_idx = events.index(r)
    assert all(i < result_idx for i, e in enumerate(events) if e["type"] == "warning")
    # 全件非選択。
    assert all(b["selected"] is False for b in r["bones"])


# --- 移植性(規約 §10)------------------------------------------------------


def test_machine_non_ascii_paths(tmp_path, capsysbinary):
    # 非 ASCII(日本語)の入出力パスで動作する。
    src = tmp_path / "入力モーション.vmd"
    out = tmp_path / "出力_疎.vmd"
    write_vmd(src, camera=linear_camera_doc())
    rc = cli.main([str(src), "-o", str(out), "--target", "camera", "--curve-mode", "linear", "--machine"])
    assert rc == 0
    r = terminal(machine_events(capsysbinary))
    assert r["mode"] == "reduce" and r["output"] == str(out)
    assert out.exists()
