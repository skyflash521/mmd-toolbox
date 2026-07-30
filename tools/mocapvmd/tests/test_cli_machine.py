"""mocapvmd CLI 機械モード(--machine)のテスト。

機械モードは stdout を JSON Lines のイベント専用にし、progress / warning / result / error を出す。
ストリームは result または error のちょうど 1 つで終端する。構造化エラーは確定 code/field/exit_code の
error イベントで終端し、非機械モードは理由を標準エラーへ 1 行出す。既定(非機械)挙動が不変であること
(後方互換)も併せて検証する。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。テストは
決定論的に実行し、外部依存を使わない。--describe は独立メタ操作として別ステップで
扱う(本モジュールは含めない)。
"""

import json

import pytest

from mocapvmd import cli, presets
from mocapvmd.model_profile import STANDARD_BONE_NAMES
from vmd import io

from .helpers import bone, build_standard_pmx, write_vmd


def machine_events(capsysbinary):
    """capsysbinary で捕捉した stdout を JSON Lines として解析しイベント配列で返す。"""
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")  # UTF-8 固定(ロケール非依存)を前提に decode
    return [json.loads(ln) for ln in text.split("\n") if ln]


def machine_error(capsysbinary):
    """機械モードの stdout を解析し、終端の error イベントを返す(失敗は error で終端)。"""
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1  # 終端はちょうど1つ
    return events[-1]


def _ramp_doc(path):
    # センターの直線ランプ(密11フレーム)。疎化で端キーへ削減される。
    write_vmd(path, bone=[bone("センター", f, pos=(float(f), 0.0, 0.0)) for f in range(11)])


def _foot_doc(path):
    # 右足ＩＫ(接地中の遅いドリフト)。足IK安定化段が走る。
    write_vmd(path, bone=[bone("右足ＩＫ", f, pos=(round(0.05 * f, 6), 0.0, 0.0)) for f in range(11)])


# --- 正常系: process result / JSON Lines / チャネル固定 ---------------------


def test_machine_emits_process_result(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    r = events[-1]
    assert r["mode"] == "process"
    assert r["output"] == str(out)
    assert r["input_keys"] == 11                 # 入力ボーンキー総数
    assert isinstance(r["output_keys"], int) and r["output_keys"] >= 1
    assert out.exists()


def test_machine_stdout_is_valid_json_lines(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    text = raw.decode("utf-8")
    assert text.endswith("\n")
    for ln in text.split("\n")[:-1]:
        assert ln != ""            # 空行を挟まない
        obj = json.loads(ln)       # 各行が単一 JSON
        assert "type" in obj


def test_machine_stdout_lf_only_no_cr(tmp_path, capsysbinary):
    # 行区切りは LF 固定で \r を一切含まない(CRLF 変換なし。バイト列で検証)。
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n")
    assert b"\r" not in raw


def test_machine_no_human_text_on_stdout(tmp_path, capsysbinary):
    # 機械モードの stdout は人間向けテキストを含まない(チャネル固定)。--verbose 併用でも JSON のみ。
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine", "--verbose"])
    text = capsysbinary.readouterr().out.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln]
    assert lines
    for ln in lines:
        json.loads(ln)  # すべて JSON、人間向けレポート行は混入しない


# --- progress イベント(3 段) --------------------------------------------


def test_machine_emits_progress_stages(tmp_path, capsysbinary):
    # 既定(denoise on / foot_ik on / reduce on)で denoise・foot_ik・reduce の 3 段の progress が出る。
    src = tmp_path / "in.vmd"
    _foot_doc(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "process"
    progress = [e for e in events if e["type"] == "progress"]
    assert progress
    stages = {p["stage"] for p in progress}
    assert {"denoise", "foot_ik", "reduce"} <= stages
    for p in progress:
        assert set(p) >= {"type", "stage", "done", "total", "note", "elapsed"}
    # 各段の開始イベントは done=0, total=null, note:"", elapsed:0.0 の固定形が 1 本ある。
    for stage in ("denoise", "foot_ik", "reduce"):
        starts = [p for p in progress if p["stage"] == stage and p["done"] == 0 and p["total"] is None]
        assert len(starts) == 1, f"{stage} 段の開始イベントは 1 本"
        assert starts[0]["note"] == "" and starts[0]["elapsed"] == 0.0


def test_machine_disabled_stages_emit_no_progress(tmp_path, capsysbinary):
    # 無効化した段(--no-denoise / --no-foot-ik-stabilize)は progress を出さない。
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine",
                   "--no-denoise", "--no-foot-ik-stabilize"])
    assert rc == 0
    stages = {e["stage"] for e in machine_events(capsysbinary) if e["type"] == "progress"}
    assert "denoise" not in stages and "foot_ik" not in stages
    assert "reduce" in stages


# --- warning 透過 -----------------------------------------------------------


def test_machine_emits_decode_error_warning(tmp_path, capsysbinary):
    # デコード不能なボーン名を含む入力 → vmd.io の decode-error 警告を warning イベントへ透過。
    # 不正な cp932 シーケンスを名前フィールドに埋めた密トラックを書く。
    from vmd.reduce import BONE_LINEAR_INTERP
    from vmd.types import BoneKey, VmdDocument
    bad_name = b"\x81\x20name".ljust(15, b"\x00")  # cp932 で復号できないバイト列
    keys = [BoneKey(bad_name, f, (float(f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)
            for f in range(4)]
    io.write_file(VmdDocument(bone=keys), str(tmp_path / "in.vmd"))
    rc = cli.main([str(tmp_path / "in.vmd"), "-o", str(tmp_path / "out.vmd"), "--machine", "--no-reduce"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result"
    warns = [e for e in events if e["type"] == "warning"]
    w = next(w for w in warns if w["code"] == "decode-error")
    assert isinstance(w["section"], list) and len(w["section"]) == 1   # 単一要素配列
    assert isinstance(w["message"], str) and w["message"]


def test_machine_warning_dedup_matches_human(tmp_path, capsysbinary):
    # 同一(code, section, message)の警告は 1 件へ集約する(人間向け経路と同じ基準)。
    from vmd.reduce import BONE_LINEAR_INTERP
    from vmd.types import BoneKey, VmdDocument
    bad_name = b"\x81\x20name".ljust(15, b"\x00")
    # 同名のデコード不能キーを複数フレーム持たせても decode-error は 1 件へ集約される。
    keys = [BoneKey(bad_name, f, (float(f), 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), BONE_LINEAR_INTERP)
            for f in range(6)]
    io.write_file(VmdDocument(bone=keys), str(tmp_path / "in.vmd"))
    rc = cli.main([str(tmp_path / "in.vmd"), "-o", str(tmp_path / "out.vmd"), "--machine", "--no-reduce"])
    assert rc == 0
    warns = [e for e in machine_events(capsysbinary) if e["type"] == "warning"]
    decode = [w for w in warns if w["code"] == "decode-error"]
    assert len(decode) == 1


# --- 構造化エラー -----------------------------------------------------------


def test_machine_error_bad_argument_unknown_option(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "--machine", "--bogus"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2
    assert e["field"] == "--bogus"   # unrecognized arguments: の先頭トークン
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_bad_argument_missing_input(capsysbinary):
    # positional input 欠落 → bad_argument、field は input。
    rc = cli.main(["--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "input"


def test_machine_error_bad_argument_type_error_field(tmp_path, capsysbinary):
    # 型エラー(--clean-strength 非数値)→ argparse 検出の bad_argument、field は長形式。
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "--machine", "--clean-strength", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--clean-strength"


@pytest.mark.parametrize("opt,val", [
    ("--clean-strength", "-0.1"),
    ("--clean-strength", "nan"),
    ("--foot-slide-suppression", "1.5"),
    ("--foot-slide-suppression", "nan"),
    ("--reduce-error-bone-pos", "-1"),
    ("--reduce-error-bone-rot", "inf"),
])
def test_machine_error_bad_argument_value_validation(tmp_path, capsysbinary, opt, val):
    # 解析後の値検証(範囲外・非有限・負)も bad_argument(該当オプションの field)。
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "--machine", opt, val])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == opt and e["exit_code"] == 2


def test_machine_error_input_not_file(tmp_path, capsysbinary):
    rc = cli.main([str(tmp_path / "nope.vmd"), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "input_not_file" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_pmx_not_file(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "--machine", "--denoise-mode", "pose", "--pmx", str(tmp_path / "nope.pmx")])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "pmx_not_file" and e["field"] == "--pmx" and e["exit_code"] == 1


def test_machine_error_output_exists(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "-o", str(src), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output"


def test_machine_error_output_exists_distinct_path(tmp_path, capsysbinary):
    # 入力と別パスの既存出力も機械モードで output_exists を返すこと。
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    out = tmp_path / "out.vmd"
    out.write_bytes(b"old content")
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_exists" and e["field"] == "--output"


def test_machine_error_not_vmd(tmp_path, capsysbinary):
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file at all")
    rc = cli.main([str(bad), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "not_vmd" and e["field"] == "input" and e["exit_code"] == 1
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_invalid_bone_values(tmp_path, capsysbinary):
    # 非有限ボーン値 → invalid_bone_values(exit 1)、field は input。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0, pos=(float("nan"), 0.0, 0.0)), bone("センター", 1)])
    rc = cli.main([str(src), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "invalid_bone_values" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_not_pmx(tmp_path, capsysbinary):
    # PMX 形式不正(PmxFormatError)→ not_pmx(exit 1)、field は --pmx。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 10, pos=(0.3, 0.0, 0.0))])
    pmx = tmp_path / "bad.pmx"
    pmx.write_bytes(b"NOTPMX")
    rc = cli.main([str(src), "--machine", "--denoise-mode", "pose", "--pmx", str(pmx)])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "not_pmx" and e["field"] == "--pmx" and e["exit_code"] == 1


def test_machine_error_model_profile_invalid(tmp_path, capsysbinary):
    # 必須標準ボーン欠落(MocapModelProfileError)→ model_profile_invalid(exit 1)、pmx 指定時 field は --pmx。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 10, pos=(0.3, 0.0, 0.0))])
    pmx = tmp_path / "model.pmx"
    names = [n for n in STANDARD_BONE_NAMES.values() if n != STANDARD_BONE_NAMES["wrist_r"]]
    pmx.write_bytes(build_standard_pmx(names))
    rc = cli.main([str(src), "--machine", "--denoise-mode", "pose", "--pmx", str(pmx)])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "model_profile_invalid" and e["field"] == "--pmx" and e["exit_code"] == 1


def test_machine_error_write_failed(tmp_path, capsysbinary):
    # 出力の I/O 失敗(親がファイル)→ write_failed(exit 3)、field は --output、path 付き。
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    clash = tmp_path / "afile"
    clash.write_bytes(b"x")
    out = str(clash / "out.vmd")
    rc = cli.main([str(src), "-o", out, "--machine"])
    assert rc == 3
    e = machine_error(capsysbinary)
    assert e["code"] == "write_failed" and e["field"] == "--output" and e["exit_code"] == 3
    assert e["path"] == out


def test_machine_error_internal_error(tmp_path, capsysbinary, monkeypatch):
    # 想定外の内部例外(reduce が RuntimeError)→ internal_error(exit 1)。安全網。
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(cli.reduce, "reduce_bones", boom)
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "internal_error" and e["exit_code"] == 1


def test_non_machine_error_prints_reason_to_stderr(tmp_path, capsys):
    # 非機械モードでも失敗理由を標準エラーへ 1 行出す。終了コードは維持、stdout に JSON は出さない。
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file")
    rc = cli.main([str(bad)])
    assert rc == 1
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")


# --- 入力検査 --machine --dry-run(mode:"inspect") -------------------------


def test_machine_dry_run_emits_inspect_result(tmp_path, capsysbinary):
    # --machine --dry-run は VMD を書かず inspect result を出す。
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "-o", str(out), "--machine", "--dry-run"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = events[-1]
    assert r["type"] == "result" and r["mode"] == "inspect"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    assert r["output"] is None and not out.exists()
    assert r["input_kind"] == "bone"
    assert r["keys"] == 11
    assert r["frame_range"] == [0, 10]
    assert r["duration_sec"] == 10 / 30.0
    assert isinstance(r["sections"], list) and "bone" in r["sections"]
    # 実行計画(解決済み引数値)。
    assert r["preset"] == "medium" and r["clean_strength"] == 1.0
    assert r["denoise"] is True and r["denoise_mode"] == "bone"
    assert r["foot_ik_stabilize"] is True and r["foot_slide_suppression"] == 1.0
    assert r["curve_mode"] == "bezier" and r["reduce"] is True
    # bones は初出順の {name, category, keys, frame_range}。
    assert isinstance(r["bones"], list) and r["bones"]
    b0 = r["bones"][0]
    assert set(b0) == {"name", "category", "keys", "frame_range"}
    assert b0["name"] == "センター" and b0["category"] == "center"
    assert b0["keys"] == 11 and b0["frame_range"] == [0, 10]
    # reduction 診断(--reduce 既定 on)。
    assert isinstance(r["reduction"], dict) and "センター" in r["reduction"]
    red = r["reduction"]["センター"]
    assert set(red) == {"input_keys", "output_keys", "tol_pos", "tol_rot", "cuts", "errors"}
    assert set(red["errors"]) == {"pos_x", "pos_y", "pos_z", "rot_deg"}
    # pose_denoise は bone モードなので null。
    assert r["pose_denoise"] is None


def test_machine_dry_run_inspect_no_reduce_null_reduction(tmp_path, capsysbinary):
    # --no-reduce のとき reduction は null。
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "--machine", "--dry-run", "--no-reduce"])
    assert rc == 0
    r = machine_events(capsysbinary)[-1]
    assert r["mode"] == "inspect" and r["reduce"] is False and r["reduction"] is None


def test_machine_dry_run_inspect_pose_denoise_payload(tmp_path, capsysbinary):
    # pose モードの inspect は curated pose_denoise 診断を載せる(既定モデルプロファイル)。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[
        bone("センター", 0), bone("センター", 10, pos=(0.3, 0.0, 0.0)),
        bone("頭", 0), bone("頭", 10, rot=(0.0, 0.0, 0.05, 0.99875)),
    ])
    rc = cli.main([str(src), "--machine", "--dry-run", "--denoise-mode", "pose"])
    assert rc == 0
    r = machine_events(capsysbinary)[-1]
    assert r["mode"] == "inspect"
    pd = r["pose_denoise"]
    assert set(pd) == {"pmx", "frames", "markers", "marker_displacement", "fit"}
    assert set(pd["markers"]) == {"available", "required_bones_ok"}
    assert set(pd["marker_displacement"]) == {"max", "mean"}
    assert set(pd["fit"]) == {"mean_error_before", "mean_error_after", "fallback_frames",
                              "max_bone_delta_deg", "max_center_delta"}
    assert pd["pmx"] is None  # 既定モデルプロファイル


# --- ボーン一覧 --machine --list-bones(mode:"list_bones") ------------------


def test_machine_list_bones_result(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("右足ＩＫ", 0), bone("センター", 1)])
    out = tmp_path / "out.vmd"
    rc = cli.main([str(src), "-o", str(out), "--machine", "--list-bones"])
    assert rc == 0
    events = machine_events(capsysbinary)
    r = events[-1]
    assert r["type"] == "result" and r["mode"] == "list_bones"
    assert not out.exists()
    names = [b["name"] for b in r["bones"]]
    assert names == ["センター", "右足ＩＫ"]  # 初出順・名前ごと
    for b in r["bones"]:
        assert set(b) == {"name", "category"}
    cats = {b["name"]: b["category"] for b in r["bones"]}
    assert cats["センター"] == "center" and cats["右足ＩＫ"] == "foot_ik"


def test_machine_list_bones_not_blocked_by_invalid_values(tmp_path, capsysbinary):
    # --list-bones は診断モード。非有限ボーン値でも弾かれず一覧を返す(処理固有検証を迂回)。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0, pos=(float("nan"), 0.0, 0.0))])
    rc = cli.main([str(src), "--machine", "--list-bones"])
    assert rc == 0
    r = machine_events(capsysbinary)[-1]
    assert r["mode"] == "list_bones"


# --- メタ操作の例外(--help / --version) ---------------------------------


def test_machine_version_stays_human(capsys):
    # --machine 併用でも --version は人間向けテキスト+exit 0、イベントに載せない。
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "mocapvmd" in out and not out.lstrip().startswith("{")


def test_machine_help_stays_human(capsys):
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


def test_help_lists_machine_flag(capsys):
    # --help に新設フラグ(--machine / --reduce / --verbose)が現れる(人間向けヘルプ)。
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "--machine" in text and "--reduce" in text and "--verbose" in text


# --- 中断 -------------------------------------------------------------------


def _raise_keyboard_interrupt(*a, **k):
    raise KeyboardInterrupt()


def test_machine_cancelled_on_keyboard_interrupt(tmp_path, capsysbinary, monkeypatch):
    # 計算中の KeyboardInterrupt → cancelled の error イベント・exit 130。出力は書かれない(原子性)。
    monkeypatch.setattr(cli.reduce, "reduce_bones", _raise_keyboard_interrupt)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "-o", str(out), "--machine"])
    assert rc == 130
    e = machine_error(capsysbinary)
    assert e["code"] == "cancelled" and e["exit_code"] == 130 and e["field"] is None
    assert not out.exists()


def test_non_machine_cancelled_on_keyboard_interrupt(tmp_path, capsys, monkeypatch):
    # 非機械モードの中断は stdout に JSON を出さず理由を標準エラーへ 1 行、exit 130。出力は書かれない。
    monkeypatch.setattr(cli.reduce, "reduce_bones", _raise_keyboard_interrupt)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _ramp_doc(src)
    rc = cli.main([str(src), "-o", str(out)])
    assert rc == 130
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")
    assert not out.exists()


# --- 自己記述 --describe -----------------------------------------------------


def describe_result(capsysbinary):
    """--describe の stdout を解析し、単一の result(mode:"describe")イベントを返す。"""
    events = machine_events(capsysbinary)
    assert len(events) == 1 and events[0]["type"] == "result" and events[0]["mode"] == "describe"
    return events[0]


def test_describe_emits_result_without_input(capsysbinary):
    # --describe は入力を要求せず、VMD を読まずに options/presets の result を出して exit 0。
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    assert isinstance(r["options"], list) and r["options"]
    assert isinstance(r["presets"], list)
    # process/inspect/list_bones のキーは describe には載せない。
    for k in ("output", "keys", "bones", "reduction", "input_kind"):
        assert k not in r


def test_describe_works_without_machine_flag(capsysbinary):
    # --describe は --machine を要さない独立メタ操作(--machine 無しでも構造化 result を出す)。
    rc = cli.main(["--describe"])
    assert rc == 0
    assert describe_result(capsysbinary)["mode"] == "describe"


def test_describe_options_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    by_name = {o["name"]: o for o in r["options"]}
    # メタ/モード操作は options に含めない。
    for meta in ("--describe", "--version", "--help", "--machine"):
        assert meta not in by_name
    # 否定形は重複列挙しない(肯定形の長形式のみ)。
    for neg in ("--no-denoise", "--no-foot-ik-stabilize", "--no-reduce"):
        assert neg not in by_name
    # 全 18 要素・各要素は常に 5 キー・help は非空文字列。
    assert len(r["options"]) == 18
    for o in r["options"]:
        assert set(o) == {"name", "type", "constraint", "default", "help"}
        assert isinstance(o["help"], str) and o["help"]
    # 18 個の全要素を {type, constraint, default} で固定する(float の constraint は
    # {min,max,exclusive_min} 3キー・enum は {choices}・flag/str は null)。
    expected = {
        "input": ("str", None, None),
        "--output": ("str", None, None),
        "--overwrite": ("flag", None, False),
        "--preset": ("enum", {"choices": list(presets.PRESET_NAMES)}, "medium"),
        "--clean-strength": ("float", {"min": 0, "max": None, "exclusive_min": False}, 1.0),
        "--denoise": ("flag", None, True),
        "--denoise-mode": ("enum", {"choices": ["bone", "pose"]}, "bone"),
        "--pmx": ("str", None, None),
        "--foot-ik-stabilize": ("flag", None, True),
        "--foot-slide-suppression": ("float", {"min": 0, "max": 1, "exclusive_min": False}, 1.0),
        "--reduce-error-bone-pos": ("float", {"min": 0, "max": None, "exclusive_min": False}, None),
        "--reduce-error-bone-rot": ("float", {"min": 0, "max": None, "exclusive_min": False}, None),
        "--curve-mode": ("enum", {"choices": ["bezier", "linear"]}, "bezier"),
        "--reduce": ("flag", None, True),
        "--list-bones": ("flag", None, False),
        "--dry-run": ("flag", None, False),
        "--quiet": ("flag", None, False),
        "--verbose": ("flag", None, False),
    }
    assert set(by_name) == set(expected)
    for name, (type_, constraint, default) in expected.items():
        o = by_name[name]
        assert o["type"] == type_, name
        assert o["constraint"] == constraint, name
        assert o["default"] == default, name


def test_describe_presets_shape_and_values(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    r = describe_result(capsysbinary)
    for p in r["presets"]:
        assert set(p) == {"name", "values"}
        assert set(p["values"]) == {"reduce_error_bone_pos", "reduce_error_bone_rot"}
    # 5 プリセットの基準位置許容・基準回転許容(プリセット基準値)を全件固定する。
    expected = {
        "slower": {"reduce_error_bone_pos": 0.05, "reduce_error_bone_rot": 0.40},
        "slow": {"reduce_error_bone_pos": 0.10, "reduce_error_bone_rot": 0.75},
        "medium": {"reduce_error_bone_pos": 0.20, "reduce_error_bone_rot": 1.50},
        "fast": {"reduce_error_bone_pos": 0.80, "reduce_error_bone_rot": 6.0},
        "faster": {"reduce_error_bone_pos": 1.60, "reduce_error_bone_rot": 12.0},
    }
    assert {p["name"]: p["values"] for p in r["presets"]} == expected


def test_describe_type_table_covers_non_meta_args():
    # _D_TYPE はメタ/モード操作を除く全 parser 引数を覆う。parser に引数を足して _D_TYPE への追加を
    # 忘れると describe から黙って抜けるため、その載せ忘れをここで検出する。
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._D_TYPE)


def test_describe_mode_arg_error_is_error_event(capsysbinary):
    # --describe(--machine 無し)も構造化出力モードなので、引数エラーは標準エラーでなく error
    # イベントでストリームを終端する。
    rc = cli.main(["--describe", "--clean-strength", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--clean-strength" and e["exit_code"] == 2


@pytest.mark.xfail(reason="impl pending: 出力先が既存ディレクトリのときの output_is_directory が未実装")
def test_machine_error_output_is_directory(tmp_path, capsysbinary):
    # 出力先が既存ディレクトリ → output_is_directory(exit 2)。ディレクトリは --overwrite でも
    # 書けないので、併用しても同じコードで拒否する(上書きの許可を促す案内へ落とさない)。
    src = tmp_path / "in.vmd"
    _ramp_doc(src)
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    for extra in ([], ["--overwrite"]):
        rc = cli.main([str(src), "-o", str(outdir), "--machine", *extra])
        assert rc == 2
        e = machine_error(capsysbinary)
        assert e["code"] == "output_is_directory" and e["field"] == "--output"
        assert e["exit_code"] == 2 and e["path"] == str(outdir)


def test_machine_list_bones_ignores_output_is_directory(tmp_path, capsysbinary):
    # --list-bones は出力を書かないので、出力先がディレクトリでも一覧を返す(検査の対象外)。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0)])
    outdir = tmp_path / "outdir"
    outdir.mkdir()
    rc = cli.main([str(src), "-o", str(outdir), "--machine", "--list-bones"])
    assert rc == 0
    assert machine_events(capsysbinary)[-1]["mode"] == "list_bones"
