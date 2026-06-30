"""shakevmd CLI 機械モードのテスト(shakevmd.md §2.8, §12)。

機械モード(--machine)の成功パスを検証する: stdout を JSON Lines のイベント専用にし、result/
warning/progress イベントを出す。終端は result または error のちょうど 1 つ(本ステップは成功=result。
error イベントは構造化エラーのステップで扱う)。非機械モードの既定挙動は不変であること(後方互換)。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。
本ステップ未実装の機械モード挙動は xfail(reason="impl pending: Step2 machine-mode events")で印を付け、
実装ステップで印を外す。テスト方針は ../../../libs/vmd/vmd.md §4 に準ずる。
"""

import json

import pytest

from shakevmd import cli
from vmd import io
from vmd.types import BoneKey, CameraKey, MorphKey, VmdDocument

LINEAR = bytes([20, 107, 20, 107]) * 6

_MACHINE_PENDING = "impl pending: Step2 machine-mode events"


def cam(frame, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0), fov=30, persp=0):
    return CameraKey(frame, dist, center, rot, LINEAR, fov, persp)


KEYS = [
    cam(0, dist=-30.0, center=(0.0, 0.0, 0.0), rot=(0.0, 0.0, 0.0)),
    cam(30, dist=-25.0, center=(10.0, 5.0, 2.0), rot=(0.2, 0.1, 0.0)),
    cam(60, dist=-20.0, center=(20.0, 0.0, -3.0), rot=(-0.1, 0.3, 0.05)),
]


def write_input(path, keys=KEYS, **doc_kwargs):
    io.write_file(VmdDocument(camera=list(keys), **doc_kwargs), str(path))
    return str(path)


def machine_events(capsysbinary):
    """capsysbinary で捕捉した stdout を JSON Lines として解析しイベント配列で返す。"""
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")  # UTF-8 固定(ロケール非依存)を前提に decode
    return [json.loads(ln) for ln in text.split("\n") if ln]


@pytest.mark.xfail(reason=_MACHINE_PENDING, strict=True)
def test_machine_emits_result_event(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    # ちょうど 1 つの終端(result)で終わる。
    assert events[-1]["type"] == "result"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1
    result = events[-1]
    assert result["mode"] == "bake"
    assert result["output"] == str(out)
    assert result["keys"] == 61  # --no-smooth の密キー(0..60)
    assert isinstance(result["applied_ranges"], list)
    assert isinstance(result["max_amplitude"], float)
    assert isinstance(result["detected_cuts"], list)


@pytest.mark.xfail(reason=_MACHINE_PENDING, strict=True)
def test_machine_stdout_is_valid_json_lines(tmp_path, capsysbinary):
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    raw = capsysbinary.readouterr().out
    text = raw.decode("utf-8")
    assert text.endswith("\n")
    objs = []
    for ln in text.split("\n")[:-1]:
        assert ln != ""  # 空行を挟まない
        obj = json.loads(ln)  # 各行が単一 JSON
        assert "type" in obj
        objs.append(obj)
    assert objs[-1]["type"] == "result" and objs[-1]["mode"] == "bake"  # 成功は result(bake)で終端


@pytest.mark.xfail(reason=_MACHINE_PENDING, strict=True)
def test_machine_no_human_text_on_stdout(tmp_path, capsysbinary):
    # 機械モードの stdout は人間向けテキスト(range:/keys:/warning: 等)を含まない(チャネル固定)。
    inp = write_input(tmp_path / "in.vmd")
    cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--verbose", "--no-smooth"])
    text = capsysbinary.readouterr().out.decode("utf-8")
    lines = [ln for ln in text.split("\n") if ln]
    assert lines  # 機械モードは少なくとも result を出す(stdout が空でない)
    for ln in lines:
        json.loads(ln)  # すべて JSON、人間向けテキスト行は混入しない


@pytest.mark.xfail(reason=_MACHINE_PENDING, strict=True)
def test_machine_emits_warning_event_for_non_camera_sections(tmp_path, capsysbinary):
    # カメラ以外のセクションを含む入力 → non_camera_sections_passthrough の warning イベント。
    # 非カメラセクションを 2 種(bone と morph)含め、section 配列が全セクション名を載せることを確認。
    bone = [BoneKey(name_raw=b"bone".ljust(15, b"\x00"), frame=0,
                    position=(0.0, 0.0, 0.0), rotation=(0.0, 0.0, 0.0, 1.0),
                    interpolation=bytes(64))]
    morph = [MorphKey(name_raw=b"morph".ljust(15, b"\x00"), frame=0, weight=0.0)]
    inp = write_input(tmp_path / "in.vmd", bone=bone, morph=morph)
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"  # 成功終端
    warns = [e for e in events if e["type"] == "warning"]
    codes = {w["code"] for w in warns}
    assert "non_camera_sections_passthrough" in codes
    w = next(w for w in warns if w["code"] == "non_camera_sections_passthrough")
    # section は透過した全セクション名の配列(§12.3)。bone と morph の両方を載せる。
    assert isinstance(w["section"], list)
    assert set(w["section"]) == {"bone", "morph"}
    assert isinstance(w["message"], str) and w["message"]  # 自由文字列の文言を message に保持(§12.3)


@pytest.mark.xfail(reason=_MACHINE_PENDING, strict=True)
def test_machine_emits_progress_events(tmp_path, capsysbinary):
    # 機械モードでは進捗をイベントとして出す(TTY 判定に依存しない)。ベイク段の progress を含む。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine", "--no-smooth"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"  # 成功終端
    progress = [e for e in events if e["type"] == "progress"]
    assert progress, "progress イベントが少なくとも 1 本出る"
    stages = {p["stage"] for p in progress}
    assert "bake" in stages
    for p in progress:
        assert set(p) >= {"type", "stage", "done", "total", "note", "elapsed"}


@pytest.mark.xfail(reason=_MACHINE_PENDING, strict=True)
def test_machine_smooth_emits_smooth_progress(tmp_path, capsysbinary):
    # 既定 on の平滑化段も機械モードで progress イベントを出す。
    inp = write_input(tmp_path / "in.vmd")
    rc = cli.main([inp, "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    events = machine_events(capsysbinary)
    assert events[-1]["type"] == "result" and events[-1]["mode"] == "bake"  # 成功終端
    stages = {e["stage"] for e in events if e["type"] == "progress"}
    assert "smooth" in stages


def test_non_machine_default_unchanged(tmp_path, capsys):
    # 後方互換: --machine なしの既定挙動(出力 VMD・終了コード)は不変。stdout に JSON を出さない。
    inp = write_input(tmp_path / "in.vmd")
    out = tmp_path / "out.vmd"
    rc = cli.main([inp, "-o", str(out), "--no-smooth"])
    assert rc == 0
    assert out.exists()
    stdout = capsys.readouterr().out
    # 既定実行は dry-run/verbose でない限り統計を出さない(JSON も出さない)。
    assert stdout.strip() == "" or not stdout.lstrip().startswith("{")


def test_machine_version_help_stay_human(capsys):
    # --machine 併用でも --version/--help は人間向けテキストを出して exit 0 で終わり、
    # イベントストリームには載せない(規約 §3 メタ操作の例外、§12.1)。argparse が両者を
    # 処理面より先に短絡するため --machine 追加の前後で不変であることを保証する。
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "shakevmd" in out and not out.lstrip().startswith("{")  # 人間向け、JSON でない
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


@pytest.mark.xfail(reason="impl pending: Step2 help text", strict=True)
def test_help_lists_machine_flag(capsys):
    # --help は人間向けテキストを出して exit 0(main は argparse の SystemExit を握って 0 を返す)。
    # 新設の --machine がヘルプに現れること(規約 §6 の人間向けヘルプ)を確認する。
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    assert "--machine" in text
