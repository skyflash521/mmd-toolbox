"""sparsevmd CLI 機械モード骨格・構造化エラーのテスト。

機械モードは stdout を JSON Lines のイベント専用にし、失敗は確定 code/field/exit_code の error
イベントで終端する。非機械モードは失敗理由を標準エラーへ 1 行出す。既定(非機械)挙動が不変である
こと(後方互換)も併せて検証する。

本モジュールは骨格(--version / --machine / --describe / --quiet / --cut-detect フラグ・
MachineArgumentParser 切替・emitter・fail() 単一失敗経路・stderr の backslashreplace 再構成・
help= 付与・input の nargs="?" 化)と構造化エラーの全経路を対象にする。成功経路のイベント
(progress / warning / result)・自己記述 result・中断は本モジュールの対象外とする。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。テストは
決定論的に実行し、外部依存を使わない。
"""

import json

import pytest

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


def machine_events(capsysbinary):
    """capsysbinary で捕捉した stdout を JSON Lines として解析しイベント配列で返す。"""
    out = capsysbinary.readouterr().out
    text = out.decode("utf-8")  # UTF-8 固定(ロケール非依存)を前提に decode
    return [json.loads(ln) for ln in text.split("\n") if ln]


def machine_error(capsysbinary):
    """機械モードの stdout を解析し、終端の error イベントを返す(失敗は error で終端)。"""
    events = machine_events(capsysbinary)
    assert events, "stdout に少なくとも 1 イベントが要る"
    assert events[-1]["type"] == "error"
    assert sum(1 for e in events if e["type"] in ("result", "error")) == 1  # 終端はちょうど1つ
    return events[-1]


def ramp_camera(path):
    write_vmd(path, camera=linear_camera_doc())


# --- メタ操作(--version / --help)------------------------------------------


def test_version_prints_and_exits_zero(capsys):
    # --version は __version__ を表示して終了コード0。番号源は __version__ 一本。
    from sparsevmd import __version__

    rc = cli.main(["--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sparsevmd" in out and __version__ in out
    assert __version__ == "0.0.1"  # 初期版の番号を固定する


def test_machine_version_stays_human(capsys):
    # --machine 併用でも --version は人間向けテキスト+exit 0、イベントに載せない。
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "sparsevmd" in out and not out.lstrip().startswith("{")


def test_machine_help_stays_human(capsys):
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


def test_help_lists_new_flags(capsys):
    # --help に新設フラグが現れる(人間向けヘルプ)。
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    for flag in ("--machine", "--describe", "--quiet", "--version", "--cut-detect"):
        assert flag in text


# --- 引数エラー(argparse 検出)--------------------------------------------


def test_machine_error_bad_argument_unknown_option(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--bogus"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2
    assert e["field"] == "--bogus"   # unrecognized arguments: の先頭トークン
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_bad_argument_missing_input(capsysbinary):
    # positional input 欠落(--describe 以外の実行)→ bad_argument、field は input。
    rc = cli.main(["--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "input" and e["exit_code"] == 2


def test_machine_error_bad_argument_type_error_field(tmp_path, capsysbinary):
    # 型エラー(--max-segment-frames 非整数)→ argparse 検出の bad_argument、field は長形式。
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--max-segment-frames", "abc"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--max-segment-frames" and e["exit_code"] == 2


@pytest.mark.parametrize("opt", ["--min-segment-frames", "--max-segment-frames"])
def test_machine_error_bad_argument_segment_below_one(tmp_path, capsysbinary, opt):
    # 解析後の単一オプション検証(区間長 < 1)も bad_argument(該当オプションの field)。
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--target", "camera", opt, "0"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == opt and e["exit_code"] == 2


def test_machine_error_segment_bounds_conflict(tmp_path, capsysbinary):
    # min > max は 2 オプションにまたがる → segment_bounds_conflict、field は null。
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--target", "camera",
                   "--min-segment-frames", "10", "--max-segment-frames", "5"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "segment_bounds_conflict" and e["field"] is None and e["exit_code"] == 2


def test_machine_error_bad_tolerance(tmp_path, capsysbinary):
    # 許容誤差の検証失敗(fov < 0.5)→ bad_tolerance、field は null(起因は message に載る)。
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--target", "camera", "--camera-fov-tol", "0.4"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_tolerance" and e["field"] is None and e["exit_code"] == 2
    assert isinstance(e["message"], str) and e["message"]


# --- 引数エラー(パス・競合・上書き)---------------------------------------


def test_machine_error_input_not_file(tmp_path, capsysbinary):
    rc = cli.main([str(tmp_path / "nope.vmd"), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "input_not_file" and e["field"] == "input" and e["exit_code"] == 2


def test_machine_error_bone_file_not_file(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30)])
    rc = cli.main([str(src), "--machine", "--target", "bone",
                   "--bone-file", str(tmp_path / "nope.txt")])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bone_file_not_file" and e["field"] == "--bone-file" and e["exit_code"] == 2


def test_machine_error_bad_bone_file(tmp_path, capsysbinary):
    # 存在するが UTF-8 デコード不能な --bone-file → bad_bone_file(読み込み失敗)。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", f, pos=(0.0, float(f), 0.0)) for f in range(11)])
    bf = tmp_path / "bones.txt"
    bf.write_bytes(b"\xff\xfe\x00 invalid utf8")
    rc = cli.main([str(src), "--machine", "--target", "bone", "--bone-file", str(bf)])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bad_bone_file" and e["field"] == "--bone-file" and e["exit_code"] == 2


def test_machine_error_target_selection_conflict(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc(), bone=[bone("センター", 0)])
    rc = cli.main([str(src), "--machine", "--target", "camera", "--bone", "センター"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "target_selection_conflict" and e["field"] is None and e["exit_code"] == 2


def test_machine_error_output_overwrites_input(tmp_path, capsysbinary):
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "-o", str(src), "--machine"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "output_overwrites_input" and e["field"] == "--output" and e["exit_code"] == 2


def test_machine_error_bone_selection_invalid(tmp_path, capsysbinary):
    # 唯一の include が不一致 glob → 最終0件で SelectionError → bone_selection_invalid、field は null。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("頭", 0), bone("頭", 30)])
    rc = cli.main([str(src), "--machine", "--target", "bone", "--bone-glob", "幻*"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "bone_selection_invalid" and e["field"] is None and e["exit_code"] == 2
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_range_invalid(tmp_path, capsysbinary):
    # 省略端解決後の逆順(999: が末尾30に展開され 999>30)→ range_invalid、field は --range。
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "--machine", "--target", "camera", "--range", "999:"])
    assert rc == 2
    e = machine_error(capsysbinary)
    assert e["code"] == "range_invalid" and e["field"] == "--range" and e["exit_code"] == 2


# --- 入力不正・処理固有の失敗 ----------------------------------------------


def test_machine_error_not_vmd(tmp_path, capsysbinary):
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file at all")
    rc = cli.main([str(bad), "--machine"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "not_vmd" and e["field"] == "input" and e["exit_code"] == 1
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_no_target_keys(tmp_path, capsysbinary):
    # --target camera だがカメラキー無し → no_target_keys(exit 1)、field は input。
    src = tmp_path / "in.vmd"
    write_vmd(src, bone=[bone("センター", 0), bone("センター", 30)])
    rc = cli.main([str(src), "--machine", "--target", "camera"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "no_target_keys" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_strict_tolerance_unmet(tmp_path, capsysbinary):
    # strict で許容を満たせない(ジグザグ)→ strict_tolerance_unmet(exit 4)、field は null。
    src = tmp_path / "in.vmd"
    cam_keys = [cam(f, center=(0.0, 0.0 if f % 2 == 0 else 5.0, 0.0)) for f in range(9)]
    write_vmd(src, camera=cam_keys)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine", "--target", "camera",
                   "--curve-mode", "linear", "--strict",
                   "--min-segment-frames", "8", "--max-segment-frames", "180"])
    assert rc == 4
    e = machine_error(capsysbinary)
    assert e["code"] == "strict_tolerance_unmet" and e["field"] is None and e["exit_code"] == 4


def test_machine_error_write_failed(tmp_path, capsysbinary):
    # 出力先の親がファイル → write_failed(exit 3)、field は --output、path 付き。
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    clash = tmp_path / "afile"
    clash.write_bytes(b"x")
    out = str(clash / "out.vmd")
    rc = cli.main([str(src), "-o", out, "--machine", "--target", "camera", "--curve-mode", "linear"])
    assert rc == 3
    e = machine_error(capsysbinary)
    assert e["code"] == "write_failed" and e["field"] == "--output" and e["exit_code"] == 3
    assert e["path"] == out


def test_machine_error_internal_error(tmp_path, capsysbinary, monkeypatch):
    # 想定外の内部例外(reduce_camera_track が RuntimeError)→ internal_error(exit 1)。安全網。
    def boom(*a, **k):
        raise RuntimeError("boom")
    monkeypatch.setattr(cli, "reduce_camera_track", boom)
    src = tmp_path / "in.vmd"
    ramp_camera(src)
    rc = cli.main([str(src), "-o", str(tmp_path / "out.vmd"), "--machine", "--target", "camera"])
    assert rc == 1
    e = machine_error(capsysbinary)
    assert e["code"] == "internal_error" and e["exit_code"] == 1 and e["field"] is None


# --- チャネル固定(JSON Lines・LF)-----------------------------------------


def test_machine_error_stdout_is_valid_json_lines_lf_only(tmp_path, capsysbinary):
    # エラー経路でも stdout は有効な JSON Lines・LF のみ(\r 不在)。
    rc = cli.main([str(tmp_path / "nope.vmd"), "--machine"])
    assert rc == 2
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n") and b"\r" not in raw
    for ln in raw.decode("utf-8").split("\n"):
        if ln:
            obj = json.loads(ln)
            assert "type" in obj


# --- 非機械モードの理由 1 行-----------------------------------


def test_non_machine_error_prints_reason_to_stderr(tmp_path, capsys):
    # 非機械モードでも失敗理由を標準エラーへ 1 行出す。終了コードは維持、stdout に JSON は出さない。
    bad = tmp_path / "bad.vmd"
    bad.write_bytes(b"not a vmd file")
    rc = cli.main([str(bad)])
    assert rc == 1
    cap = capsys.readouterr()
    assert "error:" in cap.err.lower()
    assert cap.out.strip() == "" or not cap.out.lstrip().startswith("{")


def test_non_machine_missing_input_is_arg_error(capsys):
    # 非機械・input 欠落(--describe 以外)→ 引数エラー(exit 2)+理由 1 行。
    rc = cli.main([])
    assert rc == 2
    assert "error:" in capsys.readouterr().err.lower()


# --- 進捗のライブ表示抑制------------------------------------


def test_quiet_disables_live_progress_even_on_tty(tmp_path, capsys, monkeypatch):
    # --quiet は標準エラーが端末でもライブ進捗表示を抑制する。TTY を擬装して検証する。
    import sys as _sys
    monkeypatch.setattr(_sys.stderr, "isatty", lambda: True, raising=False)
    src = tmp_path / "in.vmd"
    write_vmd(src, camera=linear_camera_doc())
    # 前提: TTY 扱いなら通常実行(--quiet なし)では進捗ラベルが標準エラーに出る。
    rc = cli.main([str(src), "-o", str(tmp_path / "base.vmd"),
                   "--target", "camera", "--curve-mode", "linear"])
    assert rc == 0
    assert "カメラ削減" in capsys.readouterr().err
    # --quiet はそのライブ進捗を抑制する(TTY でも出さない)。
    rc = cli.main([str(src), "-o", str(tmp_path / "quiet.vmd"),
                   "--target", "camera", "--curve-mode", "linear", "--quiet"])
    assert rc == 0
    assert "カメラ削減" not in capsys.readouterr().err
