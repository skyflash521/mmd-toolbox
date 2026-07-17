"""vpr2vmd CLI 機械モードのテスト。

--machine / --describe は標準出力を JSON Lines のイベント専用にし、result または error の
ちょうど 1 つで終端する。失敗は確定 code/field/exit_code の error イベントで返す。自己記述
(--describe)・入力検査(--machine --dry-run の inspect)・成功経路(warning/convert result)を
網羅する。中断・非ASCIIパスは別モジュールが扱う。

機械モード stdout は UTF-8 バイトでバイナリバッファへ書くため capsysbinary で捕捉する。
vpr.read と vmd の write_file は monkeypatch で差し替え、配線と終了コードを決定論的に検証する。
"""

import json

from vpr import Note, Part, TempoEvent, Track, VprFormatError, VprProject, VprWarning

from vpr2vmd import __version__, cli


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _note(start, dur, phonemes, *, velocity=64):
    return Note(
        start_tick=start, duration_tick=dur, pitch=60, lyric="x",
        velocity=velocity, phonemes=phonemes,
    )


def _project(notes, *, tracks=None):
    track = Track(name="Vocal", parts=[Part(name="p", start_tick=0, notes=notes)])
    return VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 120.0)],
        tracks=tracks if tracks is not None else [track],
    )


def _stub_read(monkeypatch, project, warnings=()):
    monkeypatch.setattr(cli, "read", lambda _src: (project, list(warnings)))


def _machine_events(capsysbinary):
    """capsysbinary の stdout を JSON Lines として解析しイベント配列で返す(LF のみ・UTF-8 を検証)。"""
    out = capsysbinary.readouterr().out
    assert out.endswith(b"\n") and b"\r" not in out  # 行区切りは LF 固定
    return [json.loads(ln) for ln in out.decode("utf-8").split("\n") if ln]


def _terminal_events(capsysbinary):
    """result または error のちょうど 1 つ(末尾)で終端することを確認し、全イベントを返す。"""
    events = _machine_events(capsysbinary)
    assert events, "stdout に少なくとも 1 イベントが要る"
    terminals = [e for e in events if e["type"] in ("result", "error")]
    assert len(terminals) == 1 and events[-1] is terminals[0]
    return events


def _machine_error(capsysbinary):
    events = _terminal_events(capsysbinary)
    assert events[-1]["type"] == "error"
    return events[-1]


# --- メタ操作(--version / --help は機械併用でも人間向け)-----------------------


def test_machine_version_stays_human(capsys):
    rc = cli.main(["--machine", "--version"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "vpr2vmd" in out and __version__ in out and not out.lstrip().startswith("{")


def test_machine_help_stays_human(capsys):
    rc = cli.main(["--machine", "--help"])
    assert rc == 0
    out = capsys.readouterr().out
    assert out.strip() and not out.lstrip().startswith("{")


def test_help_lists_machine_flags(capsys):
    rc = cli.main(["--help"])
    assert rc == 0
    text = capsys.readouterr().out
    for flag in ("--machine", "--describe", "--n-morph", "--verbose", "--version"):
        assert flag in text


# --- 自己記述(--describe)------------------------------------------------------


def test_describe_without_input_returns_options_and_presets(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    res = _terminal_events(capsysbinary)[-1]
    assert res["type"] == "result" and res["mode"] == "describe"

    opts = res["options"]
    assert len(opts) == 19  # 自己記述が報告する全 19 要素
    names = [o["name"] for o in opts]
    assert names[0] == "input"
    # 肯定形のみ・メタ/モード操作は除外。
    assert "--n-morph" in names and "--no-n-morph" not in names
    for meta in ("--machine", "--describe", "--version"):
        assert meta not in names
    for o in opts:
        assert set(o) == {"name", "type", "constraint", "default", "help"}
        assert o["type"] in ("float", "int", "str", "flag", "enum")
        assert o["help"]

    by = {o["name"]: o for o in opts}
    assert by["--style"]["type"] == "enum"
    assert by["--style"]["constraint"] == {"choices": ["pop", "ballad", "powerful", "whisper", "rap"]}
    assert by["--style"]["default"] == "pop"
    assert by["--n-morph"]["type"] == "flag" and by["--n-morph"]["default"] is True
    assert by["--overwrite"]["default"] is False
    assert by["--output"]["default"] is None
    # 固定既定は呼び出し先由来で報告される。
    assert by["--legato-max"]["default"] == 8.0
    assert by["--ref-bpm"]["default"] == 120.0
    assert by["--tempo-scale-min"]["default"] == 0.5
    assert by["--open-max"]["constraint"] == {"min": 0, "max": 1, "exclusive_min": False}
    assert by["--coartic-overlap"]["type"] == "int"
    assert by["--coartic-overlap"]["constraint"] == {"min": 1, "max": None, "exclusive_min": False}
    assert by["--tempo-scale-min"]["constraint"] == {"min": 0, "max": 1, "exclusive_min": True}

    presets = res["presets"]
    assert [p["name"] for p in presets] == ["pop", "ballad", "powerful", "whisper", "rap"]
    for p in presets:
        assert set(p["values"]) == {
            "open_max", "default_open", "valley_shallow", "valley_deep",
            "valley_slope", "coartic_overlap", "anticipation",
        }


def test_describe_model_name_default_is_tool_and_version(capsysbinary):
    rc = cli.main(["--describe"])
    assert rc == 0
    res = _terminal_events(capsysbinary)[-1]
    by = {o["name"]: o for o in res["options"]}
    assert by["--model-name"]["default"] == f"vpr2vmd {__version__}"


def test_describe_with_unknown_option_is_bad_argument(capsysbinary):
    # --describe と未知オプションの併用も error イベント + 終了コード 2。
    rc = cli.main(["--describe", "--bogus"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2


# --- 入力検査(--machine --dry-run の inspect)----------------------------------


def test_machine_dry_run_emits_inspect_without_writing(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--machine", "--dry-run"])
    assert rc == 0
    assert not out.exists()  # dry-run は書かない
    res = _terminal_events(capsysbinary)[-1]
    assert res["mode"] == "inspect" and res["output"] is None and res["input_kind"] == "vpr"
    assert res["track_index"] == 0 and res["track_name"] == "Vocal"
    assert res["style"] == "pop" and res["n_morph"] is True
    assert set(res["params"]) == {
        "open_max", "default_open", "legato_max", "valley_shallow", "valley_deep",
        "valley_slope", "coartic_overlap", "anticipation", "ref_bpm", "tempo_scale_min",
        "representative_bpm",
    }
    assert res["adopted_notes"] == 1
    assert isinstance(res["mouth_events"], int) and isinstance(res["morph_keys"], int)
    assert set(res["open_amounts"]) == {"min", "max", "mean"}
    assert isinstance(res["non_event_symbols"], dict)


def test_machine_dry_run_inspect_model_name_default_is_tool_and_version(
    tmp_path, capsysbinary, monkeypatch
):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--machine", "--dry-run"])
    assert rc == 0
    res = _terminal_events(capsysbinary)[-1]
    assert res["model_name"] == f"vpr2vmd {__version__}"


def test_machine_dry_run_no_adopted_warns_then_inspect(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([]))  # 発音の無いトラック
    rc = cli.main([src, "--machine", "--dry-run"])
    assert rc == 0
    events = _terminal_events(capsysbinary)
    assert any(e["type"] == "warning" and e["code"] == "no_adopted_notes" for e in events)
    assert events[-1]["mode"] == "inspect" and events[-1]["open_amounts"] is None


# --- 通常実行(convert result)--------------------------------------------------


def test_machine_convert_result_after_write(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--machine"])
    assert rc == 0
    assert out.exists()  # 実際に書き込む
    res = _terminal_events(capsysbinary)[-1]
    assert res["type"] == "result" and res["mode"] == "convert" and res["output"] == str(out)
    assert res["track_index"] == 0 and res["track_name"] == "Vocal"
    assert res["adopted_notes"] == 1
    assert isinstance(res["morph_keys"], int) and isinstance(res["mouth_events"], int)


def test_machine_warning_overlapping_notes_passthrough(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    w = VprWarning(
        code="overlapping_notes", message="発音区間が重なる音符", track_index=0, part_index=0,
        note_index=1, related_note_index=0, tick=240,
    )
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]), warnings=[w])
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 0
    events = _terminal_events(capsysbinary)
    warns = [e for e in events if e["type"] == "warning"]
    assert len(warns) == 1
    wa = warns[0]
    assert wa["code"] == "overlapping_notes" and wa["section"] is None
    assert (wa["track_index"], wa["part_index"], wa["note_index"]) == (0, 0, 1)
    assert wa["related_note_index"] == 0 and wa["tick"] == 240
    assert events[-1]["mode"] == "convert"


# --- 構造化エラー全経路 --------------------------------------------------------


def test_machine_error_unknown_option(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--machine", "--bogus"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["exit_code"] == 2 and e["field"] == "--bogus"
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_missing_input(capsysbinary):
    rc = cli.main(["--machine"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "input" and e["exit_code"] == 2


def test_machine_error_bad_value_type(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--machine", "--open-max", "abc"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--open-max" and e["exit_code"] == 2


def test_machine_error_value_out_of_range(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--machine", "--open-max", "1.5"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_argument" and e["field"] == "--open-max"


def test_machine_error_output_overwrites_input(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "-o", src, "--machine"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "output_overwrites_input" and e["field"] == "--output" and e["exit_code"] == 2


def test_machine_error_valley_bounds_inverted(tmp_path, capsysbinary):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--machine", "--valley-deep", "0.6", "--dry-run"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "valley_bounds_inverted" and e["field"] is None and e["exit_code"] == 2


def test_machine_error_bad_track(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))
    rc = cli.main([src, "--machine", "--track", "5"])
    assert rc == 2
    e = _machine_error(capsysbinary)
    assert e["code"] == "bad_track" and e["field"] == "--track" and e["exit_code"] == 2


def test_machine_error_input_not_found(tmp_path, capsysbinary):
    rc = cli.main([str(tmp_path / "nope.vpr"), "--machine"])
    assert rc == 1
    e = _machine_error(capsysbinary)
    assert e["code"] == "input_not_found" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_not_vpr(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")

    def _raise(_src):
        raise VprFormatError("壊れた vpr")

    monkeypatch.setattr(cli, "read", _raise)
    rc = cli.main([src, "--machine"])
    assert rc == 1
    e = _machine_error(capsysbinary)
    assert e["code"] == "not_vpr" and e["field"] == "input" and e["exit_code"] == 1
    assert isinstance(e["message"], str) and e["message"]


def test_machine_error_no_tracks(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([], tracks=[]))
    rc = cli.main([src, "--machine"])
    assert rc == 1
    e = _machine_error(capsysbinary)
    assert e["code"] == "no_tracks" and e["field"] == "input" and e["exit_code"] == 1


def test_machine_error_write_failed(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))

    def _raise(_doc, _path):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "write_file", _raise)
    out = str(tmp_path / "out.vmd")
    rc = cli.main([src, "-o", out, "--machine"])
    assert rc == 3
    e = _machine_error(capsysbinary)
    assert e["code"] == "write_failed" and e["field"] == "--output" and e["exit_code"] == 3
    assert e["path"] == out


def test_machine_error_internal_error(tmp_path, capsysbinary, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    _stub_read(monkeypatch, _project([_note(0, 480, ["a"])]))

    def _boom(*_a, **_k):
        raise RuntimeError("想定外")

    monkeypatch.setattr(cli, "generate_morph_keys", _boom)
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd"), "--machine"])
    assert rc == 1
    e = _machine_error(capsysbinary)
    assert e["code"] == "internal_error" and e["field"] is None and e["exit_code"] == 1


def test_machine_error_stdout_is_valid_json_lines_lf_only(tmp_path, capsysbinary):
    rc = cli.main([str(tmp_path / "nope.vpr"), "--machine"])
    assert rc == 1
    raw = capsysbinary.readouterr().out
    assert raw.endswith(b"\n") and b"\r" not in raw
    for ln in raw.decode("utf-8").split("\n"):
        if ln:
            assert "type" in json.loads(ln)


def test_describe_type_table_covers_non_meta_args():
    # _D_TYPE はメタ/モード操作を除く全 parser 引数を覆う。parser に引数を足して _D_TYPE への追加を
    # 忘れると --describe から黙って抜けるため、その載せ忘れをここで検出する。
    parser = cli._build_parser()
    meta = {"help", "version", "machine", "describe"}
    non_meta = {a.dest for a in parser._actions if a.dest not in meta}
    assert non_meta <= set(cli._D_TYPE)
