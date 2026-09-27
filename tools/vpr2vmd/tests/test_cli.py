import pytest

from vpr import Note, Part, TempoEvent, Track, VprFormatError, VprProject, VprWarning
from vpr2vmd import cli


def _assert_error_line(err):
    lines = [ln for ln in err.splitlines() if ln.strip()]
    assert any(ln.startswith("error: ") and ln[len("error: "):].strip() for ln in lines), err
    assert "Traceback" not in err, err


def _touch(path):
    path.write_bytes(b"")
    return str(path)


@pytest.fixture(autouse=True)
def _stub_read(monkeypatch):
    project = VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 120.0)],
        tracks=[Track(name="Vocal", parts=[Part(
            name="p", start_tick=0,
            notes=[Note(
                start_tick=0, duration_tick=480, pitch=60,
                lyric="x", velocity=64, phonemes=["a"],
            )],
        )])],
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, []))


def test_parses_full_option_set_in_dry_run(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    rc = cli.main([
        src,
        "-o", str(out),
        "--track", "0",
        "--model-name", "Model",
        "--style", "ballad",
        "--open-max", "0.8",
        "--default-open", "0.5",
        "--dry-run",
    ])
    assert rc == 0


def test_dry_run_plan_shows_resolved_values(tmp_path, capsys, monkeypatch):
    project = VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 150.0)],
        tracks=[Track(name="Vocal", parts=[Part(
            name="p", start_tick=0,
            notes=[Note(start_tick=0, duration_tick=480, pitch=60,
                        lyric="x", velocity=64, phonemes=["a"])],
        )])],
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, []))
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--dry-run"]) == 0
    plan_text = capsys.readouterr().out.split("--- 診断 ---")[0]
    plan = {ln.split(":", 1)[0]: ln.split(":", 1)[1].strip()
            for ln in plan_text.splitlines() if ":" in ln}

    assert "0" in plan["track"] and "Vocal" in plan["track"]
    numeric = ["open-max", "default-open", "legato-max", "coartic-overlap", "anticipation"]
    compound = ["valley(shallow/deep/slope)", "tempo(ref-bpm/scale-min)"]
    for key in numeric:
        float(plan[key])
    for key in compound:
        for value in plan[key].split("/"):
            float(value)
    assert float(plan["representative-bpm"]) == pytest.approx(150.0)


@pytest.mark.parametrize("velocities, expected", [
    ((64, 64), "default"), ((40, 100), "velocity"), ((), "なし"),
])
def test_dry_run_diagnostics_show_open_source(tmp_path, capsys, monkeypatch,
                                              velocities, expected):
    project = VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 120.0)],
        tracks=[Track(name="Vocal", parts=[Part(
            name="p", start_tick=0,
            notes=[Note(start_tick=480 * i, duration_tick=480, pitch=60, lyric="x",
                        velocity=v, phonemes=["a"])
                   for i, v in enumerate(velocities)],
        )])],
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, []))
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--dry-run"]) == 0
    diagnostics = capsys.readouterr().out.split("--- 診断 ---")[1]
    shown = {ln.split(":", 1)[0]: ln.split(":", 1)[1].strip()
             for ln in diagnostics.splitlines() if ":" in ln}
    assert shown["開き量の決定経路"] == expected


def test_dry_run_writes_no_output(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--dry-run"])
    assert rc == 0
    assert not out.exists()


@pytest.mark.parametrize("option", [
    pytest.param(["--report-json", "rep.json"], id="no_report_file_output"),
    pytest.param(["--quiet"], id="no_quiet_without_progress_display"),
    pytest.param(["--overw"], id="no_abbreviated_long_option"),
])
def test_unsupported_option_is_arg_error(tmp_path, option):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, *option, "--dry-run"]) == 2


def test_track_accepts_non_integer_name(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--track", "Vocal", "--dry-run"]) == 0


def test_unknown_option_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--bogus"]) == 2


def test_missing_positional_is_arg_error():
    assert cli.main([]) == 2


def test_unknown_style_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--style", "nope", "--dry-run"]) == 2


def test_open_max_non_float_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--open-max", "abc", "--dry-run"]) == 2


def test_default_open_non_float_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--default-open", "abc", "--dry-run"]) == 2


def test_open_max_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--open-max", "-0.1", "--dry-run"]) == 2


def test_open_max_over_one_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--open-max", "1.5", "--dry-run"]) == 2


def test_default_open_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--default-open", "-0.5", "--dry-run"]) == 2


def test_open_amount_bounds_are_accepted(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--open-max", "1.0", "--default-open", "0.0", "--dry-run"]) == 0


def test_missing_input_file_is_input_error(tmp_path):
    missing = str(tmp_path / "nope.vpr")
    assert cli.main([missing]) == 1


def test_existing_other_path_output_blocked_without_overwrite(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    out = _touch(tmp_path / "out.vmd")
    assert cli.main([src, "-o", out, "--dry-run"]) == 2


def test_existing_other_path_output_not_overwritten_without_overwrite(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    out.write_bytes(b"stale")
    rc = cli.main([src, "-o", str(out)])
    assert rc == 2
    assert out.read_bytes() == b"stale"


def test_overwrite_allows_existing_output(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    out = _touch(tmp_path / "out.vmd")
    assert cli.main([src, "-o", out, "--overwrite", "--dry-run"]) == 0


def test_overwrite_guard_blocks_input_overwrite(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "-o", src, "--dry-run"]) == 2


def test_missing_same_path_output_is_input_error_not_overwrite_guard(tmp_path):
    missing = str(tmp_path / "missing.vpr")
    assert cli.main([missing, "-o", missing, "--dry-run"]) == 1


def test_model_name_over_20_bytes_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "x" * 21, "--dry-run"]) == 2


def test_no_n_morph_accepted_in_dry_run(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--no-n-morph", "--dry-run"]) == 0


def test_dry_run_plan_reflects_n_morph_off_by_default(tmp_path, capsys):
    src = _touch(tmp_path / "in.vpr")
    cli.main([src, "--dry-run"])
    assert "n-morph: off" in capsys.readouterr().out


def test_dry_run_plan_reflects_no_n_morph(tmp_path, capsys):
    src = _touch(tmp_path / "in.vpr")
    cli.main([src, "--no-n-morph", "--dry-run"])
    assert "n-morph: off" in capsys.readouterr().out


def test_dry_run_plan_reflects_n_morph_on(tmp_path, capsys):
    src = _touch(tmp_path / "in.vpr")
    cli.main([src, "--n-morph", "--dry-run"])
    assert "n-morph: on" in capsys.readouterr().out


def test_model_name_multibyte_over_20_bytes_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "あ" * 11, "--dry-run"]) == 2


def test_model_name_multibyte_at_20_byte_limit_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "あ" * 10, "--dry-run"]) == 0


def test_model_name_non_cp932_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "\U0001F600", "--dry-run"]) == 2


def test_model_name_at_20_byte_limit_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--model-name", "x" * 20, "--dry-run"]) == 0


def test_existing_default_output_blocked_without_overwrite(tmp_path):
    src = _touch(tmp_path / "song.vpr")
    _touch(tmp_path / "song.vmd")
    assert cli.main([src, "--dry-run"]) == 2


def test_existing_default_output_allowed_with_overwrite(tmp_path):
    src = _touch(tmp_path / "song.vpr")
    _touch(tmp_path / "song.vmd")
    assert cli.main([src, "--overwrite", "--dry-run"]) == 0


def test_dry_run_does_not_write_default_output(tmp_path):
    src = _touch(tmp_path / "song.vpr")
    rc = cli.main([src, "--dry-run"])
    assert rc == 0
    assert not (tmp_path / "song.vmd").exists()


def test_default_output_replaces_input_extension_with_vmd(tmp_path):
    src = _touch(tmp_path / "song.dat")
    assert cli.main([src]) == 0
    assert (tmp_path / "song.vmd").exists()


def test_parses_tuning_options_in_dry_run(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([
        src,
        "--legato-max", "10",
        "--valley-shallow", "0.5",
        "--valley-deep", "0.25",
        "--valley-slope", "0.03",
        "--coartic-overlap", "5",
        "--anticipation", "8",
        "--ref-bpm", "160",
        "--tempo-scale-min", "0.4",
        "--dry-run",
    ])
    assert rc == 0


def test_legato_max_non_positive_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--legato-max", "0", "--dry-run"]) == 2


def test_legato_max_non_float_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--legato-max", "abc", "--dry-run"]) == 2


def test_valley_shallow_out_of_range_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--valley-shallow", "1.5", "--dry-run"]) == 2
    assert cli.main([src, "--valley-deep", "-0.1", "--dry-run"]) == 2


def test_valley_slope_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--valley-slope", "-0.1", "--dry-run"]) == 2


def test_coartic_overlap_below_one_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--coartic-overlap", "0", "--dry-run"]) == 2


def test_anticipation_negative_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--anticipation", "-1", "--dry-run"]) == 2


def test_anticipation_zero_is_accepted(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--anticipation", "0", "--dry-run"]) == 0


def test_ref_bpm_non_positive_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--ref-bpm", "0", "--dry-run"]) == 2


def test_tempo_scale_min_out_of_range_is_arg_error(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--tempo-scale-min", "0", "--dry-run"]) == 2
    assert cli.main([src, "--tempo-scale-min", "1.5", "--dry-run"]) == 2
    assert cli.main([src, "--tempo-scale-min", "1.0", "--dry-run"]) == 0


def test_valley_deep_above_shallow_is_arg_error_even_in_dry_run(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--valley-deep", "0.6", "--dry-run"]) == 2
    assert cli.main([src, "--valley-shallow", "0.1", "--valley-deep", "0.2", "--dry-run"]) == 2
    assert cli.main([src, "--valley-shallow", "0.5", "--valley-deep", "0.2", "--dry-run"]) == 0


def test_dry_run_plan_shows_tuning_overrides(tmp_path, capsys):
    src = _touch(tmp_path / "in.vpr")
    cli.main([src, "--legato-max", "12", "--anticipation", "9", "--dry-run"])
    out = capsys.readouterr().out
    assert "legato-max: 12" in out
    assert "anticipation: 9" in out


def _project_with_notes(notes):
    return VprProject(
        resolution=480,
        tempos=[TempoEvent(0, 120.0)],
        tracks=[Track(name="Vocal", parts=[Part(name="p", start_tick=0, notes=notes)])],
    )


def test_version_attr_exposed():
    import vpr2vmd

    assert isinstance(vpr2vmd.__version__, str) and vpr2vmd.__version__


def test_version_flag_prints_and_exits_zero(capsys):
    import vpr2vmd

    rc = cli.main(["--version"])
    out = capsys.readouterr().out
    assert rc == 0
    assert "vpr2vmd" in out
    assert vpr2vmd.__version__ in out


def test_n_morph_positive_flag_accepted(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    assert cli.main([src, "--n-morph", "--dry-run"]) == 0


def test_verbose_prints_plan_on_normal_run(tmp_path, capsys):
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"
    rc = cli.main([src, "-o", str(out), "--verbose"])
    captured = capsys.readouterr().out
    assert rc == 0
    assert out.exists()
    assert "採用音符数" in captured


def test_verbose_does_not_change_output_vmd(tmp_path):
    src = _touch(tmp_path / "in.vpr")
    plain = tmp_path / "plain.vmd"
    verbose = tmp_path / "verbose.vmd"
    assert cli.main([src, "-o", str(plain)]) == 0
    assert cli.main([src, "-o", str(verbose), "--verbose"]) == 0
    assert plain.read_bytes() == verbose.read_bytes()


def test_missing_input_reports_reason(tmp_path, capsys):
    missing = str(tmp_path / "nope.vpr")
    rc = cli.main([missing])
    assert rc == 1
    _assert_error_line(capsys.readouterr().err)


def test_missing_positional_reports_reason(capsys):
    rc = cli.main([])
    assert rc == 2
    assert capsys.readouterr().err.strip()


def test_not_vpr_reports_reason(tmp_path, capsys, monkeypatch):
    src = _touch(tmp_path / "in.vpr")

    def _raise(_src):
        raise VprFormatError("壊れた vpr")

    monkeypatch.setattr(cli, "read", _raise)
    rc = cli.main([src])
    assert rc == 1
    _assert_error_line(capsys.readouterr().err)


def test_unreadable_input_reports_reason(tmp_path, capsys, monkeypatch):
    src = _touch(tmp_path / "in.vpr")

    def _raise(_src):
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli, "read", _raise)
    rc = cli.main([src])
    assert rc == 1
    _assert_error_line(capsys.readouterr().err)


def _stub_normalize_warnings(monkeypatch, *injected):
    original = cli.normalize

    def _with_warnings(document, sections=None):
        doc, warnings = original(document, sections=sections)
        return doc, [*warnings, *injected]

    monkeypatch.setattr(cli, "normalize", _with_warnings)


def test_normalize_duplicate_warning_is_surfaced(tmp_path, capsys, monkeypatch):
    from vmd import VmdWarning

    src = _touch(tmp_path / "in.vpr")
    _stub_normalize_warnings(monkeypatch, VmdWarning(
        code="normalize-duplicate", message="同一キーの重複", section="morph"))
    assert cli.main([src, "-o", str(tmp_path / "out.vmd")]) == 0
    assert "warning: normalize-duplicate:" in capsys.readouterr().err


def test_normalize_sorted_warning_is_not_surfaced(tmp_path, capsys, monkeypatch):
    from vmd import VmdWarning

    src = _touch(tmp_path / "in.vpr")
    _stub_normalize_warnings(monkeypatch, VmdWarning(
        code="normalize-sorted", message="キーを並べ替えた", section="morph"))
    assert cli.main([src, "-o", str(tmp_path / "out.vmd")]) == 0
    assert "normalize-sorted" not in capsys.readouterr().err


def test_identical_normalize_warnings_share_one_line(tmp_path, capsys, monkeypatch):
    from vmd import VmdWarning

    src = _touch(tmp_path / "in.vpr")
    duplicate = VmdWarning(code="normalize-duplicate", message="同一キーの重複", section="morph")
    _stub_normalize_warnings(monkeypatch, duplicate, duplicate)
    assert cli.main([src, "-o", str(tmp_path / "out.vmd")]) == 0
    assert capsys.readouterr().err.count("warning: normalize-duplicate:") == 1


def test_identical_read_warnings_share_one_line(tmp_path, capsys, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    project = _project_with_notes(
        [Note(start_tick=0, duration_tick=480, pitch=60, lyric="x", velocity=64, phonemes=["a"])]
    )
    warnings = [
        VprWarning(code="overlapping_notes", message="重なり音符", track_index=0, part_index=0,
                   note_index=note_index, related_note_index=0, tick=0)
        for note_index in (1, 2)
    ]
    monkeypatch.setattr(cli, "read", lambda _src: (project, warnings))
    assert cli.main([src, "-o", str(tmp_path / "out.vmd")]) == 0
    assert capsys.readouterr().err.count("warning: overlapping_notes:") == 1


def test_no_tracks_reports_reason(tmp_path, capsys, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    empty = VprProject(resolution=480, tempos=[TempoEvent(0, 120.0)], tracks=[])
    monkeypatch.setattr(cli, "read", lambda _src: (empty, []))
    rc = cli.main([src])
    assert rc == 1
    _assert_error_line(capsys.readouterr().err)


def test_bad_track_reports_reason(tmp_path, capsys):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--track", "5"])
    assert rc == 2
    _assert_error_line(capsys.readouterr().err)


def test_valley_inverted_reports_reason(tmp_path, capsys):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "--valley-deep", "0.6", "--dry-run"])
    assert rc == 2
    _assert_error_line(capsys.readouterr().err)


def test_output_exists_reports_reason(tmp_path, capsys):
    src = _touch(tmp_path / "in.vpr")
    rc = cli.main([src, "-o", src, "--dry-run"])
    assert rc == 2
    _assert_error_line(capsys.readouterr().err)


def test_write_failure_reports_reason(tmp_path, capsys, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    out = tmp_path / "out.vmd"

    def _raise(_doc, _path):
        raise OSError("disk full")

    monkeypatch.setattr(cli, "write_file", _raise)
    rc = cli.main([src, "-o", str(out)])
    assert rc == 3
    _assert_error_line(capsys.readouterr().err)


def test_internal_error_reports_reason_without_traceback(tmp_path, capsys, monkeypatch):
    src = _touch(tmp_path / "in.vpr")

    def _boom(*_a, **_k):
        raise RuntimeError("想定外")

    monkeypatch.setattr(cli, "generate_morph_keys", _boom)
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    assert rc == 1
    _assert_error_line(capsys.readouterr().err)


def test_vpr_read_warning_surfaced_to_stderr(tmp_path, capsys, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    project = _project_with_notes(
        [Note(start_tick=0, duration_tick=480, pitch=60, lyric="x", velocity=64, phonemes=["a"])]
    )
    warning = VprWarning(
        code="overlapping_notes", message="重なり音符", track_index=0, part_index=0,
        note_index=1, related_note_index=0, tick=0,
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, [warning]))
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    err = capsys.readouterr().err
    assert rc == 0
    assert "重なり音符" in err or "overlapping_notes" in err


def test_no_adopted_notes_warns_and_succeeds(tmp_path, capsys, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    empty_track = _project_with_notes([])
    monkeypatch.setattr(cli, "read", lambda _src: (empty_track, []))
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    assert rc == 0
    assert "発音" in capsys.readouterr().err


def test_read_warning_line_uses_common_format(tmp_path, capsys, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    project = _project_with_notes(
        [Note(start_tick=0, duration_tick=480, pitch=60, lyric="x", velocity=64, phonemes=["a"])]
    )
    warning = VprWarning(
        code="overlapping_notes", message="重なり音符", track_index=0, part_index=0,
        note_index=1, related_note_index=0, tick=0,
    )
    monkeypatch.setattr(cli, "read", lambda _src: (project, [warning]))
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1
    prefix = "warning: overlapping_notes: "
    assert lines[0].startswith(prefix)
    body = lines[0][len(prefix):]
    assert body.strip()
    assert not body.startswith(" ")
    assert "警告:" not in err


def test_no_adopted_notes_warning_line_uses_common_format(tmp_path, capsys, monkeypatch):
    src = _touch(tmp_path / "in.vpr")
    empty_track = _project_with_notes([])
    monkeypatch.setattr(cli, "read", lambda _src: (empty_track, []))
    rc = cli.main([src, "-o", str(tmp_path / "out.vmd")])
    assert rc == 0
    out, err = capsys.readouterr()
    assert out == ""
    lines = err.splitlines()
    assert len(lines) == 1
    prefix = "warning: no_adopted_notes: "
    assert lines[0].startswith(prefix)
    body = lines[0][len(prefix):]
    assert body.strip()
    assert not body.startswith(" ")
    assert "警告:" not in err


def test_non_machine_usage_error_is_single_error_line(capsys):
    rc = cli.main(["in.vpr", "--bogus"])
    assert rc == 2
    assert capsys.readouterr().err == "error: unrecognized arguments: --bogus\n"
