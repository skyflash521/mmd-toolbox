from pathlib import Path

import pytest

from cli_events import default_output_path, guard_output


class _RecordingFail:
    def __init__(self, result=2):
        self.result = result
        self.calls = []

    def __call__(self, code, message, exit_code, *, field=None, path=None):
        self.calls.append({"code": code, "message": message, "exit_code": exit_code,
                           "field": field, "path": path})
        return self.result


def test_missing_output_passes_and_reports_nothing(tmp_path):
    fail = _RecordingFail()
    assert guard_output(str(tmp_path / "out.vmd"), overwrite=False, fail=fail) is None
    assert fail.calls == []


def test_existing_output_without_overwrite_fails_with_exit_code_2_and_returns_its_result(tmp_path):
    out = tmp_path / "out.vmd"
    out.write_bytes(b"x")
    fail = _RecordingFail(result=2)
    assert guard_output(str(out), overwrite=False, fail=fail) == 2
    assert len(fail.calls) == 1
    call = fail.calls[0]
    assert (call["code"], call["exit_code"], call["field"], call["path"]) == (
        "output_exists", 2, "--output", None)
    assert str(out) in call["message"]
    assert "--overwrite" in call["message"]


def test_return_value_of_fail_is_passed_through_unchanged(tmp_path):
    out = tmp_path / "out.vmd"
    out.write_bytes(b"x")
    assert guard_output(str(out), overwrite=False, fail=_RecordingFail(result=99)) == 99


def test_existing_output_with_overwrite_passes(tmp_path):
    out = tmp_path / "out.vmd"
    out.write_bytes(b"x")
    fail = _RecordingFail()
    assert guard_output(str(out), overwrite=True, fail=fail) is None
    assert fail.calls == []


def test_output_equal_to_the_input_file_is_protected_like_any_existing_file(tmp_path):
    source = tmp_path / "in.vmd"
    source.write_bytes(b"x")
    fail = _RecordingFail()
    assert guard_output(str(source), overwrite=False, fail=fail) == 2
    assert fail.calls[0]["code"] == "output_exists"


@pytest.mark.parametrize("overwrite", [False, True])
def test_directory_output_fails_regardless_of_overwrite_and_carries_the_path(tmp_path, overwrite):
    fail = _RecordingFail()
    assert guard_output(str(tmp_path), overwrite=overwrite, fail=fail) == 2
    call = fail.calls[0]
    assert (call["code"], call["exit_code"], call["field"], call["path"]) == (
        "output_is_directory", 2, "--output", str(tmp_path))
    assert str(tmp_path) in call["message"]


def test_non_ascii_output_path_is_supported(tmp_path):
    out = tmp_path / "出力.vmd"
    out.write_bytes(b"x")
    fail = _RecordingFail()
    assert guard_output(str(out), overwrite=False, fail=fail) == 2
    assert "出力.vmd" in fail.calls[0]["message"]


def test_guard_has_no_dry_run_parameter_so_dry_run_cannot_bypass_it(tmp_path):
    out = tmp_path / "out.vmd"
    out.write_bytes(b"x")
    with pytest.raises(TypeError):
        guard_output(str(out), overwrite=False, fail=_RecordingFail(), dry_run=True)


@pytest.mark.parametrize("input_path, suffix, expected", [
    ("a.wav", ".vmd", "a.vmd"),
    ("a.wav", "_shake.vmd", "a_shake.vmd"),
    (str(Path("dir") / "in.vmd"), "_mocap.vmd", str(Path("dir") / "in_mocap.vmd")),
    ("a.b.vmd", "_x.vmd", "a.b_x.vmd"),
    ("noext", ".vmd", "noext.vmd"),
    ("曲.mp3", ".vpr", "曲.vpr"),
])
def test_default_output_path_replaces_the_extension_keeping_the_directory(input_path, suffix, expected):
    assert default_output_path(input_path, suffix) == expected


def test_default_output_path_returns_str():
    assert isinstance(default_output_path("a.wav", ".vmd"), str)

