import io
import json
import signal
import sys

import pytest

from cli_events import (
    EventEmitter,
    MachineArgumentParser,
    add_common_arguments,
    default_output_path,
    guard_output,
    run_tool,
)

_NON_CP932_CHAR = "\U0001f3a5"
_INTERRUPT_MESSAGE = "中断された(Ctrl-C 等)"


def _build_parser():
    parser = MachineArgumentParser(prog="sample", allow_abbrev=False)
    parser.add_argument("input", nargs="?")
    add_common_arguments(parser, version="1.2.3", output_help="出力先")
    return parser


def _describe(parser):
    return {"options": [{"name": "input"}], "presets": [{"name": "p", "values": {"x": 1}}], "prog": parser.prog}


def _tool(argv, run, describe=_describe):
    return run_tool(argv, build_parser=_build_parser, describe=describe, run=run)


def _ok(args, ctx):
    return 0


class _Console:
    def __init__(self, monkeypatch, stderr=None, *, install=True):
        self._monkeypatch = monkeypatch
        self.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8", newline="")
        self.stderr = io.StringIO() if stderr is None else stderr
        if install:
            self.install()

    def install(self):
        self._monkeypatch.setattr(sys, "stdout", self.stdout)
        self._monkeypatch.setattr(sys, "stderr", self.stderr)

    def tool(self, argv, run, describe=_describe):
        self.install()
        return _tool(argv, run, describe=describe)

    def stdout_bytes(self):
        self.stdout.flush()
        return self.stdout.buffer.getvalue()

    def stdout_text(self):
        return self.stdout_bytes().decode("utf-8")

    def events(self):
        return [json.loads(line) for line in self.stdout_text().split("\n") if line]

    def stderr_text(self):
        return self.stderr.getvalue()


@pytest.fixture
def console(monkeypatch):
    return _Console(monkeypatch, install=False)


def _error_event(code, exit_code, message, *, field=None, path=None):
    return {"type": "error", "code": code, "exit_code": exit_code, "field": field, "path": path,
            "message": message}


class TestRunResult:
    def test_returns_what_run_returns(self, console):
        assert console.tool(["in.vmd"], lambda args, ctx: 4) == 4

    def test_run_receives_the_parsed_namespace(self, console):
        seen = {}

        def run(args, ctx):
            seen["input"] = args.input
            seen["dry_run"] = args.dry_run
            return 0

        console.tool(["in.vmd", "--dry-run"], run)
        assert seen == {"input": "in.vmd", "dry_run": True}

    def test_argv_none_reads_the_process_arguments(self, console, monkeypatch):
        monkeypatch.setattr(sys, "argv", ["sample", "from_argv.vmd"])
        seen = {}

        def run(args, ctx):
            seen["input"] = args.input
            return 0

        assert console.tool(None, run) == 0
        assert seen["input"] == "from_argv.vmd"

    def test_nothing_is_written_when_run_writes_nothing(self, console):
        assert console.tool(["in.vmd"], _ok) == 0
        assert console.stdout_bytes() == b""
        assert console.stderr_text() == ""


class TestContext:
    def test_without_machine_flag_there_is_no_emitter(self, console):
        seen = {}

        def run(args, ctx):
            seen["machine"] = ctx.machine
            seen["emitter"] = ctx.emitter
            return 0

        console.tool(["in.vmd"], run)
        assert seen == {"machine": False, "emitter": None}

    def test_machine_flag_gives_an_emitter_writing_utf8_lines_to_stdout(self, console):
        seen = {}

        def run(args, ctx):
            seen["machine"] = ctx.machine
            seen["emitter"] = ctx.emitter
            ctx.emitter.result(mode="process", output="出力.vmd")
            return 0

        assert console.tool(["in.vmd", "--machine"], run) == 0
        assert seen["machine"] is True
        assert isinstance(seen["emitter"], EventEmitter)
        assert console.stdout_bytes() == (
            json.dumps({"type": "result", "mode": "process", "output": "出力.vmd"}, ensure_ascii=False)
            + "\n").encode("utf-8")

    def test_fail_in_machine_mode_emits_the_error_event_and_returns_the_exit_code(self, console):
        def run(args, ctx):
            return ctx.fail("not_vmd", "読めない", 1, field="input", path="in.vmd")

        assert console.tool(["in.vmd", "--machine"], run) == 1
        assert console.events() == [_error_event("not_vmd", 1, "読めない", field="input", path="in.vmd")]
        assert console.stderr_text() == ""

    def test_fail_extra_keys_are_added_to_the_error_event(self, console):
        def run(args, ctx):
            return ctx.fail("stage_failed", "分離に失敗", 4, field="input", stage="separate")

        assert console.tool(["in.wav", "--machine"], run) == 4
        assert console.events() == [dict(_error_event("stage_failed", 4, "分離に失敗", field="input"),
                                         stage="separate")]

    def test_fail_without_machine_writes_one_error_line_to_stderr_only(self, console):
        def run(args, ctx):
            return ctx.fail("not_vmd", "読めない", 1, field="input")

        assert console.tool(["in.vmd"], run) == 1
        assert console.stderr_text() == "error: 読めない\n"
        assert console.stdout_bytes() == b""

    def test_warn_in_machine_mode_emits_a_warning_event_and_stays_off_stderr(self, console):
        def run(args, ctx):
            ctx.warn("selector_unmatched", "一致なし", section=["bone"])
            ctx.warn("plain", "セクション無し")
            return 0

        assert console.tool(["in.vmd", "--machine"], run) == 0
        assert console.events() == [
            {"type": "warning", "code": "selector_unmatched", "message": "一致なし", "section": ["bone"]},
            {"type": "warning", "code": "plain", "message": "セクション無し", "section": None},
        ]
        assert console.stderr_text() == ""

    def test_warn_extra_keys_are_added_to_the_warning_event(self, console):
        def run(args, ctx):
            ctx.warn("overlap", "重なり", track_index=3)
            return 0

        console.tool(["in.vmd", "--machine"], run)
        assert console.events() == [
            {"type": "warning", "code": "overlap", "message": "重なり", "section": None, "track_index": 3}]

    def test_warn_without_machine_writes_the_formatted_line_to_stderr_only(self, console):
        def run(args, ctx):
            ctx.warn("selector_unmatched", "一致なし", section="bone")
            ctx.warn("plain", "セクション無し")
            return 0

        assert console.tool(["in.vmd"], run) == 0
        assert console.stderr_text() == (
            "warning: selector_unmatched: 一致なし(bone)\nwarning: plain: セクション無し\n")
        assert console.stdout_bytes() == b""

    def test_warnings_are_not_suppressed_by_quiet(self, console):
        def run(args, ctx):
            ctx.warn("plain", "m")
            return 0

        console.tool(["in.vmd", "--quiet"], run)
        assert console.stderr_text() == "warning: plain: m\n"


class TestDescribe:
    def test_describe_emits_one_result_event_and_exits_zero_without_running(self, console):
        called = []
        rc = console.tool(["--describe"], lambda args, ctx: called.append(1) or 0)
        assert rc == 0
        assert called == []
        assert console.events() == [{
            "type": "result", "mode": "describe",
            "options": [{"name": "input"}],
            "presets": [{"name": "p", "values": {"x": 1}}],
            "prog": "sample",
        }]
        assert console.stderr_text() == ""

    def test_describe_does_not_need_the_input_argument_nor_the_machine_flag(self, console):
        assert console.tool(["--describe"], _ok) == 0
        assert len(console.events()) == 1

    def test_describe_with_machine_flag_is_the_same_single_event(self, console):
        assert console.tool(["--describe", "--machine"], _ok) == 0
        assert len(console.events()) == 1

    def test_describe_receives_the_built_parser(self, console):
        seen = []
        console.tool(["--describe"], _ok, describe=lambda parser: seen.append(parser) or {})
        assert len(seen) == 1
        assert seen[0].prog == "sample"

    def test_describe_result_line_is_one_json_line_ending_in_lf(self, console):
        console.tool(["--describe"], _ok)
        raw = console.stdout_bytes()
        assert raw.endswith(b"\n") and raw.count(b"\n") == 1 and b"\r" not in raw


class TestMetaOperations:
    def test_version_prints_the_version_line_on_stdout_and_exits_zero(self, console):
        assert console.tool(["--version"], _ok) == 0
        assert console.stdout_text() == "sample 1.2.3\n"
        assert console.stderr_text() == ""

    def test_version_with_machine_flag_stays_human_text(self, console):
        assert console.tool(["--version", "--machine"], _ok) == 0
        assert console.stdout_text() == "sample 1.2.3\n"

    def test_help_exits_zero_without_running(self, console):
        called = []
        assert console.tool(["--help"], lambda args, ctx: called.append(1) or 0) == 0
        assert "--machine" in console.stdout_text()
        assert called == []


class TestUsageError:
    def test_machine_mode_emits_a_bad_argument_error_event_with_the_offending_argument(self, console):
        called = []
        rc = console.tool(["--bogus", "--machine"], lambda args, ctx: called.append(1) or 0)
        assert rc == 2
        assert called == []
        assert console.events() == [
            _error_event("bad_argument", 2, "unrecognized arguments: --bogus", field="--bogus")]
        assert console.stderr_text() == ""

    def test_missing_option_value_names_the_long_option_as_field(self, console):
        assert console.tool(["--output", "--machine"], _ok) == 2
        event = console.events()[0]
        assert event["code"] == "bad_argument"
        assert event["field"] == "--output"
        assert event["message"] == "argument -o/--output: expected one argument"

    def test_without_machine_only_one_error_line_is_written_and_no_usage_text(self, console):
        assert console.tool(["--bogus"], _ok) == 2
        assert console.stderr_text() == "error: unrecognized arguments: --bogus\n"
        assert console.stdout_bytes() == b""


class TestTermination:
    def test_keyboard_interrupt_in_machine_mode_ends_the_stream_with_cancelled_and_130(self, console):
        def run(args, ctx):
            ctx.emitter.progress(stage="bake", done=1, total=2)
            raise KeyboardInterrupt

        assert console.tool(["in.vmd", "--machine"], run) == 130
        assert console.events() == [
            {"type": "progress", "stage": "bake", "done": 1, "total": 2},
            _error_event("cancelled", 130, _INTERRUPT_MESSAGE),
        ]
        assert console.stderr_text() == ""

    def test_keyboard_interrupt_without_machine_writes_one_error_line(self, console):
        def run(args, ctx):
            raise KeyboardInterrupt

        assert console.tool(["in.vmd"], run) == 130
        assert console.stderr_text() == f"error: {_INTERRUPT_MESSAGE}\n"
        assert console.stdout_bytes() == b""

    def test_unexpected_exception_in_machine_mode_ends_the_stream_with_internal_error_and_1(self, console):
        def run(args, ctx):
            raise RuntimeError("boom")

        assert console.tool(["in.vmd", "--machine"], run) == 1
        assert console.events() == [_error_event("internal_error", 1, "RuntimeError: boom")]
        assert console.stderr_text() == ""

    def test_unexpected_exception_without_machine_writes_one_error_line_and_no_traceback(self, console):
        def run(args, ctx):
            raise RuntimeError("boom")

        assert console.tool(["in.vmd"], run) == 1
        assert console.stderr_text() == "error: RuntimeError: boom\n"

    def test_exception_after_the_result_keeps_the_stream_single_terminated(self, console):
        def run(args, ctx):
            ctx.emitter.result(mode="process")
            raise RuntimeError("後始末で失敗")

        assert console.tool(["in.vmd", "--machine"], run) == 1
        assert console.events() == [{"type": "result", "mode": "process"}]
        assert console.stderr_text() == "error: RuntimeError: 後始末で失敗\n"

    def test_exception_in_describe_is_an_internal_error(self, console):
        def describe(parser):
            raise RuntimeError("boom")

        assert console.tool(["--describe"], _ok, describe=describe) == 1
        assert console.events() == [_error_event("internal_error", 1, "RuntimeError: boom")]


class _BrokenBuffer:
    def __init__(self, exc):
        self._exc = exc

    def write(self, _data):
        raise self._exc


class _BrokenStdout:
    def __init__(self, exc):
        self.buffer = _BrokenBuffer(exc)

    def write(self, _text):
        return 0

    def flush(self):
        pass


def _emit_progress(args, ctx):
    ctx.emitter.progress(stage="bake", done=0, total=None)
    return 0


def _emit_result(args, ctx):
    ctx.emitter.result(mode="process")
    return 0


def _fail_write(args, ctx):
    return ctx.fail("write_failed", "出力に失敗", 3, field="--output")


def _interrupt(args, ctx):
    raise KeyboardInterrupt


_BROKEN_STDOUT_SCENARIOS = [
    pytest.param(["in.vmd", "--machine"], _emit_progress, 1, "{exc}", id="progress_event"),
    pytest.param(["in.vmd", "--machine"], _emit_result, 1, "{exc}", id="result_event"),
    pytest.param(["--describe"], _ok, 1, "{exc}", id="describe_event"),
    pytest.param(["in.vmd", "--machine"], _fail_write, 3, "出力に失敗", id="reported_failure"),
    pytest.param(["--bogus", "--machine"], _ok, 2, "unrecognized arguments: --bogus", id="usage_error"),
    pytest.param(["in.vmd", "--machine"], _interrupt, 130, _INTERRUPT_MESSAGE, id="interrupt"),
]


class TestUnwritableStdout:
    @pytest.mark.parametrize("exc", [
        pytest.param(OSError("broken pipe"), id="broken_pipe"),
        pytest.param(ValueError("I/O operation on closed file"), id="closed_stdout"),
    ])
    @pytest.mark.parametrize("argv, run, expected_rc, message", _BROKEN_STDOUT_SCENARIOS)
    def test_reason_goes_to_stderr_as_one_line_and_exit_code_is_kept(
            self, monkeypatch, exc, argv, run, expected_rc, message):
        console = _Console(monkeypatch)
        monkeypatch.setattr(sys, "stdout", _BrokenStdout(exc))
        describe = (lambda parser: {}) if argv == ["--describe"] else _describe
        rc = _tool(argv, run, describe=describe)
        assert rc == expected_rc
        expected = message.format(exc=f"{type(exc).__name__}: {exc}")
        assert console.stderr_text() == f"error: {expected}\n"

    def test_stdout_without_a_buffer_fails_before_the_run_with_a_stderr_line(self, monkeypatch):
        console = _Console(monkeypatch)
        monkeypatch.setattr(sys, "stdout", None)
        called = []
        rc = _tool(["in.vmd", "--machine"], lambda args, ctx: called.append(1) or 0)
        assert rc == 1
        assert called == []
        lines = console.stderr_text().splitlines()
        assert len(lines) == 1 and lines[0].startswith("error: AttributeError")


class TestStderrEncoding:
    @staticmethod
    def _cp932_console(monkeypatch):
        stderr = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
        return _Console(monkeypatch, stderr=stderr)

    @staticmethod
    def _stderr_text(console):
        console.stderr.flush()
        return console.stderr.buffer.getvalue().decode("cp932")

    def test_usage_error_with_unencodable_text_still_exits_2_and_escapes_it(self, monkeypatch):
        console = self._cp932_console(monkeypatch)
        assert _tool([f"--{_NON_CP932_CHAR}"], _ok) == 2
        assert self._stderr_text(console) == "error: unrecognized arguments: --\\U0001f3a5\n"

    def test_reported_failure_with_unencodable_text_keeps_its_exit_code(self, monkeypatch):
        console = self._cp932_console(monkeypatch)

        def run(args, ctx):
            return ctx.fail("not_vmd", f"読めない{_NON_CP932_CHAR}", 1)

        assert _tool(["in.vmd"], run) == 1
        assert self._stderr_text(console) == "error: 読めない\\U0001f3a5\n"

    def test_internal_error_with_unencodable_text_exits_1(self, monkeypatch):
        console = self._cp932_console(monkeypatch)

        def run(args, ctx):
            raise RuntimeError(_NON_CP932_CHAR)

        assert _tool(["in.vmd"], run) == 1
        assert self._stderr_text(console) == "error: RuntimeError: \\U0001f3a5\n"

    def test_warning_with_unencodable_text_does_not_stop_the_run(self, monkeypatch):
        console = self._cp932_console(monkeypatch)

        def run(args, ctx):
            ctx.warn("code", f"警告{_NON_CP932_CHAR}")
            return 0

        assert _tool(["in.vmd"], run) == 0
        assert self._stderr_text(console) == "warning: code: 警告\\U0001f3a5\n"

    def test_stderr_without_reconfigure_is_accepted(self, console):
        assert not hasattr(console.stderr, "reconfigure")
        assert console.tool([f"--{_NON_CP932_CHAR}"], _ok) == 2


class _LiveLine:
    TEXT = "[bake] 1/2"
    CLEAR = "\r" + " " * len(TEXT) + "\r"

    def __init__(self, log=None):
        self.closed = 0
        self._log = log
        sys.stderr.write(self.TEXT)

    def close(self):
        self.closed += 1
        if self.closed == 1:
            sys.stderr.write(self.CLEAR)
        if self._log is not None:
            self._log.append("close")


class _LoggingStderr(io.StringIO):
    def __init__(self, log):
        super().__init__()
        self._log = log

    def write(self, text):
        self._log.append(text)
        return super().write(text)


def _fail_boom(args, ctx):
    return ctx.fail("write_failed", "boom", 3)


def _interrupt_run(args, ctx):
    raise KeyboardInterrupt


def _crash(args, ctx):
    raise RuntimeError("boom")


_FAILURE_ROUTES = [
    pytest.param(_fail_boom, 3, "boom", id="reported_failure"),
    pytest.param(_interrupt_run, 130, _INTERRUPT_MESSAGE, id="keyboard_interrupt"),
    pytest.param(_crash, 1, "RuntimeError: boom", id="unexpected_exception"),
]


class TestLiveProgressLine:
    def test_track_progress_returns_the_given_router(self, console):
        seen = {}

        def run(args, ctx):
            router = _LiveLine()
            seen["same"] = ctx.track_progress(router) is router
            return 0

        console.tool(["in.vmd"], run)
        assert seen["same"] is True

    @pytest.mark.parametrize("run_failure, expected_rc, message", _FAILURE_ROUTES)
    def test_live_line_is_closed_before_the_error_line_is_written(
            self, monkeypatch, run_failure, expected_rc, message):
        log = []
        console = _Console(monkeypatch, stderr=_LoggingStderr(log))

        def run(args, ctx):
            ctx.track_progress(_LiveLine(log))
            return run_failure(args, ctx)

        assert _tool(["in.vmd"], run) == expected_rc
        first_error = next(i for i, entry in enumerate(log) if entry.startswith("error:"))
        assert "close" in log[:first_error]
        assert console.stderr_text().endswith(_LiveLine.CLEAR + f"error: {message}\n")

    @pytest.mark.parametrize("run_failure, expected_rc, message", _FAILURE_ROUTES)
    def test_error_line_starts_on_a_cleared_line_and_nothing_follows_it(
            self, console, run_failure, expected_rc, message):
        def run(args, ctx):
            ctx.track_progress(_LiveLine())
            return run_failure(args, ctx)

        assert console.tool(["in.vmd"], run) == expected_rc
        assert console.stderr_text() == _LiveLine.TEXT + _LiveLine.CLEAR + f"error: {message}\n"

    @pytest.mark.parametrize("run_failure, expected_rc, message", _FAILURE_ROUTES)
    def test_router_is_closed_in_machine_mode_failures_too(self, console, run_failure, expected_rc, message):
        routers = []

        def run(args, ctx):
            routers.append(ctx.track_progress(_LiveLine()))
            return run_failure(args, ctx)

        assert console.tool(["in.vmd", "--machine"], run) == expected_rc
        assert routers[0].closed >= 1

    def test_router_is_closed_when_run_returns_normally(self, console):
        routers = []

        def run(args, ctx):
            routers.append(ctx.track_progress(_LiveLine()))
            return 0

        assert console.tool(["in.vmd"], run) == 0
        assert routers[0].closed >= 1
        assert console.stderr_text() == _LiveLine.TEXT + _LiveLine.CLEAR

    def test_every_tracked_router_is_closed(self, console):
        routers = []

        def run(args, ctx):
            routers.append(ctx.track_progress(_LiveLine()))
            routers.append(ctx.track_progress(_LiveLine()))
            return _fail_boom(args, ctx)

        console.tool(["in.vmd"], run)
        assert [router.closed >= 1 for router in routers] == [True, True]

    def test_warning_line_starts_on_a_cleared_line(self, console):
        def run(args, ctx):
            ctx.track_progress(_LiveLine())
            ctx.warn("code", "m")
            return 0

        assert console.tool(["in.vmd"], run) == 0
        assert console.stderr_text().startswith(_LiveLine.TEXT + _LiveLine.CLEAR + "warning: code: m\n")


class TestOutputGuardThroughTheSkeleton:
    @staticmethod
    def _guarded(args, ctx):
        output = args.output if args.output is not None else default_output_path(args.input, "_x.vmd")
        rc = guard_output(output, overwrite=args.overwrite, fail=ctx.fail)
        if rc is not None:
            return rc
        if ctx.emitter is not None:
            ctx.emitter.result(mode="process", output=output)
        return 0

    @pytest.mark.parametrize("extra", [[], ["--dry-run"], ["--verbose"], ["--quiet"]])
    def test_existing_output_without_overwrite_exits_2_with_an_error_event(self, console, tmp_path, extra):
        out = tmp_path / "out.vmd"
        out.write_bytes(b"x")
        assert console.tool(["in.vmd", "-o", str(out), "--machine", *extra], self._guarded) == 2
        events = console.events()
        assert len(events) == 1
        assert events[0]["code"] == "output_exists"
        assert events[0]["exit_code"] == 2
        assert events[0]["field"] == "--output"

    def test_dry_run_without_machine_exits_2_with_one_error_line(self, console, tmp_path):
        out = tmp_path / "out.vmd"
        out.write_bytes(b"x")
        assert console.tool(["in.vmd", "-o", str(out), "--dry-run"], self._guarded) == 2
        lines = console.stderr_text().splitlines()
        assert len(lines) == 1 and lines[0].startswith("error: ") and str(out) in lines[0]

    def test_overwrite_lets_a_dry_run_through(self, console, tmp_path):
        out = tmp_path / "out.vmd"
        out.write_bytes(b"x")
        assert console.tool(["in.vmd", "-o", str(out), "--dry-run", "--overwrite", "--machine"], self._guarded) == 0
        assert console.events() == [{"type": "result", "mode": "process", "output": str(out)}]

    def test_default_output_next_to_the_input_is_guarded(self, console, tmp_path):
        source = tmp_path / "in.vmd"
        source.write_bytes(b"x")
        (tmp_path / "in_x.vmd").write_bytes(b"x")
        assert console.tool([str(source), "--machine"], self._guarded) == 2
        assert console.events()[0]["code"] == "output_exists"


@pytest.mark.skipif(sys.platform != "win32", reason="SIGBREAK は Windows 固有")
class TestSigbreakBridge:
    def test_handler_is_installed_before_run_and_raises_keyboard_interrupt(self, console):
        seen = {}

        def run(args, ctx):
            handler = signal.getsignal(signal.SIGBREAK)
            seen["installed"] = handler not in (signal.SIG_DFL, signal.SIG_IGN, None)
            with pytest.raises(KeyboardInterrupt):
                handler(signal.SIGBREAK, None)
            return 0

        assert console.tool(["in.vmd"], run) == 0
        assert seen["installed"] is True
