import sys

from .argparse_support import ArgumentParseError, argparse_error_field
from .events import EventEmitter
from .failure import emit_failure
from .signals import install_sigbreak_handler
from .warning_line import format_warning_line

_BAD_ARGUMENT_EXIT_CODE = 2
_INTERRUPT_EXIT_CODE = 130


class ToolContext:
    def __init__(self, emitter, machine):
        self.machine = machine
        self.emitter = emitter if machine else None
        self._emitter = emitter
        self._routers = []

    def track_progress(self, router):
        self._routers.append(router)
        return router

    def close_progress(self):
        for router in self._routers:
            try:
                router.close()
            except Exception:
                pass

    def fail(self, code, message, exit_code, *, field=None, path=None, **extra):
        self.close_progress()
        return emit_failure(self._emitter, code=code, message=message, exit_code=exit_code,
                            field=field, path=path, **extra)

    def warn(self, code, message, *, section=None, **extra):
        if self.machine:
            if isinstance(section, str):
                section = [section]
            self.emitter.warning(code=code, message=message,
                                 section=list(section) if section else None, **extra)
            return
        self.close_progress()
        print(format_warning_line(code, message, section=section), file=sys.stderr)


def run_tool(argv, *, build_parser, describe, run):
    emitter = None
    ctx = None
    try:
        if argv is None:
            argv = sys.argv[1:]

        machine = "--machine" in argv
        emitter = EventEmitter(sys.stdout.buffer) if (machine or "--describe" in argv) else None
        ctx = ToolContext(emitter, machine)

        install_sigbreak_handler()
        if hasattr(sys.stderr, "reconfigure"):
            try:
                sys.stderr.reconfigure(errors="backslashreplace")
            except Exception:
                pass
        parser = build_parser()
        try:
            args = parser.parse_args(argv)
        except ArgumentParseError as e:
            return ctx.fail("bad_argument", e.message, _BAD_ARGUMENT_EXIT_CODE,
                            field=argparse_error_field(e.message))
        except SystemExit as e:
            code = e.code
            return code if isinstance(code, int) else (0 if code is None else 2)

        if args.describe:
            emitter.result(mode="describe", **describe(parser))
            return 0

        rc = run(args, ctx)
        ctx.close_progress()
        return rc
    except KeyboardInterrupt:
        return _terminate(ctx, emitter, "cancelled", "中断された(Ctrl-C 等)", _INTERRUPT_EXIT_CODE)
    except Exception as e:
        return _terminate(ctx, emitter, "internal_error", f"{type(e).__name__}: {e}", 1)


def _terminate(ctx, emitter, code, message, exit_code):
    if ctx is not None:
        ctx.close_progress()
    return emit_failure(emitter, code=code, message=message, exit_code=exit_code)
