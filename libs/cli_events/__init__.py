from .argparse_support import (
    ArgumentParseError,
    MachineArgumentParser,
    argparse_error_event,
    argparse_error_field,
)
from .common_arguments import add_common_arguments
from .events import EVENT_TYPES, EventEmitter, StreamTerminatedError, error_event
from .failure import emit_failure
from .output_guard import default_output_path, guard_output
from .run import ToolContext, run_tool
from .signals import install_sigbreak_handler
from .warning_line import format_warning_line

__all__ = [
    "ArgumentParseError",
    "EVENT_TYPES",
    "EventEmitter",
    "MachineArgumentParser",
    "StreamTerminatedError",
    "ToolContext",
    "add_common_arguments",
    "argparse_error_event",
    "argparse_error_field",
    "default_output_path",
    "emit_failure",
    "error_event",
    "format_warning_line",
    "guard_output",
    "install_sigbreak_handler",
    "run_tool",
]
