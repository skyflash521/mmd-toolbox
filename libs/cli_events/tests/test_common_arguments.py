import pytest

from cli_events import ArgumentParseError, MachineArgumentParser, add_common_arguments


def _parser(**kwargs):
    parser = MachineArgumentParser(prog="sample", allow_abbrev=False)
    parser.add_argument("input", nargs="?")
    add_common_arguments(parser, version="1.2.3", output_help="出力先(既定: <入力名>_x.vmd)", **kwargs)
    return parser


def test_returns_none_and_registers_on_the_given_parser():
    parser = MachineArgumentParser(prog="sample")
    assert add_common_arguments(parser, version="1.2.3", output_help="出力先") is None
    assert parser.parse_args([]).machine is False


def test_defaults_without_any_flag():
    args = _parser().parse_args([])
    assert args.machine is False
    assert args.describe is False
    assert args.output is None
    assert args.overwrite is False
    assert args.dry_run is False
    assert args.verbose is False
    assert args.quiet is False


@pytest.mark.parametrize("flag, dest", [
    ("--machine", "machine"),
    ("--describe", "describe"),
    ("--overwrite", "overwrite"),
    ("--dry-run", "dry_run"),
    ("--verbose", "verbose"),
    ("-v", "verbose"),
    ("--quiet", "quiet"),
])
def test_boolean_flags_are_store_true(flag, dest):
    args = _parser().parse_args([flag])
    assert getattr(args, dest) is True


@pytest.mark.parametrize("flag", ["-o", "--output"])
def test_output_takes_one_value_as_string(flag):
    assert _parser().parse_args([flag, "out.vmd"]).output == "out.vmd"


def test_output_without_value_is_a_usage_error():
    with pytest.raises(ArgumentParseError):
        _parser().parse_args(["--output"])


def test_option_strings_are_exactly_the_documented_spellings():
    registered = set(_parser()._option_string_actions)
    assert {"--machine", "--describe", "--version", "-o", "--output", "--overwrite", "--dry-run",
            "--verbose", "-v", "--quiet"} <= registered


def test_flags_are_not_mutually_exclusive():
    args = _parser().parse_args(
        ["in", "--machine", "--quiet", "--verbose", "--dry-run", "--overwrite", "-o", "x"])
    assert (args.machine, args.quiet, args.verbose, args.dry_run, args.overwrite) == (
        True, True, True, True, True)
    assert args.input == "in"


def test_version_prints_prog_and_version_on_stdout_and_exits_zero(capsys):
    with pytest.raises(SystemExit) as excinfo:
        _parser().parse_args(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out == "sample 1.2.3\n"


def test_version_does_not_add_a_namespace_attribute():
    assert not hasattr(_parser().parse_args([]), "version")


def test_every_common_option_has_a_nonempty_help_and_output_help_is_the_given_text():
    helps = {}
    for action in _parser()._actions:
        for option in action.option_strings:
            helps[option] = action.help
    for option in ("--machine", "--describe", "--version", "--output", "--overwrite", "--dry-run",
                   "--verbose", "--quiet"):
        assert isinstance(helps[option], str) and helps[option]
    assert helps["--output"] == "出力先(既定: <入力名>_x.vmd)"


def test_quiet_can_be_left_out_for_tools_without_live_progress():
    parser = _parser(quiet=False)
    assert "--quiet" not in parser._option_string_actions
    assert not hasattr(parser.parse_args([]), "quiet")
    with pytest.raises(ArgumentParseError):
        parser.parse_args(["--quiet"])


def test_quiet_is_registered_by_default():
    assert "--quiet" in _parser()._option_string_actions


def test_negated_forms_are_not_registered():
    registered = set(_parser()._option_string_actions)
    assert not {"--no-machine", "--no-overwrite", "--no-dry-run", "--no-verbose", "--no-quiet"} & registered


def test_tool_specific_arguments_can_be_added_before_and_after():
    parser = MachineArgumentParser(prog="sample", allow_abbrev=False)
    parser.add_argument("--before", type=int, default=1)
    add_common_arguments(parser, version="1.2.3", output_help="出力先")
    parser.add_argument("--after", type=int, default=2)
    args = parser.parse_args(["--before", "5", "--after", "6", "--dry-run"])
    assert (args.before, args.after, args.dry_run) == (5, 6, True)


def _helps(parser):
    helps = {}
    for action in parser._actions:
        for option in action.option_strings:
            helps[option] = action.help
    return helps


def test_help_overrides_replace_only_the_given_flags():
    default = _helps(_parser())
    overridden = _helps(_parser(help_overrides={
        "--verbose": "通常実行でも dry-run と同じ診断を表示する(出力vprは書く)",
        "--dry-run": "出力vprを書かず診断だけ表示する",
    }))
    assert overridden["--verbose"] == "通常実行でも dry-run と同じ診断を表示する(出力vprは書く)"
    assert overridden["-v"] == overridden["--verbose"]
    assert overridden["--dry-run"] == "出力vprを書かず診断だけ表示する"
    for flag in ("--machine", "--describe", "--version", "--overwrite", "--quiet", "--output"):
        assert overridden[flag] == default[flag]


def test_help_overrides_none_and_empty_keep_the_defaults():
    default = _helps(_parser())
    assert _helps(_parser(help_overrides=None)) == default
    assert _helps(_parser(help_overrides={})) == default


def test_help_overrides_unknown_key_is_a_value_error():
    with pytest.raises(ValueError, match="--verbos"):
        _parser(help_overrides={"--verbos": "x"})
    with pytest.raises(ValueError):
        _parser(help_overrides={"-v": "x"})
