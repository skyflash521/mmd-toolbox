import json
import types

import pytest

from cli_resource_watch import watch as _watch_module
from song2vpr import cli
from vocal_analysis.separator import SeparationError

from .support import front_stage_result

_MIB = 2**20


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _events(capsysbinary):
    return [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]


def _force_gpu_oversubscription_from_the_second_probe(monkeypatch):
    seq = iter([(4000 * _MIB, 8192 * _MIB, 0), (4000 * _MIB, 8192 * _MIB, 5600 * _MIB)])
    monkeypatch.setattr(_watch_module, "_default_gpu_probe",
                        lambda: next(seq, (4000 * _MIB, 8192 * _MIB, 5600 * _MIB)))
    monkeypatch.setattr(_watch_module, "_default_ram_probe", lambda: (0, 0, 0))


def _stub_pipeline_reporting_stages(monkeypatch, stages=("load", "separate")):
    def fake_run(input_path, **kwargs):
        for stage in stages:
            kwargs["progress"].stage(stage)
        return front_stage_result()

    monkeypatch.setattr(cli, "_pipeline", types.SimpleNamespace(run=fake_run))


def test_resource_warning_is_emitted_as_a_machine_event_with_observed_values(tmp_path, monkeypatch,
                                                                             capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription_from_the_second_probe(monkeypatch)
    _stub_pipeline_reporting_stages(monkeypatch)

    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr")]) == 0
    warnings = [e for e in _events(capsysbinary)
                if e["type"] == "warning" and e["code"] == "gpu_memory_oversubscribed"]
    assert len(warnings) == 1
    assert warnings[0]["stage"] == "separate"
    assert warnings[0]["reserved_mib"] == 5600
    assert warnings[0]["free_at_start_mib"] == 4000
    assert warnings[0]["total_mib"] == 8192


def test_resource_warning_is_printed_to_stderr_with_values_and_remedy_in_human_mode(tmp_path, monkeypatch,
                                                                                    capsys):
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription_from_the_second_probe(monkeypatch)
    _stub_pipeline_reporting_stages(monkeypatch)

    assert cli.main([src, "-o", str(tmp_path / "out.vpr")]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "warning: gpu_memory_oversubscribed:" in captured.err
    assert "5600MiB" in captured.err and "--device cpu" in captured.err


def test_resource_warning_survives_quiet(tmp_path, monkeypatch, capsys):
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription_from_the_second_probe(monkeypatch)
    _stub_pipeline_reporting_stages(monkeypatch)

    assert cli.main([src, "-o", str(tmp_path / "out.vpr"), "--quiet"]) == 0
    assert "gpu_memory_oversubscribed" in capsys.readouterr().err


def test_human_warning_closes_the_live_line_before_writing(tmp_path, monkeypatch):
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription_from_the_second_probe(monkeypatch)
    _stub_pipeline_reporting_stages(monkeypatch)
    order = []

    class _OrderReporter:
        def stage(self, stage_id, **kwargs):
            pass

        def close(self):
            order.append("close")

        def summary(self, message):
            pass

    class _OrderStderr:
        def write(self, text):
            if text.startswith("warning:"):
                order.append("warning")

        def flush(self):
            pass

    monkeypatch.setattr(cli._progress, "build_router", lambda **kwargs: _OrderReporter())
    monkeypatch.setattr(cli.sys, "stderr", _OrderStderr())

    assert cli.main([src, "-o", str(tmp_path / "out.vpr")]) == 0
    assert order[:2] == ["close", "warning"]


@pytest.mark.parametrize("outcome,expected", [
    pytest.param(None, 0, id="success"),
    pytest.param(KeyboardInterrupt(), 130, id="interrupted"),
    pytest.param(RuntimeError("boom"), 1, id="unexpected_failure"),
    pytest.param(SeparationError("分離に失敗"), 4, id="classified_failure"),
])
def test_live_line_is_closed_on_every_exit(tmp_path, monkeypatch, outcome, expected):
    src = _touch(tmp_path / "in.wav")
    closed = []

    class _Reporter:
        def stage(self, stage_id, **kwargs):
            pass

        def close(self):
            closed.append(True)

        def summary(self, message):
            pass

    def fake_run(input_path, **kwargs):
        if outcome is not None:
            raise outcome
        return front_stage_result()

    monkeypatch.setattr(cli._progress, "build_router", lambda **kwargs: _Reporter())
    monkeypatch.setattr(cli, "_pipeline", types.SimpleNamespace(run=fake_run))

    assert cli.main([src, "-o", str(tmp_path / "out.vpr")]) == expected
    assert closed


def test_resource_warning_does_not_change_the_exit_code(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription_from_the_second_probe(monkeypatch)
    _stub_pipeline_reporting_stages(monkeypatch)

    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr"), "--dry-run"]) == 0


def test_gpu_configuration_warning_precedes_the_first_progress_event_in_the_stream(tmp_path, monkeypatch,
                                                                                   capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "torch_gpu_warning",
                        lambda device: ("cpu_only_torch", {"torch_version": "2.13.0"}))
    _stub_pipeline_reporting_stages(monkeypatch, stages=("load",))

    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr")]) == 0
    kinds = [(e.get("type"), e.get("code")) for e in _events(capsysbinary)]
    assert ("warning", "cpu_only_torch") in kinds
    assert kinds.index(("warning", "cpu_only_torch")) < next(
        i for i, (kind, _) in enumerate(kinds) if kind == "progress")


def test_gpu_configuration_warning_carries_the_torch_version(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "torch_gpu_warning",
                        lambda device: ("cuda_unavailable", {"torch_version": "2.13.0+cu126"}))
    _stub_pipeline_reporting_stages(monkeypatch, stages=("load",))

    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr")]) == 0
    warning = next(e for e in _events(capsysbinary) if e["type"] == "warning")
    assert warning["code"] == "cuda_unavailable"
    assert warning["torch_version"] == "2.13.0+cu126"


def test_no_gpu_configuration_warning_when_the_check_returns_none(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "torch_gpu_warning", lambda device: None)
    _stub_pipeline_reporting_stages(monkeypatch, stages=("load",))

    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr")]) == 0
    gpu_codes = {"cpu_only_torch", "cuda_unavailable"}
    assert not [e for e in _events(capsysbinary)
                if e["type"] == "warning" and e["code"] in gpu_codes]


def test_gpu_configuration_check_receives_the_selected_device(tmp_path, monkeypatch, capsysbinary):
    src = _touch(tmp_path / "in.wav")
    seen = []
    monkeypatch.setattr(cli, "torch_gpu_warning", lambda device: seen.append(device))
    _stub_pipeline_reporting_stages(monkeypatch, stages=("load",))

    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr"), "--device", "cpu"]) == 0
    assert seen == ["cpu"]
