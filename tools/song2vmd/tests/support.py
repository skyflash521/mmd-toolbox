import io
import json
import os
import sys

import pytest

from cli_resource_watch import torch_config as _torch_config
from song2vmd import cli
from song2vmd import events as _events
from song2vmd import pipeline as _pipeline
from song2vmd import report as _report
from vmd import VmdDocument

CHAR_OUTSIDE_CP932 = "\U0001f3a5"


@pytest.fixture
def isolate_gpu_environment(monkeypatch):
    monkeypatch.setattr(_torch_config.shutil, "which", lambda name: None)
    saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    yield
    if saved is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved


def touch(path):
    path.write_bytes(b"")
    return str(path)


def make_result():
    document = VmdDocument(model_name_raw=b"\x00" * 20, morph=[])
    diagnostics = _report.build_diagnostics(
        segments=[], mouth_events=[], mora_event_group_sizes=[],
        event_diagnostics=_events.EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=0),
        backends={"separator": "audio-separator-htdemucs-ft", "recognizer": "openai/whisper-medium"},
        style="pop", separated=True, duration_sec=2.5, keys=3, forced_split=False,
    )
    return _pipeline.PipelineResult(
        document=document, diagnostics=diagnostics, sample_rate=44100, channels=2)


def stub_pipeline(monkeypatch):
    monkeypatch.setattr(cli._pipeline, "run", lambda input_path, **kwargs: make_result())


def strict_cp932_stderr(monkeypatch):
    wrapper = io.TextIOWrapper(io.BytesIO(), encoding="cp932", errors="strict", newline="")
    monkeypatch.setattr(sys, "stderr", wrapper)
    return wrapper


def written_text(wrapper):
    wrapper.flush()
    return wrapper.buffer.getvalue().decode("cp932")


def machine_events(capsysbinary):
    raw = capsysbinary.readouterr().out.decode("utf-8")
    return [json.loads(ln) for ln in raw.split("\n") if ln]
