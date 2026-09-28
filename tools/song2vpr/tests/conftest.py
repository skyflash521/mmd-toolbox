import os

import pytest

from cli_events.testing import restore_sigbreak_handler as _restore_sigbreak_handler  # noqa: F401
from cli_resource_watch import torch_config as _torch_config


@pytest.fixture(autouse=True)
def _isolate_gpu_environment(monkeypatch):
    monkeypatch.setattr(_torch_config.shutil, "which", lambda name: None)
    saved = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    yield
    if saved is None:
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = saved
