import builtins

import pytest

from cli_resource_watch import torch_config, torch_gpu_warning


def _fake_torch(monkeypatch, *, version, cuda_build, available):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.version, "cuda", cuda_build)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def _with_nvidia_smi(monkeypatch, present):
    monkeypatch.setattr(torch_config.shutil, "which",
                        lambda name: "nvidia-smi" if present else None)


def test_reports_cpu_only_build(monkeypatch):
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0", cuda_build=None, available=False)

    assert torch_gpu_warning("auto") == ("cpu_only_torch", {"torch_version": "2.13.0"})


def test_reports_unusable_cuda_build(monkeypatch):
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0+cu126", cuda_build="12.6", available=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert torch_gpu_warning("auto") == ("cuda_unavailable", {"torch_version": "2.13.0+cu126"})


def test_unusable_cuda_build_is_silent_when_visible_devices_is_set(monkeypatch):
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0+cu126", cuda_build="12.6", available=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    assert torch_gpu_warning("auto") is None


def test_cpu_only_build_is_reported_even_when_visible_devices_is_set(monkeypatch):
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0", cuda_build=None, available=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    assert torch_gpu_warning("auto") == ("cpu_only_torch", {"torch_version": "2.13.0"})


def test_silent_when_gpu_is_usable(monkeypatch):
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0+cu126", cuda_build="12.6", available=True)

    assert torch_gpu_warning("auto") is None


def test_cpu_only_build_is_silent_without_nvidia_smi(monkeypatch):
    _with_nvidia_smi(monkeypatch, False)
    _fake_torch(monkeypatch, version="2.13.0", cuda_build=None, available=False)

    assert torch_gpu_warning("auto") is None


def test_cpu_mode_is_silent_without_looking_for_nvidia_smi(monkeypatch):
    def _fail(name):
        raise AssertionError("cpu 指定では GPU の有無を調べてはならない")

    monkeypatch.setattr(torch_config.shutil, "which", _fail)

    assert torch_gpu_warning("cpu") is None


@pytest.mark.parametrize("device_mode", ["auto", "gpu", "cuda"])
def test_any_mode_other_than_cpu_is_judged(monkeypatch, device_mode):
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0", cuda_build=None, available=False)

    assert torch_gpu_warning(device_mode) == ("cpu_only_torch", {"torch_version": "2.13.0"})


def test_repeated_calls_return_the_same_warning(monkeypatch):
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0", cuda_build=None, available=False)

    first = torch_gpu_warning("auto")
    assert first == ("cpu_only_torch", {"torch_version": "2.13.0"})
    assert torch_gpu_warning("auto") == first


@pytest.mark.parametrize("error", [
    pytest.param(ImportError("no module"), id="not_installed"),
    pytest.param(OSError("DLL load failed"), id="broken_install"),
])
def test_silent_when_torch_cannot_be_imported(monkeypatch, error):
    _with_nvidia_smi(monkeypatch, True)
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "torch":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    assert torch_gpu_warning("auto") is None
