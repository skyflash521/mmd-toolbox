"""GPU を使えない実行構成の判定の単体テスト。

観測を差し替える引数を持たない判定なので、コマンドの有無・torch の状態・環境変数を差し替えて
検証する。
"""

import builtins

import pytest

from cli_resource_watch import torch_config, torch_gpu_warning


def _fake_torch(monkeypatch, *, version, cuda_build, available):
    """判定が読む torch の属性を差し替える。"""
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch, "__version__", version)
    monkeypatch.setattr(torch.version, "cuda", cuda_build)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available)


def _with_nvidia_smi(monkeypatch, present):
    monkeypatch.setattr(torch_config.shutil, "which",
                        lambda name: "nvidia-smi" if present else None)


def test_reports_cpu_only_build(monkeypatch):
    # CPU 専用版の torch では NVIDIA GPU があっても使えない。利用者が導入手順を踏み損ねた
    # (または別の環境へ入れた)状態で、無警告のまま CPU で処理されるのを防ぐ。
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0", cuda_build=None, available=False)

    assert torch_gpu_warning("auto") == ("cpu_only_torch", {"torch_version": "2.13.0"})


def test_reports_unusable_cuda_build(monkeypatch):
    # CUDA 版が入っているのに CUDA を使えない状態。torch の入れ替えでは直らず、CUDA の
    # バージョン選択をやり直す必要がある。
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0+cu126", cuda_build="12.6", available=False)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    assert torch_gpu_warning("auto") == ("cuda_unavailable", {"torch_version": "2.13.0+cu126"})


def test_silent_when_visible_devices_is_restricted(monkeypatch):
    # 利用者が見せる GPU を自分で絞った結果として使えないことがあり、その構成では
    # 「別の CUDA のバージョンで入れ直せ」が効かない対処になる。
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0+cu126", cuda_build="12.6", available=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    assert torch_gpu_warning("auto") is None


def test_cpu_only_build_is_reported_even_with_restricted_visible_devices(monkeypatch):
    # CPU 専用版であることは可視デバイスの絞り込みと無関係に成立する。
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0", cuda_build=None, available=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")

    assert torch_gpu_warning("auto") == ("cpu_only_torch", {"torch_version": "2.13.0"})


def test_silent_when_gpu_is_usable(monkeypatch):
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0+cu126", cuda_build="12.6", available=True)

    assert torch_gpu_warning("auto") is None


def test_silent_without_nvidia_driver(monkeypatch):
    # nvidia-smi が無い環境(GPU 非搭載機・macOS)では、CPU 専用版が正しい構成なので出さない。
    _with_nvidia_smi(monkeypatch, False)
    _fake_torch(monkeypatch, version="2.13.0", cuda_build=None, available=False)

    assert torch_gpu_warning("auto") is None


def test_silent_when_cpu_is_requested(monkeypatch):
    # 実行デバイスに cpu を選んだ状態は利用者の選択なので、構成の指摘にはならない。
    def _fail(name):
        raise AssertionError("cpu 指定では GPU の有無を調べてはならない")

    monkeypatch.setattr(torch_config.shutil, "which", _fail)

    assert torch_gpu_warning("cpu") is None


@pytest.mark.parametrize("device_mode", ["auto", "gpu", "cuda"])
def test_any_mode_other_than_cpu_is_judged(monkeypatch, device_mode):
    # どんな選択肢を持つかは呼び出し側が決めるので、cpu 以外の値はすべて判定する。
    _with_nvidia_smi(monkeypatch, True)
    _fake_torch(monkeypatch, version="2.13.0", cuda_build=None, available=False)

    assert torch_gpu_warning(device_mode) == ("cpu_only_torch", {"torch_version": "2.13.0"})


@pytest.mark.parametrize("error", [
    ImportError("no module"),
    OSError("DLL load failed"),
])
def test_silent_when_torch_cannot_be_imported(monkeypatch, error):
    # torch を読み込めない環境では GPU 構成を判定できないので、判定を飛ばして処理を続ける。
    # 取り込みは導入の破損で入出力の例外になることもあり、そこで処理前に落ちてはならない。
    _with_nvidia_smi(monkeypatch, True)
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "torch":
            raise error
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    assert torch_gpu_warning("auto") is None
