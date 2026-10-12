import gc
import sys

import pytest


def _byte_bar_download(updates, *, total=100, initial=0, disable=False, count_bar=True):
    def fake_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
        import tqdm.auto

        assert issubclass(tqdm_class, tqdm.auto.tqdm)
        if count_bar:
            file_bar = tqdm_class(total=2, desc="Fetching 2 files")
            file_bar.update(1)
            file_bar.update(1)
            file_bar.close()
        byte_bar = tqdm_class(
            total=total, initial=initial, desc="Downloading", unit="B", unit_scale=True, disable=disable)
        for n in updates:
            byte_bar.update(n)
        byte_bar.close()

    return fake_snapshot_download


def test_select_device_returns_cuda_when_available(monkeypatch):
    torch = pytest.importorskip("torch")
    from vocal_analysis import model_store

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    assert model_store.select_device() == "cuda"


def test_select_device_returns_cpu_when_gpu_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    from vocal_analysis import model_store

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    assert model_store.select_device() == "cpu"


def test_select_dtype_uses_float16_on_cuda_and_float32_on_cpu():
    torch = pytest.importorskip("torch")
    from vocal_analysis import model_store

    assert model_store.select_dtype("cuda") == torch.float16
    assert model_store.select_dtype("cpu") == torch.float32


def test_hf_snapshot_download_forwards_arguments_and_ignores_non_pytorch_weights(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from vocal_analysis import model_store

    captured = {}

    def fake_snapshot_download(repo_id, *, revision=None, tqdm_class=None, ignore_patterns=None):
        captured.update(
            repo_id=repo_id, revision=revision, tqdm_class=tqdm_class, ignore_patterns=ignore_patterns)
        return "snapshot-dir"

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    sentinel_class = object()

    result = model_store.hf_snapshot_download("org/model", revision="rev1", tqdm_class=sentinel_class)

    assert result == "snapshot-dir"
    assert captured["repo_id"] == "org/model"
    assert captured["revision"] == "rev1"
    assert captured["tqdm_class"] is sentinel_class
    assert captured["ignore_patterns"] == model_store.PYTORCH_IGNORE_PATTERNS
    for pattern in ("*.h5", "*.msgpack", "*.tflite", "*.onnx", "*.ot", "*.mlmodel"):
        assert pattern in model_store.PYTORCH_IGNORE_PATTERNS


def test_hf_snapshot_download_accepts_explicit_ignore_patterns(monkeypatch):
    huggingface_hub = pytest.importorskip("huggingface_hub")
    from vocal_analysis import model_store

    captured = {}
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download",
        lambda repo_id, **kwargs: captured.update(kwargs))

    model_store.hf_snapshot_download("org/model", ignore_patterns=["*.bin"])

    assert captured["ignore_patterns"] == ["*.bin"]


def test_prefetch_relays_byte_download_percentage_and_ignores_file_count_bar(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from vocal_analysis import model_store

    monkeypatch.setattr(model_store, "hf_snapshot_download", _byte_bar_download([40, 60]))
    notes = []

    downloaded = model_store.prefetch_with_progress("org/model", "rev1", notes.append)

    assert downloaded is True
    assert notes == ["ダウンロード中: org/model 40%", "ダウンロード中: org/model 100%"]


def test_prefetch_passes_repo_and_revision_to_download(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from vocal_analysis import model_store

    calls = []

    def fake_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
        calls.append((repo_id, revision))

    monkeypatch.setattr(model_store, "hf_snapshot_download", fake_snapshot_download)

    model_store.prefetch_with_progress("org/model", None, lambda note: None)

    assert calls == [("org/model", None)]


def test_prefetch_reports_no_transfer_when_already_cached(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from vocal_analysis import model_store

    monkeypatch.setattr(model_store, "hf_snapshot_download", _byte_bar_download([]))
    notes = []

    downloaded = model_store.prefetch_with_progress("org/model", "main", notes.append)

    assert downloaded is False
    assert notes == []


def test_prefetch_counts_from_initial_value_of_resumed_download(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from vocal_analysis import model_store

    monkeypatch.setattr(model_store, "hf_snapshot_download", _byte_bar_download([25], initial=50))
    notes = []

    model_store.prefetch_with_progress("org/model", "main", notes.append)

    assert notes == ["ダウンロード中: org/model 75%"]


def test_prefetch_relays_percentage_even_when_tqdm_display_is_disabled(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from vocal_analysis import model_store

    monkeypatch.setattr(model_store, "hf_snapshot_download", _byte_bar_download([30, 70], disable=True))
    notes = []

    downloaded = model_store.prefetch_with_progress("org/model", "main", notes.append)

    assert downloaded is True
    assert notes == ["ダウンロード中: org/model 30%", "ダウンロード中: org/model 100%"]


def test_load_with_progress_orders_download_notes_loading_note_build_and_clear(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from vocal_analysis import model_store

    timeline = []
    base = _byte_bar_download([100])

    def fake_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
        timeline.append(("snapshot", repo_id, revision))
        base(repo_id, revision=revision, tqdm_class=tqdm_class)

    monkeypatch.setattr(model_store, "hf_snapshot_download", fake_snapshot_download)

    def build():
        timeline.append("build")
        return "MODEL"

    result = model_store.load_with_progress(
        "org/model", "rev1",
        on_progress=lambda note: timeline.append(("on_progress", note)),
        loading_note="読み込み中: org/model", build=build)

    assert result == "MODEL"
    assert timeline == [
        ("snapshot", "org/model", "rev1"),
        ("on_progress", "ダウンロード中: org/model 100%"),
        ("on_progress", "読み込み中: org/model"),
        "build",
        ("on_progress", ""),
    ]


def test_load_with_progress_shows_loading_note_but_no_clear_when_already_cached(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from vocal_analysis import model_store

    monkeypatch.setattr(model_store, "hf_snapshot_download", _byte_bar_download([]))
    notes = []

    model_store.load_with_progress(
        "org/model", "main", on_progress=notes.append, loading_note="読み込み中: org/model",
        build=lambda: None)

    assert notes == ["読み込み中: org/model"]


def test_load_with_progress_without_on_progress_neither_downloads_ahead_nor_reports(monkeypatch):
    from vocal_analysis import model_store

    def fail_download(*args, **kwargs):
        raise AssertionError("on_progress が無いときは事前取得しない")

    monkeypatch.setattr(model_store, "hf_snapshot_download", fail_download)

    result = model_store.load_with_progress(
        "org/model", "main", on_progress=None, loading_note="読み込み中", build=lambda: "MODEL")

    assert result == "MODEL"


def test_load_with_progress_clears_download_note_when_build_fails(monkeypatch):
    pytest.importorskip("huggingface_hub")
    from vocal_analysis import model_store

    monkeypatch.setattr(model_store, "hf_snapshot_download", _byte_bar_download([100]))
    notes = []

    def build():
        raise OSError("broken weights")

    with pytest.raises(OSError, match="broken weights"):
        model_store.load_with_progress(
            "org/model", "main", on_progress=notes.append, loading_note="読み込み中", build=build)

    assert notes[-1] == ""


def test_release_gpu_memory_collects_garbage_before_emptying_cuda_cache(monkeypatch):
    torch = pytest.importorskip("torch")
    from vocal_analysis import model_store

    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    model_store.release_gpu_memory()

    assert calls == ["gc", "empty_cache"]


def test_release_gpu_memory_skips_cuda_cache_when_gpu_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    from vocal_analysis import model_store

    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: calls.append("gc"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty_cache"))

    model_store.release_gpu_memory()

    assert calls == ["gc"]


def test_release_gpu_memory_tolerates_missing_torch(monkeypatch):
    from vocal_analysis import model_store

    calls = []
    monkeypatch.setattr(gc, "collect", lambda *a, **k: calls.append("gc"))
    monkeypatch.setitem(sys.modules, "torch", None)

    model_store.release_gpu_memory()

    assert calls == ["gc"]
