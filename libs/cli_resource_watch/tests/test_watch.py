import builtins

import pytest

from cli_resource_watch import ProgressWithResourceCheck, ResourceWatch
from cli_resource_watch import watch as watch_module
from cli_resource_watch.watch import PROCESS_GROWTH_FLOOR_MIB, SWAP_INCREASE_FLOOR_MIB

_MIB = 2**20


class _Emit:
    def __init__(self):
        self.calls = []

    def __call__(self, code, fields):
        self.calls.append((code, fields))


def _probe_repeating_last(values):
    state = {"i": 0}

    def probe():
        i = min(state["i"], len(values) - 1)
        state["i"] += 1
        return values[i]

    return probe


def _mib_probe(values):
    return _probe_repeating_last([
        v if v is None else tuple(x * _MIB for x in v) for v in values
    ])


def _ram_quiet():
    return _mib_probe([(1000, 500, 65457)])


def _gpu_unavailable():
    return _probe_repeating_last([None])


def test_gpu_oversubscription_fires_once_with_observed_fields():
    emit = _Emit()
    gpu = _mib_probe([(4000, 8192, 0), (4000, 8192, 5600), (4000, 8192, 6000)])
    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=_ram_quiet())

    watch.check("load")
    watch.check("separate")
    watch.check("recognize")

    assert emit.calls == [("gpu_memory_oversubscribed",
                           {"stage": "separate", "reserved_mib": 5600,
                            "free_at_start_mib": 4000, "total_mib": 8192})]


def test_first_call_only_records_the_baseline_even_when_conditions_hold():
    emit = _Emit()
    gpu = _mib_probe([(4000, 8192, 5600), (4000, 8192, 5600)])
    ram = _mib_probe([
        (1000, 500, 65457),
        (1000 + SWAP_INCREASE_FLOOR_MIB, 500 + PROCESS_GROWTH_FLOOR_MIB, 65457),
    ])
    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=ram)

    watch.check("load")
    assert emit.calls == []
    watch.check("separate")
    assert [c[0] for c in emit.calls] == ["gpu_memory_oversubscribed", "swap_detected"]


def test_gpu_no_warning_when_reserved_fits_free_at_start():
    emit = _Emit()
    gpu = _mib_probe([(4000, 8192, 0), (4000, 8192, 4000), (100, 8192, 3999)])
    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=_ram_quiet())
    for stage in ("load", "separate", "recognize"):
        watch.check(stage)
    assert emit.calls == []


def test_gpu_oversubscription_by_one_byte_fires_though_mib_fields_are_equal():
    emit = _Emit()
    free_at_start = 4000 * _MIB
    gpu = _probe_repeating_last([(free_at_start, 8192 * _MIB, 0),
                                 (free_at_start, 8192 * _MIB, free_at_start + 1)])
    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=_ram_quiet())
    watch.check("load")
    watch.check("separate")
    assert [c[0] for c in emit.calls] == ["gpu_memory_oversubscribed"]
    fields = emit.calls[0][1]
    assert fields["reserved_mib"] == fields["free_at_start_mib"] == 4000


def test_gpu_probe_none_disables_gpu_side_without_probing_again():
    emit = _Emit()
    calls = {"n": 0}

    def gpu():
        calls["n"] += 1
        return None

    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=_ram_quiet())
    for stage in ("load", "separate", "recognize"):
        watch.check(stage)
    assert emit.calls == []
    assert calls["n"] == 1


def test_gpu_probe_exception_disables_gpu_side_silently():
    emit = _Emit()

    def gpu():
        raise RuntimeError("CUDA driver error")

    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=_ram_quiet())
    for stage in ("load", "separate"):
        watch.check(stage)
    assert emit.calls == []


def test_swap_detection_fires_once_with_observed_fields():
    emit = _Emit()
    ram = _mib_probe([
        (1000, 500, 65457),
        (1000 + SWAP_INCREASE_FLOOR_MIB, 500 + PROCESS_GROWTH_FLOOR_MIB, 65457),
        (9000, 20000, 65457),
    ])
    watch = ResourceWatch(emit, gpu_probe=_gpu_unavailable(), ram_probe=ram)

    watch.check("load")
    watch.check("separate")
    watch.check("recognize")

    assert emit.calls == [("swap_detected",
                           {"stage": "separate", "swap_increase_mib": SWAP_INCREASE_FLOOR_MIB,
                            "process_rss_mib": 500 + PROCESS_GROWTH_FLOOR_MIB,
                            "ram_total_mib": 65457})]


def test_swap_increase_below_floor_does_not_fire():
    emit = _Emit()
    ram = _mib_probe([
        (1000, 500, 65457),
        (1000 + SWAP_INCREASE_FLOOR_MIB - 1, 500 + PROCESS_GROWTH_FLOOR_MIB, 65457),
    ])
    watch = ResourceWatch(emit, gpu_probe=_gpu_unavailable(), ram_probe=ram)
    watch.check("load")
    watch.check("separate")
    assert emit.calls == []


def test_process_growth_below_floor_does_not_fire():
    emit = _Emit()
    ram = _mib_probe([
        (1000, 500, 65457),
        (1000 + SWAP_INCREASE_FLOOR_MIB, 500 + PROCESS_GROWTH_FLOOR_MIB - 1, 65457),
    ])
    watch = ResourceWatch(emit, gpu_probe=_gpu_unavailable(), ram_probe=ram)
    watch.check("load")
    watch.check("separate")
    assert emit.calls == []


def test_swap_without_own_process_growth_does_not_fire():
    emit = _Emit()
    ram = _mib_probe([
        (1000, 500, 65457),
        (1000 + SWAP_INCREASE_FLOOR_MIB * 10, 500, 65457),
    ])
    watch = ResourceWatch(emit, gpu_probe=_gpu_unavailable(), ram_probe=ram)
    watch.check("load")
    watch.check("separate")
    assert emit.calls == []


def test_ram_probe_exception_disables_ram_side_and_keeps_gpu_side():
    emit = _Emit()

    def ram():
        raise ImportError("psutil 未導入")

    gpu = _mib_probe([(4000, 8192, 0), (4000, 8192, 5600)])
    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=ram)
    watch.check("load")
    watch.check("separate")
    assert [c[0] for c in emit.calls] == ["gpu_memory_oversubscribed"]


def test_emit_exception_propagates_to_the_caller():
    error = RuntimeError("閉じたストリームへの書き込み")

    def emit(code, fields):
        raise error

    gpu = _mib_probe([(4000, 8192, 0), (4000, 8192, 5600)])
    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=_ram_quiet())
    watch.check("load")
    with pytest.raises(RuntimeError) as exc:
        watch.check("separate")
    assert exc.value is error


def test_emit_exception_does_not_disable_the_other_side():
    calls = []

    def emit(code, fields):
        calls.append(code)
        if len(calls) == 1:
            raise RuntimeError("閉じたストリームへの書き込み")

    gpu = _mib_probe([(4000, 8192, 0), (4000, 8192, 5600)])
    ram = _mib_probe([
        (1000, 500, 65457),
        (1000 + SWAP_INCREASE_FLOOR_MIB, 500 + PROCESS_GROWTH_FLOOR_MIB, 65457),
    ])
    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=ram)
    watch.check("load")
    with pytest.raises(RuntimeError):
        watch.check("separate")
    assert calls == ["gpu_memory_oversubscribed"]
    watch.check("recognize")
    assert calls == ["gpu_memory_oversubscribed", "swap_detected"]


class _FakeReporter:
    def __init__(self):
        self.calls = []

    def stage(self, stage_id, **kwargs):
        self.calls.append(("stage", stage_id, kwargs))

    def close(self):
        self.calls.append(("close",))

    def summary(self, message):
        self.calls.append(("summary", message))


class _FakeWatch:
    def __init__(self):
        self.checked = []

    def check(self, stage_id):
        self.checked.append(stage_id)


def test_wrapper_checks_only_on_stage_change_and_delegates():
    reporter = _FakeReporter()
    watch = _FakeWatch()
    progress = ProgressWithResourceCheck(reporter, watch)

    progress.stage("recognize", done=0, total=None, note="", elapsed=0.0)
    progress.stage("recognize", done=0, total=None, note="ダウンロード中", elapsed=1.0)
    progress.stage("rms")
    progress.close()
    progress.summary("完了")

    assert watch.checked == ["recognize", "rms"]
    assert reporter.calls[0] == ("stage", "recognize",
                                 {"done": 0, "total": None, "note": "", "elapsed": 0.0})
    assert reporter.calls[1][2]["note"] == "ダウンロード中"
    assert reporter.calls[2] == ("stage", "rms", {})
    assert reporter.calls[3] == ("close",)
    assert reporter.calls[4] == ("summary", "完了")


def test_wrapper_runs_check_before_reporter_stage():
    order = []

    class _Reporter:
        def stage(self, stage_id, **kwargs):
            order.append(("stage", stage_id))

    class _Watch:
        def check(self, stage_id):
            order.append(("check", stage_id))

    progress = ProgressWithResourceCheck(_Reporter(), _Watch())
    progress.stage("load")
    assert order == [("check", "load"), ("stage", "load")]


def test_default_probes_are_used_when_not_injected(monkeypatch):
    emit = _Emit()
    monkeypatch.setattr(watch_module, "_default_gpu_probe", _probe_repeating_last([None]))
    monkeypatch.setattr(watch_module, "_default_ram_probe", _probe_repeating_last([(0, 0, 0)]))
    watch = ResourceWatch(emit)
    watch.check("load")
    watch.check("separate")
    assert emit.calls == []


def test_default_probes_stay_silent_without_torch_and_psutil(monkeypatch):
    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name in ("torch", "psutil"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)
    emit = _Emit()
    watch = ResourceWatch(emit)
    watch.check("load")
    watch.check("separate")
    assert emit.calls == []


def test_default_gpu_probe_is_none_when_cuda_is_unavailable(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert watch_module._default_gpu_probe() is None


def test_default_gpu_probe_returns_free_total_and_peak_reserved_bytes(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (3 * _MIB, 8 * _MIB))
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 5 * _MIB)
    assert watch_module._default_gpu_probe() == (3 * _MIB, 8 * _MIB, 5 * _MIB)


def test_default_ram_probe_sums_rss_over_process_tree_skipping_vanished(monkeypatch):
    psutil = pytest.importorskip("psutil")

    class _Proc:
        def __init__(self, rss, children=()):
            self._rss = rss
            self._children = list(children)

        def memory_info(self):
            if self._rss is None:
                raise psutil.NoSuchProcess(0)
            return type("MemInfo", (), {"rss": self._rss})()

        def children(self, recursive):
            assert recursive
            return self._children

    root = _Proc(100, children=[_Proc(20), _Proc(None), _Proc(3)])
    monkeypatch.setattr(psutil, "Process", lambda: root)
    monkeypatch.setattr(psutil, "swap_memory", lambda: type("Swap", (), {"used": 7})())
    monkeypatch.setattr(psutil, "virtual_memory", lambda: type("Vm", (), {"total": 900})())
    assert watch_module._default_ram_probe() == (7, 123, 900)
