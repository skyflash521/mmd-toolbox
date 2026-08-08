"""資源逼迫の観測と判定の単体テスト。

GPU超過の算術判定・スワップ検出のノイズ床と帰属条件・1実行1回・観測不能時の縮退と、
進捗ラッパーの委譲(段の切り替わり時だけ判定)を、観測関数を注入した決定論的単体テストで
検証する。実GPU・実psutilは使わない。
"""

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


def _seq_probe(values):
    """呼び出しごとに values の要素を順に返す probe(尽きたら最後の値を返し続ける)。"""
    state = {"i": 0}

    def probe():
        i = min(state["i"], len(values) - 1)
        state["i"] += 1
        return values[i]

    return probe


def _mib_probe(values):
    """MiB単位で書いたタプル列をバイト値の probe にする(probe 契約はバイト値)。"""
    return _seq_probe([
        v if v is None else tuple(x * _MIB for x in v) for v in values
    ])


def _ram_quiet():
    return _mib_probe([(1000, 500, 65457)])


def _gpu_unavailable():
    return _seq_probe([None])


# --- GPU超過 -------------------------------------------------------------------


def test_gpu_oversubscription_fires_once_with_observed_fields():
    emit = _Emit()
    gpu = _mib_probe([(4000, 8192, 0), (4000, 8192, 5600), (4000, 8192, 6000)])
    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=_ram_quiet())

    watch.check("load")       # 基準取得(開始時空き 4000MiB)
    watch.check("separate")   # 予約量の最大 5600 > 4000 → 警告
    watch.check("recognize")  # 2度目は出さない

    assert emit.calls == [("gpu_memory_oversubscribed",
                           {"stage": "separate", "reserved_mib": 5600,
                            "free_at_start_mib": 4000, "total_mib": 8192})]


def test_first_call_only_records_the_baseline():
    # 最初の呼び出しの観測値が既に超過条件を満たしていても報告しない(そこが基準時点になる)。
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


def test_gpu_oversubscription_below_one_mib_still_fires():
    """1MiB未満の超過も取りこぼさない(判定は切り捨て前のバイト値で行う)。"""
    emit = _Emit()
    free_at_start = 4000 * _MIB
    gpu = _seq_probe([(free_at_start, 8192 * _MIB, 0),
                      (free_at_start, 8192 * _MIB, free_at_start + 1)])
    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=_ram_quiet())
    watch.check("load")
    watch.check("separate")
    assert [c[0] for c in emit.calls] == ["gpu_memory_oversubscribed"]
    fields = emit.calls[0][1]
    # 表示は切り捨てMiBなので両者は同値になるが、判定はバイト値なので警告は成立する。
    assert fields["reserved_mib"] == fields["free_at_start_mib"] == 4000


def test_gpu_probe_none_disables_gpu_side_silently():
    emit = _Emit()
    calls = {"n": 0}

    def gpu():
        calls["n"] += 1
        return None

    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=_ram_quiet())
    for stage in ("load", "separate", "recognize"):
        watch.check(stage)
    assert emit.calls == []
    assert calls["n"] == 1  # 無効化後は probe 自体を呼ばない


def test_gpu_probe_exception_disables_gpu_side_silently():
    emit = _Emit()

    def gpu():
        raise RuntimeError("CUDA driver error")

    watch = ResourceWatch(emit, gpu_probe=gpu, ram_probe=_ram_quiet())
    for stage in ("load", "separate"):
        watch.check(stage)  # 例外を外へ漏らさない
    assert emit.calls == []


# --- スワップ検出 --------------------------------------------------------------


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


def test_swap_without_own_process_growth_does_not_fire():
    """他プロセス起因のスワップ(自プロセスの使用量が増えていない)では警告しない(帰属条件)。"""
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


# --- 警告の送出失敗 ------------------------------------------------------------


def test_emit_exception_propagates_to_the_caller():
    """警告の出し先が壊れている事実は観測の失敗と別物なので、握り潰さず呼び出し元へ通す。"""
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
    """送出の失敗は観測の失敗ではないので、その側の観測を無効化しない。"""
    calls = []

    def emit(code, fields):
        calls.append(code)
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


# --- 進捗との接続 --------------------------------------------------------------


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

    assert watch.checked == ["recognize", "rms"]  # 同一段内の進行更新では判定しない
    assert reporter.calls[0] == ("stage", "recognize",
                                 {"done": 0, "total": None, "note": "", "elapsed": 0.0})
    assert reporter.calls[1][2]["note"] == "ダウンロード中"
    assert reporter.calls[2] == ("stage", "rms", {})
    assert reporter.calls[3] == ("close",)
    assert reporter.calls[4] == ("summary", "完了")


def test_wrapper_runs_check_before_reporter_stage():
    """警告発行(ライブ行の消去を伴う)が新しい段の表示より先に行われる順序の検証。"""
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
    """観測関数を指定しないときはモジュールの既定の観測が使われる。"""
    emit = _Emit()
    monkeypatch.setattr(watch_module, "_default_gpu_probe", _seq_probe([None]))
    monkeypatch.setattr(watch_module, "_default_ram_probe", _seq_probe([(0, 0, 0)]))
    watch = ResourceWatch(emit)
    watch.check("load")
    watch.check("separate")
    assert emit.calls == []
