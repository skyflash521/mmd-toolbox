"""song2vpr CLI の警告の接続のテスト。

資源逼迫(GPUメモリの超過・スワップ)と GPU を使えない構成の判定は共有の cli_resource_watch が持つ。
ここでは song2vpr 側の接続——判定を呼ぶ位置、成立した警告を機械モードの warning イベントと人間向け
1行へ振り分けること、警告が出力・終了コード・診断を変えないこと——を検証する。
"""

import json
import types

import pytest

from cli_resource_watch import watch as _watch_module
from song2vpr import cli

pytestmark = pytest.mark.xfail(reason="警告の判定を呼ぶ接続がまだ無い", strict=True)

_MIB = 2**20


def _touch(path):
    path.write_bytes(b"")
    return str(path)


def _events(capsysbinary):
    return [json.loads(ln) for ln in capsysbinary.readouterr().out.decode("utf-8").split("\n") if ln]


def _force_gpu_oversubscription(monkeypatch):
    """GPU の観測を、2回目の判定で超過が成立する系列(バイト値)へ差し替える。"""
    seq = iter([(4000 * _MIB, 8192 * _MIB, 0), (4000 * _MIB, 8192 * _MIB, 5600 * _MIB)])
    monkeypatch.setattr(_watch_module, "_default_gpu_probe",
                        lambda: next(seq, (4000 * _MIB, 8192 * _MIB, 5600 * _MIB)))
    monkeypatch.setattr(_watch_module, "_default_ram_probe", lambda: (0, 0, 0))


def _stub_pipeline_reporting_stages(monkeypatch, stages=("load", "separate")):
    """パイプラインを、渡された中継先へ段を報告するだけのスタブへ差し替える。"""
    def fake_run(input_path, **kwargs):
        for stage in stages:
            kwargs["progress"].stage(stage)
        return None

    monkeypatch.setattr(cli, "_pipeline", types.SimpleNamespace(run=fake_run))


# --- 資源逼迫の警告 ----------------------------------------------------------


def test_resource_warning_is_emitted_as_a_machine_event(tmp_path, monkeypatch, capsysbinary):
    """成立した警告は、観測値を載せた warning イベントで出る。"""
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription(monkeypatch)
    _stub_pipeline_reporting_stages(monkeypatch)

    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr")]) == 0
    warnings = [e for e in _events(capsysbinary)
                if e["type"] == "warning" and e["code"] == "gpu_memory_oversubscribed"]
    assert len(warnings) == 1
    assert warnings[0]["stage"] == "separate"
    assert warnings[0]["reserved_mib"] == 5600
    assert warnings[0]["free_at_start_mib"] == 4000
    assert warnings[0]["total_mib"] == 8192


def test_resource_warning_is_printed_to_stderr_in_human_mode(tmp_path, monkeypatch, capsys):
    """非機械モードでは標準エラーへ1行出す(観測値と対処を添える)。"""
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription(monkeypatch)
    _stub_pipeline_reporting_stages(monkeypatch)

    assert cli.main([src, "-o", str(tmp_path / "out.vpr")]) == 0
    captured = capsys.readouterr()
    assert captured.out == ""  # 人間向けの警告は標準出力を汚さない
    assert "warning: gpu_memory_oversubscribed:" in captured.err
    assert "5600MiB" in captured.err and "--device cpu" in captured.err


def test_resource_warning_survives_quiet(tmp_path, monkeypatch, capsys):
    """--quiet は進捗表示だけを抑制する(警告は抑制しない)。"""
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription(monkeypatch)
    _stub_pipeline_reporting_stages(monkeypatch)

    assert cli.main([src, "-o", str(tmp_path / "out.vpr"), "--quiet"]) == 0
    assert "gpu_memory_oversubscribed" in capsys.readouterr().err


def test_human_warning_closes_the_live_line_before_writing(tmp_path, monkeypatch):
    """人間向けの警告は、ライブ進捗行の消去→標準エラーへの書き込みの順で出す(混線の防止)。"""
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription(monkeypatch)
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


def test_resource_warning_does_not_change_the_exit_code(tmp_path, monkeypatch, capsysbinary):
    """警告は副作用専用で、終了コードを変えない。"""
    src = _touch(tmp_path / "in.wav")
    _force_gpu_oversubscription(monkeypatch)
    _stub_pipeline_reporting_stages(monkeypatch)

    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr"), "--dry-run"]) == 0


# --- GPU を使えない構成の警告 ------------------------------------------------


def test_gpu_configuration_warning_precedes_the_processing(tmp_path, monkeypatch, capsysbinary):
    """数分かかる処理を終えてから伝えても手遅れなので、処理の開始より前に出す。

    判定の呼び出し順ではなくイベントストリーム上の並びで固定する(判定だけ先に行い、送出を後ろへ
    動かす実装を通さないため)。
    """
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


def test_no_gpu_configuration_warning_when_it_does_not_hold(tmp_path, monkeypatch, capsysbinary):
    """判定が何も返さない構成では警告を出さない。"""
    src = _touch(tmp_path / "in.wav")
    monkeypatch.setattr(cli, "torch_gpu_warning", lambda device: None)
    _stub_pipeline_reporting_stages(monkeypatch, stages=("load",))

    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr")]) == 0
    assert not [e for e in _events(capsysbinary) if e["type"] == "warning"]


def test_gpu_configuration_check_receives_the_selected_device(tmp_path, monkeypatch, capsysbinary):
    """判定へ渡すのは利用者が選んだ実行デバイス(CPU 実行では判定しないと共有側が決める)。"""
    src = _touch(tmp_path / "in.wav")
    seen = []
    monkeypatch.setattr(cli, "torch_gpu_warning", lambda device: seen.append(device))
    _stub_pipeline_reporting_stages(monkeypatch, stages=("load",))

    assert cli.main(["--machine", src, "-o", str(tmp_path / "out.vpr"), "--device", "cpu"]) == 0
    assert seen == ["cpu"]
