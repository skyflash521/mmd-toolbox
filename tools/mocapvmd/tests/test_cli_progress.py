import sys

from mocapvmd import cli
from vmd import io

from .helpers import bone, write_vmd

_EXIT_OK = 0
_EXIT_INTERNAL_ERROR = 1


def _write_center_quadratic_curve(path):
    write_vmd(path, bone=[bone("センター", f, pos=(round(0.05 * f * f, 6), 0.0, 0.0)) for f in range(11)])


class _RecordingRouter:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.events = []
        _RecordingRouter.instances.append(self)

    def stage(self, stage_id, **kwargs):
        self.events.append(("stage", stage_id, kwargs))

    def close(self):
        self.events.append(("close",))

    def summary(self, message):
        self.events.append(("summary", message))


def _install(monkeypatch):
    _RecordingRouter.instances = []
    monkeypatch.setattr(cli, "ProgressRouter", _RecordingRouter)


def _stage_ids(router):
    return [e[1] for e in router.events if e[0] == "stage"]


def test_quiet_does_not_change_output(tmp_path):
    src = tmp_path / "in.vmd"
    out_q = tmp_path / "q.vmd"
    out_n = tmp_path / "n.vmd"
    _write_center_quadratic_curve(src)
    assert cli.main([str(src), "-o", str(out_q), "--quiet"]) == _EXIT_OK
    assert cli.main([str(src), "-o", str(out_n)]) == _EXIT_OK
    assert io.read(str(out_q))[0].bone == io.read(str(out_n))[0].bone


def test_router_receives_mode_quiet_stream_and_labels(tmp_path, monkeypatch):
    _install(monkeypatch)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_quadratic_curve(src)

    assert cli.main([str(src), "-o", str(out), "--quiet"]) == _EXIT_OK
    kwargs = _RecordingRouter.instances[-1].kwargs
    assert kwargs["machine"] is False and kwargs["quiet"] is True
    assert kwargs["emitter"] is None and kwargs["stream"] is sys.stderr
    assert kwargs["labels"] == {
        "denoise": "ノイズ軽減", "foot_ik": "足IK接地安定化", "reduce": "キーフレーム圧縮"}

    assert cli.main([str(src), "-o", str(out), "--overwrite"]) == _EXIT_OK
    assert _RecordingRouter.instances[-1].kwargs["quiet"] is False

    assert cli.main([str(src), "-o", str(out), "--overwrite", "--machine"]) == _EXIT_OK
    kwargs = _RecordingRouter.instances[-1].kwargs
    assert kwargs["machine"] is True and kwargs["emitter"] is not None


def test_stages_reported_in_order_reduce_updates_progress_and_summary_comes_last(tmp_path, monkeypatch):
    _install(monkeypatch)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_quadratic_curve(src)
    assert cli.main([str(src), "-o", str(out)]) == _EXIT_OK
    router = _RecordingRouter.instances[-1]
    assert _stage_ids(router)[:3] == ["denoise", "foot_ik", "reduce"]
    reduce_progress_updates = [e for e in router.events if e[0] == "stage" and e[1] == "reduce" and e[2]]
    assert reduce_progress_updates
    assert all("elapsed" in e[2] for e in reduce_progress_updates)
    assert ("close",) in router.events
    assert router.events[-1] == ("summary", f"完了 {out}")


def test_disabled_stages_are_skipped(tmp_path, monkeypatch):
    _install(monkeypatch)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_quadratic_curve(src)
    assert cli.main([str(src), "-o", str(out), "--no-denoise", "--no-foot-ik-stabilize"]) == _EXIT_OK
    assert set(_stage_ids(_RecordingRouter.instances[-1])) == {"reduce"}
    assert cli.main([
        str(src), "-o", str(out), "--overwrite", "--no-denoise", "--no-foot-ik-stabilize", "--no-reduce",
    ]) == _EXIT_OK
    assert _stage_ids(_RecordingRouter.instances[-1]) == []


def test_dry_run_closes_progress_without_summary(tmp_path, monkeypatch):
    _install(monkeypatch)
    src = tmp_path / "in.vmd"
    _write_center_quadratic_curve(src)
    assert cli.main([str(src), "--dry-run"]) == _EXIT_OK
    router = _RecordingRouter.instances[-1]
    assert ("close",) in router.events
    assert not any(e[0] == "summary" for e in router.events)


def test_unexpected_reduce_exception_becomes_internal_error_with_close_and_no_summary(tmp_path, monkeypatch):
    _install(monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cli.reduce, "reduce_bones", _boom)
    src = tmp_path / "in.vmd"
    out = tmp_path / "out.vmd"
    _write_center_quadratic_curve(src)
    rc = cli.main([str(src), "-o", str(out)])
    assert rc == _EXIT_INTERNAL_ERROR
    router = _RecordingRouter.instances[-1]
    assert ("close",) in router.events
    assert not any(e[0] == "summary" for e in router.events)
