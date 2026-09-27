import io
import sys
import threading
import time

import pytest

from cli_progress import progress

ProgressReporter = progress.ProgressReporter


class _TTYStream(io.StringIO):
    def isatty(self):
        return True


class _RaisingStream(io.StringIO):
    def __init__(self, exc=None):
        super().__init__()
        self._exc = exc if exc is not None else OSError("write failed")

    def isatty(self):
        return True

    def write(self, _s):
        raise self._exc


class _WriteBlockingStream:
    def __init__(self):
        self.write_started = threading.Event()
        self.release = threading.Event()

    def write(self, s):
        self.write_started.set()
        self.release.wait(2.0)

    def flush(self):
        pass

    def isatty(self):
        return True


def _wait_until(pred, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met within timeout")


def test_format_line_shows_label_count_elapsed():
    assert progress._format_line("キーフレーム圧縮", 5, 88, 73.0) == "[キーフレーム圧縮] 5/88 経過 1:13"


def test_format_line_hides_count_when_total_unset():
    assert progress._format_line("キーフレーム圧縮", 0, None, 2.0) == "[キーフレーム圧縮] 経過 0:02"


def test_format_line_appends_note_only_when_nonempty():
    assert (progress._format_line("キーフレーム圧縮", 60, 60, 5.0, "ボーン: センター")
            == "[キーフレーム圧縮] 60/60 経過 0:05 ボーン: センター")
    assert progress._format_line("キーフレーム圧縮", 60, 60, 5.0, "") == "[キーフレーム圧縮] 60/60 経過 0:05"


def test_display_width_counts_fullwidth_as_two_and_halfwidth_as_one():
    assert progress._display_width("キーフレーム圧縮") == 16
    assert progress._display_width("[X] 0/5") == 7


def test_disabled_is_noop_no_thread_no_output():
    stream = io.StringIO()
    r = ProgressReporter(stream, enabled=False, now=lambda: 0.0)
    r.stage("キーフレーム圧縮")
    r.update(1, 2)
    r.update(1, 2, note="ボーン: センター")
    r.close()
    r.summary("完了 out.vmd")
    assert stream.getvalue() == ""
    assert r._thread is None


def test_auto_enabled_follows_isatty():
    assert ProgressReporter(_TTYStream(), now=lambda: 0.0).enabled is True
    assert ProgressReporter(io.StringIO(), now=lambda: 0.0).enabled is False


def test_default_stream_is_stderr(monkeypatch):
    stderr = _TTYStream()
    monkeypatch.setattr(sys, "stderr", stderr)
    r = ProgressReporter(now=lambda: 0.0)
    r.summary("完了 out.vmd")
    assert stderr.getvalue() == "完了 out.vmd\n"


def test_update_before_stage_is_ignored():
    stream = io.StringIO()
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0)
    r.update(1, 2, note="ボーン: センター")
    r._draw()
    assert stream.getvalue() == ""


def test_update_does_not_write_heartbeat_is_sole_writer():
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    clock[0] = 5.0
    r.update(3, 10)
    assert stream.getvalue() == ""
    r.close()


def test_stages_share_one_line_without_newline():
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    r.update(3, 10, note="カメラ")
    r._draw()
    r.stage("キーフレーム圧縮")
    r.update(1, 5, note="ボーン: センター")
    r._draw()
    out = stream.getvalue()
    assert "\n" not in out
    assert out.count("\r") == 2
    assert out.rsplit("\r", 1)[1] == "[キーフレーム圧縮] 1/5 経過 0:00 ボーン: センター"
    r.close()


def test_update_note_appears_in_drawn_line_and_stage_resets_it():
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    r.update(60, 60, note="ボーン: センター")
    r._draw()
    assert "ボーン: センター" in stream.getvalue().rsplit("\r", 1)[1]
    r.stage("キーフレーム圧縮")
    r._draw()
    assert "ボーン: センター" not in stream.getvalue().rsplit("\r", 1)[1]
    r.close()


def test_heartbeat_advances_elapsed_without_update():
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.stage("キーフレーム圧縮")
    r.update(2, 10)
    clock[0] = 5.0
    _wait_until(lambda: "経過 0:05" in stream.getvalue())
    clock[0] = 60.0
    _wait_until(lambda: "経過 1:00" in stream.getvalue())
    r.close()
    out = stream.getvalue()
    assert "\r[キーフレーム圧縮] 2/10 経過 0:05" in out
    assert "[キーフレーム圧縮] 2/10 経過 1:00" in out


def test_elapsed_counts_from_stage_start_not_construction_and_resets_per_stage():
    stream = io.StringIO()
    clock = [50.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    clock[0] = 100.0
    r.stage("キーフレーム圧縮")
    clock[0] = 108.0
    r._draw()
    clock[0] = 200.0
    r.stage("キーフレーム圧縮")
    clock[0] = 205.0
    r._draw()
    segments = stream.getvalue().split("\r")[1:]
    assert segments[0].startswith("[キーフレーム圧縮] 経過 0:08")
    assert segments[1].startswith("[キーフレーム圧縮] 経過 0:05")
    r.close()


def test_close_clears_line_leaves_nothing():
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    r.update(2, 4)
    r._draw()
    drawn = stream.getvalue()
    width = progress._display_width("[キーフレーム圧縮] 2/4 経過 0:00")
    r.close()
    assert stream.getvalue() == drawn + "\r" + " " * width + "\r"
    assert "\n" not in stream.getvalue()
    assert r._thread is None


def test_close_stops_live_heartbeat_thread():
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.stage("キーフレーム圧縮")
    r.update(1, 2)
    clock[0] = 3.0
    _wait_until(lambda: "経過 0:03" in stream.getvalue())
    live = r._thread
    assert live is not None and live.is_alive()
    r.close()
    assert not live.is_alive()
    assert r._thread is None


def test_close_without_stage_is_noop_even_when_repeated():
    stream = io.StringIO()
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0)
    r.close()
    r.close()
    assert stream.getvalue() == ""
    assert r._thread is None


def test_stage_while_active_stops_old_heartbeat():
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=0.01)
    r.stage("キーフレーム圧縮")
    first = r._thread
    assert first is not None and first.is_alive()
    r.stage("キーフレーム圧縮")
    assert not first.is_alive()
    assert r._thread is not None and r._thread is not first and r._thread.is_alive()
    r.close()


def test_summary_writes_line_only_when_enabled():
    stream = io.StringIO()
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0)
    r.summary("完了 out.vmd")
    assert stream.getvalue() == "完了 out.vmd\n"
    stream2 = io.StringIO()
    r2 = ProgressReporter(stream2, enabled=False, now=lambda: 0.0)
    r2.summary("完了 out.vmd")
    assert stream2.getvalue() == ""


def test_shorter_line_is_padded_by_width_difference_to_erase_residue():
    count_part = " 1/1"
    stream = _TTYStream()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("処理その一")
    r.update(1, 1)
    r._draw()
    r.stage("処理その二")
    r._draw()
    assert stream.getvalue().rsplit("\r", 1)[1] == "[処理その二] 経過 0:00" + " " * len(count_part)
    r.close()


_WRITE_FAILURES = [OSError("io failed"), ValueError("bad stream"), UnicodeError("encode failed")]


@pytest.mark.parametrize("exc", _WRITE_FAILURES, ids=lambda e: type(e).__name__)
def test_draw_write_failure_disables_without_raising_and_close_stops_thread(exc):
    stream = _RaisingStream(exc)
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    live = r._thread
    assert live is not None and live.is_alive()
    r.update(1, 2)
    r._draw()
    assert r.enabled is False
    r.close()
    assert not live.is_alive()
    assert r._thread is None
    r.update(3, 4, note="ボーン: センター")
    r._draw()
    r.summary("完了 out.vmd")


@pytest.mark.parametrize("exc", _WRITE_FAILURES, ids=lambda e: type(e).__name__)
def test_summary_write_failure_disables_without_raising(exc):
    stream = _RaisingStream(exc)
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0)
    r.summary("完了 out.vmd")
    assert r.enabled is False


@pytest.mark.parametrize("exc", _WRITE_FAILURES, ids=lambda e: type(e).__name__)
def test_close_erase_failure_disables_without_raising_and_stops_thread(exc):
    stream = _TTYStream()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    r.update(2, 4)
    r._draw()
    live = r._thread
    r._stream = _RaisingStream(exc)
    r.close()
    assert r.enabled is False
    assert not live.is_alive()
    assert r._thread is None


def test_draw_holds_write_lock_during_write():
    lock = threading.Lock()
    stream = _WriteBlockingStream()
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0, write_lock=lock)
    r.stage("キーフレーム圧縮")

    t = threading.Thread(target=r._draw)
    t.start()
    try:
        assert stream.write_started.wait(2.0)
        assert lock.acquire(blocking=False) is False
    finally:
        stream.release.set()
        t.join(2.0)
        r._stop_heartbeat()


def test_close_holds_write_lock_during_write():
    lock = threading.Lock()
    stream = _WriteBlockingStream()
    r = ProgressReporter(_TTYStream(), enabled=True, now=lambda: 0.0, interval=3600.0, write_lock=lock)
    r.stage("キーフレーム圧縮")
    r._draw()
    r._stop_heartbeat()
    r._stream = stream

    t = threading.Thread(target=r.close)
    t.start()
    try:
        assert stream.write_started.wait(2.0)
        assert lock.acquire(blocking=False) is False
    finally:
        stream.release.set()
        t.join(2.0)


def test_summary_holds_write_lock_during_write():
    lock = threading.Lock()
    stream = _WriteBlockingStream()
    r = ProgressReporter(stream, enabled=True, now=lambda: 0.0, interval=3600.0, write_lock=lock)

    t = threading.Thread(target=r.summary, args=("完了 out.vmd",))
    t.start()
    try:
        assert stream.write_started.wait(2.0)
        assert lock.acquire(blocking=False) is False
    finally:
        stream.release.set()
        t.join(2.0)


def test_writes_without_write_lock_when_unspecified():
    stream = io.StringIO()
    clock = [0.0]
    r = ProgressReporter(stream, enabled=True, now=lambda: clock[0], interval=3600.0)
    r.stage("キーフレーム圧縮")
    r.update(1, 2)
    r._draw()
    r.summary("完了 out.vmd")
    r.close()
    assert "[キーフレーム圧縮] 1/2 経過 0:00" in stream.getvalue()
