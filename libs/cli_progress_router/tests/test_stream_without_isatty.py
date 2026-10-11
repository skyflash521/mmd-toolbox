import pytest

from cli_progress_router import ProgressRouter

_LABELS = {"load": "音声読み込み", "write": "書き出し"}


class _NoIsattyStream:
    def __init__(self):
        self.writes = []

    def write(self, text):
        self.writes.append(text)
        return len(text)

    def flush(self):
        pass


class _NoneIsattyStream(_NoIsattyStream):
    isatty = None


class _RaisingStream:
    def write(self, _text):
        raise AssertionError("TTY でないストリームへ書いた")

    def flush(self):
        pass


def _drive(router):
    router.stage("load", done=0, total=None)
    router.stage("load", done=1, total=2, note="n", elapsed=0.5)
    router.stage("write")
    router.close()
    router.summary("完了 out.vmd")
    router.close()


@pytest.mark.parametrize("stream_cls", [_NoIsattyStream, _NoneIsattyStream])
@pytest.mark.parametrize("quiet", [False, True])
def test_stream_without_a_usable_isatty_builds_a_silent_router(stream_cls, quiet):
    stream = stream_cls()
    router = ProgressRouter(machine=False, quiet=quiet, emitter=None, stream=stream, labels=_LABELS)
    _drive(router)
    assert stream.writes == []


def test_stream_without_isatty_never_receives_live_output_even_with_a_fast_interval():
    router = ProgressRouter(machine=False, quiet=False, emitter=None, stream=_RaisingStream(),
                            labels=_LABELS, interval=0.001)
    _drive(router)


def test_machine_mode_does_not_need_isatty_on_the_stream():
    events = []

    class _Emitter:
        def progress(self, **fields):
            events.append(fields)

    stream = _NoIsattyStream()
    router = ProgressRouter(machine=True, quiet=False, emitter=_Emitter(), stream=stream, labels=_LABELS)
    router.stage("load", done=1, total=2)
    router.close()
    assert events == [{"stage": "load", "done": 1, "total": 2, "note": "", "elapsed": 0.0}]
    assert stream.writes == []
