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


def _drive(router):
    router.stage("load", done=0, total=None)
    router.stage("load", done=1, total=2, note="n", elapsed=0.5)
    router.stage("write")
    router.close()
    router.summary("完了 out.vmd")
    router.close()


def test_stream_without_isatty_builds_a_silent_router():
    stream = _NoIsattyStream()
    router = ProgressRouter(machine=False, quiet=False, emitter=None, stream=stream, labels=_LABELS)
    _drive(router)
    assert stream.writes == []


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
