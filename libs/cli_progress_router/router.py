from cli_progress import progress as _display_module


class ProgressEmitError(Exception):
    pass


class ProgressRouter:
    def __init__(self, *, machine, quiet, emitter, stream, labels,
                 write_lock=None, now=None, interval=None):
        self._machine = machine
        self._emitter = emitter if machine else None
        self._labels = labels
        self._current_stage = None
        if machine:
            self._display = None
        else:
            interval_option = {} if interval is None else {"interval": interval}
            self._display = _display_module.ProgressReporter(
                stream=stream, enabled=(not quiet) and stream.isatty(),
                now=now, write_lock=write_lock, **interval_option)

    def stage(self, stage_id, *, done=0, total=None, note="", elapsed=0.0):
        if self._machine:
            try:
                self._emitter.progress(
                    stage=stage_id, done=done, total=total, note=note, elapsed=elapsed)
            except Exception as e:
                raise ProgressEmitError(f"{type(e).__name__}: {e}") from e
            return
        if stage_id != self._current_stage:
            self._current_stage = stage_id
            self._display.stage(self._labels.get(stage_id, stage_id))
        self._display.update(done, total, note)

    def close(self):
        if not self._machine:
            self._display.close()
        self._current_stage = None

    def summary(self, message):
        if not self._machine:
            self._display.summary(message)
