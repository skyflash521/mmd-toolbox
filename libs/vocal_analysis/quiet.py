import contextlib
import importlib
import os
import sys
import threading
from functools import partialmethod

_WRITE_ERRORS = (OSError, ValueError, UnicodeError)

_STATE_LOCK = threading.Lock()
_depth = 0
_restorers: list = []

# Windows の実コンソールでは、fd 2 へ書き込む最中に os.dup2 で差し替えると書き込み側で OSError が起きうる。
# fd 2 へ書く別スレッドは、このロックを取ってから書くこと。
STDERR_WRITE_LOCK = threading.RLock()


@contextlib.contextmanager
def suppress_native_stderr():
    with STDERR_WRITE_LOCK:
        try:
            sys.stderr.flush()
        except _WRITE_ERRORS:
            pass
        saved_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        try:
            os.dup2(devnull_fd, 2)
            yield
        finally:
            os.dup2(saved_fd, 2)
            os.close(devnull_fd)
            os.close(saved_fd)
            try:
                sys.stderr.flush()
            except _WRITE_ERRORS:
                pass


def _silence_tqdm():
    import tqdm.std

    saved_init = tqdm.std.tqdm.__dict__["__init__"]
    tqdm.std.tqdm.__init__ = partialmethod(saved_init, disable=True)

    def restore():
        tqdm.std.tqdm.__init__ = saved_init

    return restore


def _silence_huggingface_hub():
    hf_tqdm = importlib.import_module("huggingface_hub.utils.tqdm")
    from huggingface_hub.utils import (
        are_progress_bars_disabled,
        disable_progress_bars,
        enable_progress_bars,
    )
    from huggingface_hub.utils import logging as hf_logging

    progress_bars_were_disabled = are_progress_bars_disabled()
    saved_states = getattr(hf_tqdm, "progress_bar_states", None)
    saved_states = None if saved_states is None else dict(saved_states)
    saved_verbosity = hf_logging.get_verbosity()
    disable_progress_bars()
    hf_logging.set_verbosity(hf_logging.CRITICAL)

    def restore():
        hf_logging.set_verbosity(saved_verbosity)
        if saved_states is not None:
            hf_tqdm.progress_bar_states.clear()
            hf_tqdm.progress_bar_states.update(saved_states)
        elif not progress_bars_were_disabled:
            enable_progress_bars()

    return restore


def _silence_transformers():
    import transformers

    saved_verbosity = transformers.utils.logging.get_verbosity()
    transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)

    def restore():
        transformers.utils.logging.set_verbosity(saved_verbosity)

    return restore


@contextlib.contextmanager
def silenced_third_party_output():
    """入れ子のときは最外の出口で、入る前の状態へ戻す。tqdm・huggingface_hub・transformers が
    導入されていなければ、その部分は何もしない。"""
    global _depth, _restorers
    with _STATE_LOCK:
        if _depth == 0:
            restorers = []
            try:
                for silence in (_silence_tqdm, _silence_huggingface_hub, _silence_transformers):
                    try:
                        restorers.append(silence())
                    except ImportError:
                        pass
            except BaseException:
                for restore in reversed(restorers):
                    restore()
                raise
            _restorers = restorers
        _depth += 1
    try:
        yield
    finally:
        with _STATE_LOCK:
            _depth -= 1
            if _depth == 0:
                restorers, _restorers = _restorers, []
                for restore in reversed(restorers):
                    restore()
