import contextlib
import os
import sys
import threading
from functools import partialmethod

_WRITE_ERRORS = (OSError, ValueError, UnicodeError)

_third_party_output_silenced = False

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


def silence_third_party_output():
    global _third_party_output_silenced
    if _third_party_output_silenced:
        return
    _third_party_output_silenced = True

    try:
        from tqdm import tqdm

        tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)
    except ImportError:
        pass

    try:
        from huggingface_hub.utils import disable_progress_bars
        from huggingface_hub.utils import logging as hf_logging

        disable_progress_bars()
        hf_logging.set_verbosity(hf_logging.CRITICAL)
    except ImportError:
        pass

    try:
        import transformers

        transformers.utils.logging.set_verbosity(transformers.logging.CRITICAL)
    except ImportError:
        pass
