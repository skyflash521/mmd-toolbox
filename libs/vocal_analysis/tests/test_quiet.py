import threading

import pytest

from vocal_analysis import quiet


@pytest.fixture(autouse=True)
def _reset_silenced_flag():
    quiet._third_party_output_silenced = False
    yield
    quiet._third_party_output_silenced = False


def test_silence_third_party_output_disables_tqdm_by_default():
    tqdm_mod = pytest.importorskip("tqdm")

    quiet.silence_third_party_output()

    bar = tqdm_mod.tqdm(total=1)
    try:
        assert bar.disable is True
    finally:
        bar.close()


def test_silence_third_party_output_can_be_called_twice():
    pytest.importorskip("tqdm")

    quiet.silence_third_party_output()
    quiet.silence_third_party_output()


def test_silence_third_party_output_sets_huggingface_hub_verbosity_to_critical():
    hf_logging = pytest.importorskip("huggingface_hub.utils.logging")

    quiet.silence_third_party_output()

    assert hf_logging.get_verbosity() == hf_logging.CRITICAL


def test_silence_third_party_output_sets_transformers_verbosity_to_critical():
    transformers = pytest.importorskip("transformers")

    quiet.silence_third_party_output()

    assert transformers.utils.logging.get_verbosity() == transformers.logging.CRITICAL


def test_suppress_native_stderr_hides_direct_fd_writes(capfd):
    import os

    os.write(2, b"before\n")
    with quiet.suppress_native_stderr():
        os.write(2, b"hidden\n")
    os.write(2, b"after\n")

    captured = capfd.readouterr()
    assert "before" in captured.err
    assert "hidden" not in captured.err
    assert "after" in captured.err


def test_suppress_native_stderr_restores_fd_after_exception(capfd):
    import os

    with pytest.raises(ValueError):
        with quiet.suppress_native_stderr():
            raise ValueError("boom")
    os.write(2, b"after-exception\n")

    captured = capfd.readouterr()
    assert "after-exception" in captured.err


def test_suppress_native_stderr_restores_fd_even_if_flush_fails(monkeypatch, capfd):
    import os

    class _BrokenStderr:
        def flush(self):
            raise OSError("flush failed")

    monkeypatch.setattr(quiet.sys, "stderr", _BrokenStderr())

    with quiet.suppress_native_stderr():
        pass
    os.write(2, b"after-flush-failure\n")

    captured = capfd.readouterr()
    assert "after-flush-failure" in captured.err


def _try_acquire_stderr_write_lock_from_other_thread():
    result = {}

    def _try_acquire():
        result["acquired"] = quiet.STDERR_WRITE_LOCK.acquire(blocking=False)
        if result["acquired"]:
            quiet.STDERR_WRITE_LOCK.release()

    # RLock は保有スレッド自身なら常に再取得できる。
    t = threading.Thread(target=_try_acquire)
    t.start()
    t.join(2.0)
    return result.get("acquired", False)


def test_suppress_native_stderr_holds_stderr_write_lock_for_other_threads():
    with quiet.suppress_native_stderr():
        acquired = _try_acquire_stderr_write_lock_from_other_thread()

    assert acquired is False


def test_suppress_native_stderr_releases_stderr_write_lock_after_exit():
    with quiet.suppress_native_stderr():
        pass

    assert _try_acquire_stderr_write_lock_from_other_thread() is True
