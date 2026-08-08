"""外部ツールの進捗表示・ログ抑制(quiet.silence_third_party_output)のテスト。

対象ライブラリ(tqdm・huggingface_hub・transformers)が実際に導入されている環境でのみ検証する
(vocal-analysis extra が無い最小環境では該当箇所を skip する)。
"""

import threading

import pytest

from vocal_analysis import quiet


@pytest.fixture(autouse=True)
def _reset_done_flag():
    # 各テストで実際に抑制処理が走るよう、モジュールの一度きりガードを毎回リセットする。
    quiet._done = False
    yield
    quiet._done = False


def test_silence_third_party_output_disables_tqdm_by_default():
    tqdm_mod = pytest.importorskip("tqdm")

    quiet.silence_third_party_output()

    bar = tqdm_mod.tqdm(total=1)
    try:
        assert bar.disable is True
    finally:
        bar.close()


def test_silence_third_party_output_is_idempotent():
    pytest.importorskip("tqdm")

    quiet.silence_third_party_output()
    quiet.silence_third_party_output()  # 2回目も例外なく完了する


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

    # sys.stderr を丸ごと差し替える(quiet.sys は sys そのもの)。capfd は fd 2 の複製・
    # リダイレクトで捕捉するため sys.stderr オブジェクトの差し替えでは壊れず、monkeypatch が
    # テスト終了時に元へ戻すので他テストへは波及しない。
    monkeypatch.setattr(quiet.sys, "stderr", _BrokenStderr())

    with quiet.suppress_native_stderr():
        pass
    os.write(2, b"after-flush-failure\n")

    captured = capfd.readouterr()
    assert "after-flush-failure" in captured.err


def _try_acquire_stderr_write_lock_from_other_thread():
    """別スレッドから STDERR_WRITE_LOCK の非ブロッキング取得を試み、成否を返す。

    取得できれば解放してから返す(呼び出し元スレッドに残さない)。RLock は保有スレッド自身の
    再取得は常に成功するため、この関数は必ずテスト本体と別のスレッドで呼ぶこと。
    """
    result = {}

    def _try_acquire():
        result["acquired"] = quiet.STDERR_WRITE_LOCK.acquire(blocking=False)
        if result["acquired"]:
            quiet.STDERR_WRITE_LOCK.release()

    t = threading.Thread(target=_try_acquire)
    t.start()
    t.join(2.0)
    return result.get("acquired", False)


def test_suppress_native_stderr_holds_stderr_write_lock_for_other_threads():
    # fd 差し替え中、別スレッドは STDERR_WRITE_LOCK を取れない(取れると呼び出し側 CLI の
    # 進捗表示等がこの区間に同時書き込みしてしまい、Windowsの実コンソールハンドルで書き込み
    # 失敗が起こりうる)。この排他が実際に効いていることを固定する。
    with quiet.suppress_native_stderr():
        acquired = _try_acquire_stderr_write_lock_from_other_thread()

    assert acquired is False


def test_suppress_native_stderr_releases_stderr_write_lock_after_exit():
    # ロックが恒久的に保持されたままにならないことも確認する(抜けた後は別スレッドが取得できる)。
    # RLock は保有スレッド自身の再取得が常に成功するため、テスト本体と別のスレッドから確認する。
    with quiet.suppress_native_stderr():
        pass

    assert _try_acquire_stderr_write_lock_from_other_thread() is True
