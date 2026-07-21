"""外部ツールの進捗表示・ログ抑制(quiet.silence_third_party_output)のテスト。

対象ライブラリ(tqdm・huggingface_hub・transformers)が実際に導入されている環境でのみ検証する
(vocal-analysis extra が無い最小環境では該当箇所を skip する)。
"""

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
