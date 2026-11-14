import pytest

from cli_resource_watch import warning_texts


def test_gpu_memory_oversubscribed_texts_are_fixed():
    message, human_text = warning_texts(
        "gpu_memory_oversubscribed",
        {"stage": "separate", "reserved_mib": 5600, "free_at_start_mib": 4000, "total_mib": 8192})

    assert message == "GPUメモリの要求量が空き容量を超過しました"
    assert human_text == (
        "GPUメモリの要求量が空き容量を超過しました(要求 5600MiB / 開始時空き 4000MiB)。"
        "超過分は主記憶へ退避され処理が大幅に低速化します。--device cpu での実行を検討してください")


def test_swap_detected_texts_are_fixed():
    message, human_text = warning_texts(
        "swap_detected",
        {"stage": "separate", "swap_increase_mib": 64, "process_rss_mib": 564,
         "ram_total_mib": 65457})

    assert message == "メモリ不足によるスワップを検出しました"
    assert human_text == (
        "メモリ不足によるスワップを検出しました(スワップ増加 64MiB・本プロセス使用 564MiB)。"
        "処理が大幅に低速化します。他のアプリケーションを終了してメモリを確保してください")


def test_cpu_only_torch_texts_are_fixed():
    message, human_text = warning_texts("cpu_only_torch", {"torch_version": "2.13.0"})

    assert message == "導入されている torch では GPU を扱えません"
    assert human_text == (
        "導入されている torch では GPU を扱えません(torch 2.13.0 は CPU 専用版)。CPU で処理します")


def test_cuda_unavailable_texts_are_fixed():
    message, human_text = warning_texts("cuda_unavailable", {"torch_version": "2.13.0+cu126"})

    assert message == "この GPU で使えない CUDA 版の torch が入っています"
    assert human_text == (
        "この GPU で使えない CUDA 版の torch が入っています(torch 2.13.0+cu126)。"
        "別の CUDA のバージョンで入れ直してください")


def test_message_does_not_embed_observed_values():
    message, _ = warning_texts("cpu_only_torch", {"torch_version": "9.9.9"})

    assert "9.9.9" not in message


@pytest.mark.parametrize("code", [
    "gpu_memory_oversubscribed", "swap_detected", "cpu_only_torch", "cuda_unavailable"])
def test_missing_observed_field_raises_key_error(code):
    with pytest.raises(KeyError):
        warning_texts(code, {})


def test_unknown_code_raises_key_error():
    with pytest.raises(KeyError):
        warning_texts("no_such_warning", {})
