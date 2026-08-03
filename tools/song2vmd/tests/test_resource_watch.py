"""song2vmd の資源逼迫・GPU 構成の警告の文言のテスト。

何を観測しどの事実で警告が成立するかの検証は共有の cli_resource_watch が担う。ここでは
song2vmd が組み立てる本文——公開する4コードすべてで、観測値を人間向け1行へ載せ、勧める対処を
song2vmd のオプション名で書くこと——だけを検証する。
"""

from song2vmd.resource_watch import warning_texts


def test_gpu_oversubscription_texts_carry_observed_values_and_remedy():
    message, human_text = warning_texts(
        "gpu_memory_oversubscribed",
        {"stage": "separate", "reserved_mib": 5600, "free_at_start_mib": 4000, "total_mib": 8192})
    assert message == "GPUメモリの要求量が空き容量を超過しました"
    assert "5600MiB" in human_text and "4000MiB" in human_text
    assert "--device cpu" in human_text  # 勧める対処は song2vmd のオプション名で書く


def test_swap_detection_texts_carry_observed_values_and_remedy():
    message, human_text = warning_texts(
        "swap_detected",
        {"stage": "separate", "swap_increase_mib": 64, "process_rss_mib": 564,
         "ram_total_mib": 65457})
    assert message == "メモリ不足によるスワップを検出しました"
    assert "64MiB" in human_text and "564MiB" in human_text
    assert "他のアプリケーション" in human_text


def test_cpu_only_torch_texts():
    message, human_text = warning_texts("cpu_only_torch", {"torch_version": "2.13.0"})
    assert message == "導入されている torch では GPU を扱えません"
    assert human_text == (
        "導入されている torch では GPU を扱えません(torch 2.13.0 は CPU 専用版)。CPU で処理します")


def test_cuda_unavailable_texts():
    message, human_text = warning_texts("cuda_unavailable", {"torch_version": "2.13.0+cu126"})
    assert message == "この GPU で使えない CUDA 版の torch が入っています"
    assert human_text == (
        "この GPU で使えない CUDA 版の torch が入っています(torch 2.13.0+cu126)。"
        "別の CUDA のバージョンで入れ直してください")
