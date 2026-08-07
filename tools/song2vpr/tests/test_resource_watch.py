"""song2vpr の資源逼迫・GPU 構成の警告の文言のテスト。

何を観測しどの事実で警告が成立するかの検証は共有の cli_resource_watch が担う。ここでは
song2vpr が組み立てる本文——公開する4コードすべてで、観測値を人間向け1行へ載せ、勧める対処を
song2vpr のオプション名で書くこと——だけを検証する。
"""

import pytest

from song2vpr.resource_watch import warning_texts


def test_gpu_oversubscription_texts_carry_observed_values_and_remedy():
    message, human_text = warning_texts(
        "gpu_memory_oversubscribed",
        {"stage": "separate", "reserved_mib": 5600, "free_at_start_mib": 4000, "total_mib": 8192})
    assert message
    assert "5600MiB" in human_text and "4000MiB" in human_text
    assert "--device cpu" in human_text  # 勧める対処は song2vpr のオプション名で書く


def test_swap_detection_texts_carry_observed_values_and_remedy():
    message, human_text = warning_texts(
        "swap_detected",
        {"stage": "separate", "swap_increase_mib": 64, "process_rss_mib": 564,
         "ram_total_mib": 65457})
    assert message
    assert "64MiB" in human_text and "564MiB" in human_text
    assert "他のアプリケーション" in human_text  # 勧める対処


def test_cpu_only_torch_texts_carry_version_and_remedy():
    message, human_text = warning_texts("cpu_only_torch", {"torch_version": "2.13.0"})
    assert message
    assert "2.13.0" in human_text
    assert "CPU" in human_text  # CPU で処理する旨


def test_cuda_unavailable_texts_carry_version_and_remedy():
    message, human_text = warning_texts("cuda_unavailable", {"torch_version": "2.13.0+cu126"})
    assert message
    assert "2.13.0+cu126" in human_text
    assert "CUDA" in human_text  # 別の CUDA のバージョンでの入れ直し


_FIELDS = {"stage": "separate", "reserved_mib": 5600, "free_at_start_mib": 4000,
           "total_mib": 8192, "swap_increase_mib": 64, "process_rss_mib": 564,
           "ram_total_mib": 65457, "torch_version": "2.13.0"}


@pytest.mark.parametrize("code,remedy", [
    ("gpu_memory_oversubscribed", "--device cpu"),
    ("swap_detected", "他のアプリケーション"),
])
def test_machine_message_omits_the_remedy(code, remedy):
    """機械モードの本文は成立した事実だけを述べ、対処は人間向け1行が持つ。

    照合するのは対処にしか現れない語に限る(事実の主題と重なる語で照合すると、事実を素直に述べた
    本文まで弾いてしまう)。
    """
    message, _ = warning_texts(code, _FIELDS)
    assert remedy not in message


@pytest.mark.parametrize("code", [
    "gpu_memory_oversubscribed", "swap_detected", "cpu_only_torch", "cuda_unavailable"])
def test_machine_message_omits_the_observed_values(code):
    """観測値は本文に重ねない(呼び出し側が同じ値を warning イベントのフィールドへ載せるため)。"""
    message, _ = warning_texts(code, _FIELDS)
    assert "MiB" not in message and "2.13.0" not in message
