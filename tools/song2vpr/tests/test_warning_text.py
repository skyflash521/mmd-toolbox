"""song2vpr が出す警告の文言のテスト。

資源逼迫・GPU 構成について何を観測しどの事実で警告が成立するかの検証は共有の cli_resource_watch が
担う。ここでは song2vpr が組み立てる本文——観測値を人間向け1行へ載せ、勧める対処を song2vpr の
オプション名で書くこと——だけを検証する。
"""

import pytest

from song2vpr.warning_text import warning_texts


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


# --- 処理の結果から判定する警告 ------------------------------------------------


@pytest.mark.parametrize(("code", "fields", "expected"), [
    ("no_notes", {}, "ボーカル分離"),
    ("tempo_defaulted", {}, "--tempo"),
    ("forced_split", {}, "境界"),
])
def test_result_warning_texts_tell_what_to_do(code, fields, expected):
    """人間向け1行は、何が起きたかと、利用者が次に取れる手を書く。"""
    message, human_text = warning_texts(code, fields)
    assert message
    assert message in human_text
    assert expected in human_text


def test_kana_reading_warning_carries_the_counts_behind_the_judgement():
    """割合の判定に使った2つの件数を載せる(受け手が判定を再現できるようにするため)。"""
    message, human_text = warning_texts("kana_reading_ineffective",
                                        {"unconverted_chars": 7, "counted_chars": 20})
    assert message
    assert "7" in human_text and "20" in human_text
