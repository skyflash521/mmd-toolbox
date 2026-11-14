import pytest

from song2vpr.warning_text import warning_texts


def test_gpu_oversubscription_texts_carry_observed_values_and_remedy():
    message, human_text = warning_texts(
        "gpu_memory_oversubscribed",
        {"stage": "separate", "reserved_mib": 5600, "free_at_start_mib": 4000, "total_mib": 8192})
    assert message
    assert "5600MiB" in human_text and "4000MiB" in human_text
    assert "--device cpu" in human_text


_FIELDS = {"stage": "separate", "reserved_mib": 5600, "free_at_start_mib": 4000,
           "total_mib": 8192, "swap_increase_mib": 64, "process_rss_mib": 564,
           "ram_total_mib": 65457, "torch_version": "2.13.0"}


@pytest.mark.parametrize(("code", "fields", "expected"), [
    ("no_notes", {}, "ボーカル分離"),
    ("tempo_defaulted", {}, "--tempo"),
    ("forced_split", {}, "境界"),
])
def test_result_warning_human_text_contains_the_message_and_the_next_step(code, fields, expected):
    message, human_text = warning_texts(code, fields)
    assert message
    assert message in human_text
    assert expected in human_text


def test_kana_reading_warning_carries_the_unconverted_and_counted_char_counts():
    message, human_text = warning_texts("kana_reading_ineffective",
                                        {"unconverted_chars": 7, "counted_chars": 20})
    assert message
    assert "7" in human_text and "20" in human_text
