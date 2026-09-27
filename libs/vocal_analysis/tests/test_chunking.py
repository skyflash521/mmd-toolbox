import pytest

from vocal_analysis import Segment, chunking


def seg(type_, start, end, phoneme=None, confidence=None):
    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=phoneme, confidence=confidence)


def test_audio_not_longer_than_max_duration_has_no_boundaries():
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=200.0, rms_times_sec=[0.0, 100.0, 199.0], rms_values=[0.5, 0.5, 0.5],
        max_duration_sec=300.0,
    )
    assert boundaries == []


def test_boundary_picks_quietest_point_among_multiple_silence_candidates():
    times = [290.0, 295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 303.0, 305.0, 310.0]
    values = [0.8, 0.7, 0.05, 0.5, 0.02, 0.5, 0.6, 0.04, 0.8, 0.9]
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=400.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(298.0, False)]


def test_ties_pick_the_earliest_of_the_quietest_points():
    times = [296.0, 298.0, 300.0, 302.0, 304.0]
    values = [0.5, 0.02, 0.5, 0.02, 0.5]
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=400.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(298.0, False)]


def test_forced_split_at_target_when_no_silence_in_window():
    times = [295.0, 297.0, 299.0, 300.0, 301.0, 303.0, 305.0]
    values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=400.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(300.0, True)]


def test_second_target_is_computed_from_actual_cut_point_not_fixed_grid():
    times = [float(t) for t in range(0, 651)]
    values = [0.5] * len(times)
    first_cut = 298
    silence_only_reachable_from_actual_cut = 594
    values[first_cut] = 0.01
    values[silence_only_reachable_from_actual_cut] = 0.01
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=650.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(298.0, False), (594.0, False)]


@pytest.mark.parametrize("quietest_value,expected_forced", [(0.06, False), (0.0601, True)])
def test_default_silence_threshold_is_0_06_inclusive(quietest_value, expected_forced):
    times = [298.0, 300.0, 302.0]
    values = [0.5, quietest_value, 0.5]
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=400.0, rms_times_sec=times, rms_values=values, max_duration_sec=300.0,
    )
    assert boundaries == [(300.0, expected_forced)]


def test_forced_split_target_within_duration_is_still_applied():
    times = [float(t) for t in range(0, 320)]
    values = [0.5] * len(times)
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=320.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(300.0, True)]


def test_boundaries_keep_advancing_when_max_duration_is_small_relative_to_search_window():
    times = [round(t * 0.1, 1) for t in range(0, 200)]
    values = [0.01] * len(times)
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=10.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=1.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries
    times_only = [t for t, _ in boundaries]
    assert times_only == sorted(times_only)
    assert len(times_only) == len(set(times_only))
    assert all(forced is False for _, forced in boundaries)


def test_forced_split_flag_distinguishes_silence_and_forced_boundaries_in_same_call():
    times = [float(t) for t in range(0, 620)]
    values = [0.5] * len(times)
    values[298] = 0.01
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=620.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(298.0, False), (598.0, True)]


def test_single_chunk_no_boundaries_returns_offset_segments():
    chunk_segments = [[seg("vowel", 0.0, 1.0, phoneme="a"), seg("gap", 1.0, 2.0)]]
    merged = chunking.merge_chunk_segments(chunk_segments, chunk_offsets_sec=[0.0], boundaries_sec=[])
    assert [(s.type, s.start_sec, s.end_sec) for s in merged] == [
        ("vowel", 0.0, 1.0), ("gap", 1.0, 2.0),
    ]


def test_overlap_is_trimmed_at_boundary_and_consonants_across_it_are_not_merged():
    chunk0 = [seg("vowel", 0.0, 9.0, phoneme="a"), seg("consonant", 9.0, 11.0, phoneme="k")]
    chunk1 = [seg("consonant", 0.0, 2.0, phoneme="k"), seg("vowel", 2.0, 11.0, phoneme="i")]
    merged = chunking.merge_chunk_segments(
        [chunk0, chunk1], chunk_offsets_sec=[0.0, 9.0], boundaries_sec=[10.0],
    )
    assert merged[0] == seg("vowel", 0.0, 9.0, phoneme="a")
    assert merged[1].type == "consonant" and merged[1].phoneme == "k"
    assert merged[1].start_sec == 9.0 and merged[1].end_sec == 10.0
    assert merged[2].type == "consonant" and merged[2].phoneme == "k"
    assert merged[2].start_sec == 10.0 and merged[2].end_sec == 11.0
    assert merged[3] == seg("vowel", 11.0, 20.0, phoneme="i")


def test_matching_vowel_across_boundary_merges_into_one_event_without_confidence():
    chunk0 = [seg("vowel", 0.0, 10.5, phoneme="a", confidence=0.9)]
    chunk1 = [seg("vowel", 0.0, 5.0, phoneme="a", confidence=0.7)]
    merged = chunking.merge_chunk_segments(
        [chunk0, chunk1], chunk_offsets_sec=[0.0, 9.0], boundaries_sec=[10.0],
    )
    assert len(merged) == 1
    assert merged[0].type == "vowel" and merged[0].phoneme == "a"
    assert merged[0].start_sec == 0.0 and merged[0].end_sec == 14.0
    assert merged[0].confidence is None


def test_differing_vowels_across_boundary_do_not_merge():
    chunk0 = [seg("vowel", 0.0, 10.5, phoneme="a")]
    chunk1 = [seg("vowel", 0.0, 5.0, phoneme="i")]
    merged = chunking.merge_chunk_segments(
        [chunk0, chunk1], chunk_offsets_sec=[0.0, 9.0], boundaries_sec=[10.0],
    )
    assert len(merged) == 2
    assert merged[0].phoneme == "a" and merged[0].end_sec == 10.0
    assert merged[1].phoneme == "i" and merged[1].start_sec == 10.0


def test_merged_segments_cover_full_duration_without_gaps():
    chunk0 = [seg("vowel", 0.0, 6.0, phoneme="a"), seg("gap", 6.0, 11.0)]
    chunk1 = [seg("gap", 0.0, 2.0), seg("vowel", 2.0, 11.0, phoneme="i")]
    merged = chunking.merge_chunk_segments(
        [chunk0, chunk1], chunk_offsets_sec=[0.0, 9.0], boundaries_sec=[10.0],
    )
    assert merged[0].start_sec == 0.0
    assert merged[-1].end_sec == 20.0
    for prev, nxt in zip(merged, merged[1:], strict=False):
        assert prev.end_sec == nxt.start_sec
