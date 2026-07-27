"""song2vmd 長尺分割の境界検出・セグメント結合のテスト。

処理資源対策のための自動分割(`--max-duration`)を検証する。範囲は次の2点に限る:
1. 目標境界±5秒(初期値)の探索窓で無音点を探し、無音が無ければ強制分割する境界検出
   (chunking.find_chunk_boundaries)。
2. チャンクごとのセグメント列(チャンクローカル時刻)を、オーバーラップ区間はチャンク境界で切り詰め、
   境界をまたぐ連続母音区間(同一vowel)を1つへ結合して全曲セグメント列へ統合する処理
   (chunking.merge_chunk_segments)。

音声ファイルの実際の分割・vocal_analysis(S0〜S2)呼び出し・RMSの曲全体基準での算出は
実音声を要する統合の関心事であり、本モジュールの対象外(強弱の相対正規化を曲全体基準で行う要件は
RMS算出を1回だけ行うことで満たされ、本モジュールが担うセグメント結合とは別の関心事)。
"""

from song2vmd import chunking
from vocal_analysis import Segment


def seg(type_, start, end, phoneme=None, confidence=None):
    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=phoneme, confidence=confidence)


# --- find_chunk_boundaries ---------------------------------------------------


def test_short_audio_has_no_boundaries():
    # 曲長がmax_duration以下なら分割しない。
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=200.0, rms_times_sec=[0.0, 100.0, 199.0], rms_values=[0.5, 0.5, 0.5],
        max_duration_sec=300.0,
    )
    assert boundaries == []


def test_boundary_picks_quietest_point_among_multiple_silence_candidates():
    # 探索窓内にしきい値以下の候補が複数(296s=0.05・298s=0.02・303s=0.04)あるとき、
    # 単に最初に見つかった無音点ではなく、最も静かな点(298s)を選ぶ。無音採用点なのでforced=False。
    times = [290.0, 295.0, 296.0, 297.0, 298.0, 299.0, 300.0, 303.0, 305.0, 310.0]
    values = [0.8, 0.7, 0.05, 0.5, 0.02, 0.5, 0.6, 0.04, 0.8, 0.9]
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=400.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(298.0, False)]


def test_forced_split_when_no_silence_in_window():
    # 探索窓内に無音(しきい値以下)が無ければ、目標境界そのもので強制分割する
    # (ロングトーン・ライブ音源等)。強制分割点なのでforced=True。
    times = [295.0, 297.0, 299.0, 300.0, 301.0, 303.0, 305.0]
    values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]  # 終始しきい値(0.06)を上回る
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=400.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(300.0, True)]


def test_second_target_is_computed_from_actual_cut_point_not_fixed_grid():
    # 次の目標境界は固定グリッド(2×max_duration=600s)ではなく、実際に切った点(298s)から
    # max_duration秒後(598s)を起点にする。無音点を594sに置くと、
    # 起点598sの探索窓[593,603]には入るが、固定グリッド起点600sの探索窓[595,605]には
    # 入らない(594<595)ため、両設計を区別できる。両方とも無音採用点なのでforced=False。
    times = [float(t) for t in range(0, 651)]
    values = [0.5] * len(times)
    values[298] = 0.01  # 最初の目標境界(300s)付近の無音点
    values[594] = 0.01  # 実境界基準の次目標(598s)の探索窓には入るが、固定グリッド基準(600s)には入らない
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=650.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(298.0, False), (594.0, False)]


def test_forced_split_target_within_duration_is_still_applied():
    times = [float(t) for t in range(0, 320)]
    values = [0.5] * len(times)
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=320.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(300.0, True)]


def test_boundaries_keep_advancing_when_max_duration_is_small_relative_to_search_window():
    # max_duration_secがsearch_window_sec以下(--max-durationに小さい値を与えた場合)でも、
    # 終始無音(=常に候補になりうる)なデータで境界検出が前進し続け、有限回で終わることを確認する。
    times = [round(t * 0.1, 1) for t in range(0, 200)]  # 0.0, 0.1, ..., 19.9
    values = [0.01] * len(times)  # 終始無音しきい値以下
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=10.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=1.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries
    times_only = [t for t, _ in boundaries]
    assert times_only == sorted(times_only)
    assert len(times_only) == len(set(times_only))  # 重複無く単調に前進している
    assert all(forced is False for _, forced in boundaries)  # 終始無音なので全て無音採用


def test_forced_split_flag_distinguishes_silence_and_forced_boundaries_in_same_call():
    # 1回の呼び出し内で、1つ目の境界は無音採用(forced=False)・2つ目は強制分割(forced=True)になる
    # 混在ケース(単一の代表値で一括判定せず境界ごとに判定することを確認する)。
    times = [float(t) for t in range(0, 620)]
    values = [0.5] * len(times)
    values[298] = 0.01  # 最初の目標境界(300s)付近だけ無音
    # 2つ目の目標(298+300=598s)付近は終始しきい値超のまま(強制分割になる)
    boundaries = chunking.find_chunk_boundaries(
        duration_sec=620.0, rms_times_sec=times, rms_values=values,
        max_duration_sec=300.0, search_window_sec=5.0, silence_threshold=0.06,
    )
    assert boundaries == [(298.0, False), (598.0, True)]


# --- merge_chunk_segments -----------------------------------------------------


def test_single_chunk_no_boundaries_returns_offset_segments():
    chunk_segments = [[seg("vowel", 0.0, 1.0, phoneme="a"), seg("gap", 1.0, 2.0)]]
    merged = chunking.merge_chunk_segments(chunk_segments, chunk_offsets_sec=[0.0], boundaries_sec=[])
    assert [(s.type, s.start_sec, s.end_sec) for s in merged] == [
        ("vowel", 0.0, 1.0), ("gap", 1.0, 2.0),
    ]


def test_overlap_is_trimmed_at_boundary():
    # チャンク0はローカル時刻でオフセット0(グローバル時刻もそのまま)、チャンク1はオフセット9
    # (チャンク1のローカル0.0はグローバル9.0に相当)。境界は10.0。オーバーラップ区間
    # (グローバル[9,11])は、チャンク0側が[9,10)・チャンク1側が[10,11)を採用する形で
    # 境界で切り詰められる。子音は母音ではないため、境界をまたいでも結合しない
    # (結合対象は「連続母音区間」のみ)。
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


def test_matching_vowel_across_boundary_merges_into_one_event():
    chunk0 = [seg("vowel", 0.0, 10.5, phoneme="a")]
    chunk1 = [seg("vowel", 0.0, 5.0, phoneme="a")]  # オフセット9 → グローバル[9,14]
    merged = chunking.merge_chunk_segments(
        [chunk0, chunk1], chunk_offsets_sec=[0.0, 9.0], boundaries_sec=[10.0],
    )
    assert len(merged) == 1
    assert merged[0].type == "vowel" and merged[0].phoneme == "a"
    assert merged[0].start_sec == 0.0 and merged[0].end_sec == 14.0


def test_differing_segments_across_boundary_do_not_merge():
    chunk0 = [seg("vowel", 0.0, 10.5, phoneme="a")]
    chunk1 = [seg("vowel", 0.0, 5.0, phoneme="i")]  # オフセット9 → グローバル[9,14]
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
