"""song2vmd レポート/診断のテスト。

口形イベント列・音素セグメント列・EventDiagnostics から診断データ(Diagnostics)を組み立てる
report.build_diagnostics と、それを人間向けテキスト(--dry-run)・機械モード result ペイロード
(--machine の mode:"run"/"inspect")へ整形する各関数を検証する。実際の音声処理・外部呼び出しは
対象外(合成データのみを扱う純粋ロジック)。
"""

import pytest

from lipsync import MouthEvent, MouthShape

from song2vmd import events, report
from vocal_analysis import Segment


def seg(type_, start, end):
    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=None, confidence=None)


def mev(shape, start, end, open_amount=0.0):
    return MouthEvent(shape=shape, start=start, end=end, open_amount=open_amount)


def diag_of(**overrides):
    kw = dict(
        segments=[], mouth_events=[],
        event_diagnostics=events.EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=0),
        backends={}, style="pop", separated=False, duration_sec=1.0, keys=0,
    )
    kw.update(overrides)
    if "mora_event_group_sizes" not in overrides:
        # 明示指定が無ければ、各MouthEventを1モーラ(分割なし)として自動導出する。
        mora_count = sum(1 for e in kw["mouth_events"] if e.shape in report._MORA_SHAPES)
        kw["mora_event_group_sizes"] = [1] * mora_count
    return report.build_diagnostics(**kw)


# --- build_diagnostics --------------------------------------------------------


def test_phonemes_counts_vowel_and_consonant_segments_not_gap():
    segments = [seg("gap", 0.0, 0.1), seg("consonant", 0.1, 0.15), seg("vowel", 0.15, 0.3), seg("gap", 0.3, 0.4)]
    diag = diag_of(segments=segments, duration_sec=0.4)
    assert diag.phonemes == 2


def test_coverage_is_one_minus_gap_fraction_of_duration():
    segments = [seg("gap", 0.0, 0.1), seg("vowel", 0.1, 0.4)]
    diag = diag_of(segments=segments, duration_sec=0.4)
    assert diag.coverage == pytest.approx(0.75)


def test_coverage_is_zero_when_duration_is_zero_or_negative():
    assert diag_of(duration_sec=0.0).coverage == 0.0
    assert diag_of(duration_sec=-1.0).coverage == 0.0


def test_morae_counts_all_six_vowel_like_shapes():
    mouth_events = [
        mev(MouthShape.A, 0, 10, 0.5), mev(MouthShape.I, 10, 20, 0.5), mev(MouthShape.U, 20, 30, 0.5),
        mev(MouthShape.E, 30, 40, 0.5), mev(MouthShape.O, 40, 50, 0.5), mev(MouthShape.N, 50, 55, 0.3),
        mev(MouthShape.BILABIAL, 55, 57), mev(MouthShape.SILENCE, 57, 60),
    ]
    diag = diag_of(mouth_events=mouth_events, keys=8)
    assert diag.morae == 6


def test_closed_ranges_counts_bilabial_and_silence():
    mouth_events = [mev(MouthShape.A, 0, 10, 0.5), mev(MouthShape.BILABIAL, 10, 12), mev(MouthShape.SILENCE, 12, 20)]
    diag = diag_of(mouth_events=mouth_events, keys=3)
    assert diag.closed_ranges == 2


def test_max_opening_is_max_of_open_amounts():
    mouth_events = [mev(MouthShape.A, 0, 10, 0.4), mev(MouthShape.I, 10, 20, 0.9), mev(MouthShape.O, 20, 30, 0.2)]
    diag = diag_of(mouth_events=mouth_events, keys=3)
    assert diag.max_opening == pytest.approx(0.9)


def test_max_opening_defaults_to_zero_when_no_mouth_events():
    diag = diag_of()
    assert diag.max_opening == 0.0


def test_max_opening_is_zero_when_only_closed_shapes_present():
    mouth_events = [mev(MouthShape.BILABIAL, 0, 5), mev(MouthShape.SILENCE, 5, 10)]
    diag = diag_of(mouth_events=mouth_events, keys=2)
    assert diag.max_opening == 0.0


def test_merged_morae_comes_from_event_diagnostics():
    diag = diag_of(event_diagnostics=events.EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=7))
    assert diag.merged_morae == 7


def test_low_dynamics_comes_from_event_diagnostics():
    # low_dynamicsは機械モードのresultペイロードには載せない(呼び出し側のwarningイベント判定専用)
    # ため、result_run_fields/result_inspect_fieldsのキー集合検証とは別に
    # Diagnostics自体のフィールドとして直接検証する。
    assert diag_of(
        event_diagnostics=events.EventDiagnostics(weak_vowels=0, low_dynamics=True, merged_morae=0)
    ).low_dynamics is True
    assert diag_of(
        event_diagnostics=events.EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=0)
    ).low_dynamics is False


@pytest.mark.xfail(reason="impl pending: 項目1 forced_split警告", strict=True)
def test_forced_split_is_a_diagnostics_field_not_in_result_payload():
    # forced_splitはlow_dynamicsと同じ位置づけ(機械モードのresultペイロードには含めない、
    # 呼び出し側のwarningイベント判定専用のDiagnosticsフィールド)。build_diagnosticsは
    # forced_splitを必須キーワード引数として受け取る。
    kw = dict(
        segments=[], mouth_events=[], mora_event_group_sizes=[],
        event_diagnostics=events.EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=0),
        backends={}, style="pop", separated=False, duration_sec=1.0, keys=0,
    )
    assert report.build_diagnostics(**kw, forced_split=True).forced_split is True
    assert report.build_diagnostics(**kw, forced_split=False).forced_split is False
    fields = report.result_run_fields(report.build_diagnostics(**kw, forced_split=True), output="x.vmd")
    assert "forced_split" not in fields


def test_mora_details_lists_only_vowel_like_events_with_shape_and_hold():
    mouth_events = [mev(MouthShape.A, 0, 10, 0.4), mev(MouthShape.BILABIAL, 10, 12), mev(MouthShape.N, 12, 18, 0.3)]
    diag = diag_of(mouth_events=mouth_events, keys=3)
    assert [(m.shape, m.open_amount, m.hold_frames) for m in diag.mora_details] == [
        ("a", 0.4, 10.0), ("n", 0.3, 6.0),
    ]


# --- render_report_text -------------------------------------------------------


def test_render_report_text_includes_backends_style_params_and_stats():
    diag = diag_of(
        segments=[seg("vowel", 0.0, 0.5)], mouth_events=[mev(MouthShape.A, 0, 15, 0.6)],
        event_diagnostics=events.EventDiagnostics(weak_vowels=1, low_dynamics=False, merged_morae=2),
        backends={"separator": "sep-x", "recognizer": "rec-y", "forced_aligner": "aligner-z"},
        style="ballad", separated=True,
        duration_sec=0.5, keys=4,
    )
    text = report.render_report_text(diag, {"open_lo": 0.2, "open_hi": 0.55})
    # 定めた列挙順(バックエンド・style・params → 分離有無・音素数・モーラ数・被覆率 →
    # モーラごとの明細・併合数 → 閉口区間数・最大開き量・生成キー数・尺)どおりであることを、
    # 各行の出現順(部分文字列の存在でなく行インデックス)で確認する。
    lines = text.splitlines()

    def index_of(substring):
        return next(i for i, line in enumerate(lines) if substring in line)

    order = [
        "separator: sep-x", "recognizer: rec-y", "forced_aligner: aligner-z", "style: ballad",
        "open_lo: 0.2", "open_hi: 0.55",
        "separated: True", "phonemes: 1", "morae: 1", "coverage: 1.0000",
        "mora[1]: shape=a open_amount=0.6000 hold_frames=15.00", "merged_morae: 2", "closed_ranges: 0",
        "max_opening: 0.6000", "keys: 4", "duration_sec: 0.500",
    ]
    indices = [index_of(s) for s in order]
    assert indices == sorted(indices)  # 仕様6.7の列挙順どおりに単調増加


# --- result_run_fields / result_inspect_fields ---------------------------------


def test_result_run_fields_has_exactly_the_12_1_keys_with_diag_values_transcribed():
    diag = diag_of(
        segments=[seg("vowel", 0.0, 0.5), seg("gap", 0.5, 0.6)],
        mouth_events=[mev(MouthShape.A, 0, 15, 0.6), mev(MouthShape.SILENCE, 15, 18)],
        event_diagnostics=events.EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=3),
        backends={"separator": "sep-x", "recognizer": "rec-y"}, style="pop", separated=True,
        duration_sec=0.6, keys=4,
    )
    fields = report.result_run_fields(diag, output="out.vmd")
    assert set(fields) == {
        "output", "keys", "backends", "style", "separated", "phonemes", "morae", "merged_morae",
        "coverage", "closed_ranges", "max_opening", "duration_sec",
    }
    assert fields["output"] == "out.vmd"
    assert fields["keys"] == diag.keys == 4
    assert fields["backends"] == diag.backends == {"separator": "sep-x", "recognizer": "rec-y"}
    assert fields["style"] == diag.style == "pop"
    assert fields["separated"] == diag.separated is True
    assert fields["phonemes"] == diag.phonemes == 1
    assert fields["morae"] == diag.morae == 1
    assert fields["merged_morae"] == diag.merged_morae == 3
    assert fields["coverage"] == diag.coverage == pytest.approx(5.0 / 6.0)
    assert fields["closed_ranges"] == diag.closed_ranges == 1
    assert fields["max_opening"] == diag.max_opening == pytest.approx(0.6)
    assert fields["duration_sec"] == diag.duration_sec == 0.6


def test_result_inspect_fields_adds_input_metadata_and_null_output_with_diag_values_transcribed():
    diag = diag_of(
        segments=[seg("vowel", 0.0, 1.0)], mouth_events=[mev(MouthShape.I, 0, 30, 0.4)],
        event_diagnostics=events.EventDiagnostics(weak_vowels=0, low_dynamics=True, merged_morae=1),
        backends={"separator": "sep-x", "recognizer": "rec-y"}, style="rap", separated=False,
        duration_sec=1.0, keys=2,
    )
    fields = report.result_inspect_fields(diag, input_kind="audio", sample_rate=44100, channels=2)
    assert set(fields) == {
        "output", "keys", "backends", "style", "separated", "phonemes", "morae", "merged_morae",
        "coverage", "closed_ranges", "max_opening", "duration_sec", "input_kind", "sample_rate", "channels",
    }
    assert fields["output"] is None
    assert fields["input_kind"] == "audio"
    assert fields["sample_rate"] == 44100
    assert fields["channels"] == 2
    assert fields["keys"] == diag.keys == 2
    assert fields["backends"] == diag.backends
    assert fields["style"] == diag.style == "rap"
    assert fields["separated"] == diag.separated is False
    assert fields["phonemes"] == diag.phonemes == 1
    assert fields["morae"] == diag.morae == 1
    assert fields["merged_morae"] == diag.merged_morae == 1
    assert fields["coverage"] == diag.coverage == pytest.approx(1.0)
    assert fields["closed_ranges"] == diag.closed_ranges == 0
    assert fields["max_opening"] == diag.max_opening == pytest.approx(0.4)
    assert fields["duration_sec"] == diag.duration_sec == 1.0


# --- 長時間モーラのサブウィンドウ分割時のモーラ単位集計 -------------------------
#
# events.confirm_mouth_events の3件目の戻り値(母音的口形ユニットごとの生成MouthEvent数の
# 列)を build_diagnostics が受け取り、morae・mora_details をサブウィンドウ単位でなく実際の
# モーラ単位で集計することを検証する。


def test_split_mora_counts_as_one_mora_with_averaged_open_amount_and_summed_hold_frames():
    mouth_events = [
        mev(MouthShape.A, 0, 10, 0.2),
        mev(MouthShape.A, 10, 25, 0.5),
        mev(MouthShape.A, 25, 40, 0.8),
    ]
    diag = diag_of(mouth_events=mouth_events, keys=3, mora_event_group_sizes=[3])
    assert diag.morae == 1
    assert len(diag.mora_details) == 1
    assert diag.mora_details[0].shape == "a"
    assert diag.mora_details[0].open_amount == pytest.approx((0.2 + 0.5 + 0.8) / 3)
    assert diag.mora_details[0].hold_frames == pytest.approx(40.0)  # 40-0 の合計


def test_unsplit_morae_alongside_split_mora_count_correctly():
    # 分割されない短いモーラ(先頭・末尾)と分割されたモーラ(中央、2分割)が混在する場合の集計。
    mouth_events = [
        mev(MouthShape.I, 0, 5, 0.3),
        mev(MouthShape.A, 5, 15, 0.2),
        mev(MouthShape.A, 15, 30, 0.6),
        mev(MouthShape.O, 30, 35, 0.4),
    ]
    diag = diag_of(mouth_events=mouth_events, keys=4, mora_event_group_sizes=[1, 2, 1])
    assert diag.morae == 3
    assert [m.shape for m in diag.mora_details] == ["i", "a", "o"]
    assert diag.mora_details[1].open_amount == pytest.approx((0.2 + 0.6) / 2)
    assert diag.mora_details[1].hold_frames == pytest.approx(25.0)  # 30-5 の合計


def test_closed_ranges_and_max_opening_unaffected_by_split():
    # closed_ranges・max_openingは分割数列の影響を受けず、全MouthEventを対象に集計する。
    mouth_events = [
        mev(MouthShape.A, 0, 10, 0.2), mev(MouthShape.A, 10, 20, 0.9),
        mev(MouthShape.BILABIAL, 20, 22), mev(MouthShape.SILENCE, 22, 30),
    ]
    diag = diag_of(mouth_events=mouth_events, keys=4, mora_event_group_sizes=[2])
    assert diag.closed_ranges == 2
    assert diag.max_opening == pytest.approx(0.9)
