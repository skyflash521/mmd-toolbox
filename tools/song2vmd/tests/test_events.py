"""song2vmd 口形イベント確定のテスト。

vocal_analysis の音素セグメント列(母音/子音/gap)と相対正規化RMSから、`lipsync` へ渡す口形イベント列
(`MouthEvent`)と各モーラの開き量を確定する入口処理(events.confirm_mouth_events)を検証する。
音素記号は採用構成(G2P強制アライメント)が実際に出力する記号に基づく:
撥音「ん」は専用記号 `ɴ`、ま行/ば行/ぱ行頭子音は `m`/`mʲ`・`b`/`bʲ`・`p`/`pʲ`。

範囲は口形イベント列(母音・撥音「ん」・両唇閉鎖・無音)と開き量の確定に限る。lipsync へのモーフキー
生成・VMD書き出しは対象外。
"""

import numpy as np
import pytest

from lipsync import ApertureClass, ConsonantClass, MouthShape
from vocal_analysis import RmsEnvelope, Segment

from song2vmd import events

FRAME_RATE = 30.0


def seg(type_, start, end, phoneme=None, confidence=None):
    return Segment(type=type_, start_sec=start, end_sec=end, phoneme=phoneme, confidence=confidence)


def rms_env(times_sec, values, dynamic_range_db=20.0):
    return RmsEnvelope(
        times_sec=np.array(times_sec, dtype=float),
        values=np.array(values, dtype=float),
        dynamic_range_db=dynamic_range_db,
    )


# vocal_analysis.rms の実際のフレーム化(FRAME_SEC=0.025・HOP_SEC=0.010)に合わせ、times_sec は
# 各フレームの中心時刻(先頭フレーム中心はFRAME_SEC/2)とする(libs/vocal_analysis/rms.py・types.py)。
_FRAME_CENTER_OFFSET_SEC = 0.0125


def flat_rms(duration_sec, value, dynamic_range_db=20.0, hop_sec=0.010):
    n = max(2, round(duration_sec / hop_sec) + 1)
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop_sec for i in range(n)]
    values = [value] * n
    return rms_env(times, values, dynamic_range_db)


# 共通の呼び出し既定値(pop相当)。--vowel-gain は lipsync へ渡す vowel_scale の
# 一部であり、song2vmd入口のRMS→開き量写像自体には関与しないため、
# ここでは扱わない。
_DEFAULT_KW = dict(
    open_lo=0.30, open_hi=0.75, open_max=0.90, intensity_curve=0.6,
    silence_on=0.06, use_n_morph=True,
)


def confirm(segments, rms, **overrides):
    kw = dict(_DEFAULT_KW)
    kw.update(overrides)
    return events.confirm_mouth_events(segments, rms, **kw)


# --- 音素由来分類(IPA写像・両唇閉鎖・撥音「ん」。RMS非依存) --------------------


def test_vowel_ipa_maps_to_mouth_shape():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.3, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A]


@pytest.mark.parametrize("phoneme,letter", [
    ("a", MouthShape.A), ("i", MouthShape.I), ("ɯ", MouthShape.U),
    ("e̞", MouthShape.E), ("o̞", MouthShape.O),
])
def test_all_five_vowels_map(phoneme, letter):
    segments = [seg("vowel", 0.0, 0.3, phoneme=phoneme, confidence=0.9)]
    rms = flat_rms(0.3, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == letter


def test_unmapped_vowel_symbol_is_treated_as_gap():
    # IPA写像表に無い母音記号は gap 扱い(phonemes.espeak_ipa_to_vowel)。
    # 先頭gapは無音になる。
    segments = [seg("vowel", 0.0, 0.3, phoneme="ʔ", confidence=0.9)]
    rms = flat_rms(0.3, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.SILENCE


@pytest.mark.parametrize("phoneme", ["m", "mʲ", "b", "bʲ", "p", "pʲ"])
def test_bilabial_consonants_become_independent_closure_event(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.BILABIAL
    assert mouth_events[1].shape == MouthShape.A


def test_fu_consonant_is_not_bilabial_closure():
    # ɸ(ふ)は両唇音だが閉口しない。独立イベントを作らず後続母音へ吸収される。
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="ɸ"),
        seg("vowel", 0.05, 0.35, phoneme="ɯ", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.U
    assert mouth_events[0].start == pytest.approx(0.0)  # 子音区間を吸収して開始が前へ寄る


def test_moraic_nasal_becomes_n_shape():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("consonant", 0.2, 0.35, phoneme="ɴ"),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.N]


def test_no_n_morph_flag_forces_silence_instead_of_n():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("consonant", 0.2, 0.35, phoneme="ɴ"),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms, use_n_morph=False)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.SILENCE]


def test_head_nasal_consonant_before_vowel_is_not_n_mora():
    # な行等の頭子音(n)は撥音「ん」の専用記号ɴと異なるため独立イベントを作らず母音へ吸収される。
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="n"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A]


def test_plain_consonant_creates_no_independent_event():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="k"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert len(mouth_events) == 1
    assert mouth_events[0].shape == MouthShape.A


# --- 先頭子音種別(ConsonantClass)の付与 -------------------------------------


def test_consonant_class_none_when_no_preceding_consonant():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.3, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].consonant_class == ConsonantClass.NONE


@pytest.mark.parametrize("phoneme", ["ɸ", "w"])
def test_consonant_class_rounded(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].consonant_class == ConsonantClass.ROUNDED


@pytest.mark.parametrize("phoneme", ["ɕ", "tɕ", "dʑ", "ɲ", "ç", "kʲ", "bʲ"])
def test_consonant_class_spread(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].consonant_class == ConsonantClass.SPREAD


def test_consonant_class_neutral_for_unlisted_consonant():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="k"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].consonant_class == ConsonantClass.NEUTRAL


# --- 開口減衰種別(ApertureClass)の付与 ---------------------------------------


def test_aperture_class_none_when_no_preceding_consonant():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.3, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].aperture_class == ApertureClass.NONE


@pytest.mark.parametrize("phoneme", ["t", "d", "n", "ts", "ɲ"])
def test_aperture_class_firm_closure(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.FIRM_CLOSURE


@pytest.mark.parametrize("phoneme", ["s", "z", "ɕ", "tɕ", "dʑ", "ç", "j"])
def test_aperture_class_narrow_channel(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.NARROW_CHANNEL


@pytest.mark.parametrize("phoneme", ["k", "ɡ", "ɾ", "kʲ", "ɡʲ"])
def test_aperture_class_slight_closure(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.SLIGHT_CLOSURE


@pytest.mark.parametrize("phoneme", ["ɸ", "w", "h", "v"])
def test_aperture_class_none_for_listed_none_phonemes(phoneme):
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.35, phoneme="ɯ" if phoneme in ("ɸ", "w") else "a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.NONE


def test_aperture_class_none_for_unlisted_consonant():
    # 判定表に無い子音(声門破裂音相当の仮記号)はNONEに倒す。
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="ʔ"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.NONE


def test_aperture_class_does_not_apply_generic_yod_suffix_rule():
    # ConsonantClassのSPREAD(語尾ʲの一致)とは異なり、ApertureClassの拗音は判定表の明示メンバー
    # (kʲ・ɡʲ)だけを判定する(語尾ʲの一致では判定しない)。表に無い拗音はNONE。
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="sʲ"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.NONE


def test_aperture_class_uses_strongest_among_consonant_run_when_strong_comes_last():
    # k(SLIGHT_CLOSURE)→t(FIRM_CLOSURE)の連続子音列では、区切りからその母音までの間で最も強い
    # クラスを採る。
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="k"),
        seg("consonant", 0.03, 0.06, phoneme="t"),
        seg("vowel", 0.06, 0.36, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.36, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.FIRM_CLOSURE


def test_aperture_class_strongest_wins_when_strong_comes_first():
    # t(FIRM_CLOSURE)→k(SLIGHT_CLOSURE)の順(最後の子音は弱い)でも列内の最強クラスを採用する。
    # 「最後の子音だけを見る」誤った実装だとSLIGHT_CLOSUREになってしまう違いを検出する。
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="t"),
        seg("consonant", 0.03, 0.06, phoneme="k"),
        seg("vowel", 0.06, 0.36, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.36, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].aperture_class == ApertureClass.FIRM_CLOSURE


def test_aperture_class_resets_after_bilabial_closure():
    # 両唇閉鎖は区切りとして扱われ、それより前の子音による開口減衰は閉鎖後の母音へ引き継がない。
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="t"),
        seg("consonant", 0.03, 0.08, phoneme="m"),
        seg("vowel", 0.08, 0.38, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.38, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    assert a_event.aperture_class == ApertureClass.NONE


def test_aperture_class_resets_after_moraic_nasal():
    # ɴ(撥音「ん」)自身が区切りとして扱われることを、母音を挟まず子音→ɴ→子音と連続させて検証する
    # (母音を挟む構成だと母音境界だけでリセットが説明でき、ɴ自体のリセットを弁別できない)。
    # ɴがリセットしない誤実装だと列は[t,k]→FIRM_CLOSUREになってしまう。正しくはɴ後のkのみで
    # 判定しSLIGHT_CLOSUREになる。
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="t"),
        seg("consonant", 0.03, 0.13, phoneme="ɴ"),
        seg("consonant", 0.13, 0.16, phoneme="k"),
        seg("vowel", 0.16, 0.46, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.46, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert i_event.aperture_class == ApertureClass.SLIGHT_CLOSURE


def test_aperture_class_resets_after_gap():
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="t"),
        seg("gap", 0.03, 0.13),
        seg("vowel", 0.13, 0.43, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.43, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    assert a_event.aperture_class == ApertureClass.NONE


def test_aperture_class_independent_from_consonant_class():
    # ɲ(にゃ行)はConsonantClass=SPREADかつApertureClass=FIRM_CLOSURE(両軸が一致しない具体例)。
    # 一方の判定がもう一方に影響しないことを確認する。
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="ɲ"),
        seg("vowel", 0.05, 0.35, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.35, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].consonant_class == ConsonantClass.SPREAD
    assert mouth_events[-1].aperture_class == ApertureClass.FIRM_CLOSURE


def test_consonant_run_yields_independent_classes_on_both_axes():
    # k(SLIGHT_CLOSURE)→w(ROUNDED)→aの連続子音では、ConsonantClassは最後の子音(w)から、
    # ApertureClassは列内の最強クラス(k)から、それぞれ独立に決まる。
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="k"),
        seg("consonant", 0.03, 0.06, phoneme="w"),
        seg("vowel", 0.06, 0.36, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.36, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[-1].consonant_class == ConsonantClass.ROUNDED
    assert mouth_events[-1].aperture_class == ApertureClass.SLIGHT_CLOSURE


def test_aperture_class_resets_after_vowel_boundary():
    # 母音イベント自体も区切りとして扱われ、それより前の子音による開口減衰は次のモーラへ
    # 引き継がない。t(FIRM_CLOSURE)の後の母音でリセットされないままだと、
    # 2モーラ目のkはFIRM_CLOSURE(tとの併合で強い方が残る)になってしまうが、正しくは
    # k単独のSLIGHT_CLOSUREになる。
    segments = [
        seg("consonant", 0.0, 0.03, phoneme="t"),
        seg("vowel", 0.03, 0.33, phoneme="a", confidence=0.9),
        seg("consonant", 0.33, 0.36, phoneme="k"),
        seg("vowel", 0.36, 0.66, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.66, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert i_event.aperture_class == ApertureClass.SLIGHT_CLOSURE


# --- gap解決(RMS依存)・低ダイナミクス抑制 -----------------------------------


def test_leading_and_trailing_gap_are_always_silence():
    segments = [
        seg("gap", 0.0, 0.1),
        seg("vowel", 0.1, 0.4, phoneme="a", confidence=0.9),
        seg("gap", 0.4, 0.5),
    ]
    rms = flat_rms(0.5, 0.9)  # 高RMSでも先頭・末尾gapは無音
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.SILENCE
    assert mouth_events[-1].shape == MouthShape.SILENCE


def test_gap_below_silence_threshold_becomes_silence():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.8)
    # gap区間(0.2〜0.4)に中心が落ちる全フレームを無音しきい値(0.06)以下にする。
    for i in range(len(rms.times_sec)):
        if 0.2 <= rms.times_sec[i] < 0.4:
            rms.values[i] = 0.02
    mouth_events, diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.A, MouthShape.SILENCE, MouthShape.I]
    # gap先頭から無音の連続が始まる場合はgap全体が無音になる。直前母音は
    # gap側へ延長されず(継続断片が生じず)、モーラ併合の診断値も増えない。
    assert mouth_events[0].end == pytest.approx(0.2 * FRAME_RATE)
    assert diag.merged_morae == 0


def test_gap_above_silence_threshold_continues_preceding_vowel():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.5)  # 無音しきい値(0.06)を上回る一定音量
    mouth_events, _diag = confirm(segments, rms)
    # gapが直前母音(A)を継続し、後続の同母音(A)とも連結して1イベントになる。
    assert [e.shape for e in mouth_events] == [MouthShape.A]
    assert mouth_events[0].start == pytest.approx(0.0 * FRAME_RATE)
    assert mouth_events[0].end == pytest.approx(0.6 * FRAME_RATE)


def test_gap_above_silence_threshold_continues_preceding_n_mora():
    # 撥音「ん」も母音的口形なので、後続gapのRMSが無音しきい値を上回れば継続する。
    # 末尾に母音区間を置き、末尾gapの強制無音規則(RMSに依らず常に無音)と混同しないようにする。
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("consonant", 0.2, 0.3, phoneme="ɴ"),
        seg("gap", 0.3, 0.5),
        seg("vowel", 0.5, 0.7, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.7, 0.5)  # 無音しきい値(0.06)を上回る一定音量
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.A, MouthShape.N, MouthShape.I]
    # gapがNへ吸収され、Nイベントの終端がgapの終端まで伸びている。
    n_event = mouth_events[1]
    assert n_event.end == pytest.approx(0.5 * FRAME_RATE)


def test_gap_after_bilabial_does_not_continue_as_open():
    # 両唇閉鎖(BILABIAL)は閉口なので、直後のgapはRMSが高くても継続対象にならない(母音的口形ではない)。
    segments = [
        seg("consonant", 0.0, 0.1, phoneme="m"),
        seg("gap", 0.1, 0.3),
        seg("vowel", 0.3, 0.5, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.5, 0.5)  # 無音しきい値を上回る一定音量でも継続しない
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.BILABIAL, MouthShape.SILENCE, MouthShape.A]


def test_gap_after_silence_does_not_continue_as_open():
    # 直前が無音(SILENCE)のgapは、RMSが高くても母音的口形が無いため継続扱いにならない。
    segments = [
        seg("gap", 0.0, 0.1),  # 先頭gap: 常に無音
        seg("gap", 0.1, 0.3),  # 直前が無音のgap: RMSが高くても継続不可
        seg("vowel", 0.3, 0.5, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.5, 0.5)
    mouth_events, _diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.SILENCE, MouthShape.A]


def test_gap_with_voiced_head_and_silent_tail_splits_at_voice_end():
    # 伸ばして歌う発声の尾部(有声)と真の無音が1つのgapに混在する場合、gap全体の一発判定ではなく
    # 走査で分割し、発声が終わった時点から無音にする(gap走査)。
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="e̞", confidence=0.9),
        seg("gap", 0.2, 2.0),
        seg("vowel", 2.0, 2.2, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(2.2, 0.8)
    # gapの前半(0.2〜1.0秒)は発声継続(0.8のまま)、後半(1.0秒〜)を無音レベルにする。
    for i in range(len(rms.times_sec)):
        if 1.0 <= rms.times_sec[i] < 2.0:
            rms.values[i] = 0.02
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.E, MouthShape.SILENCE, MouthShape.A]
    # E(継続込み)は発声が終わる1.0秒付近まで、無音はそこからgap終端(2.0秒)まで。
    assert mouth_events[0].end == pytest.approx(1.0 * FRAME_RATE, abs=0.5)
    assert mouth_events[1].end == pytest.approx(2.0 * FRAME_RATE, abs=0.5)


def test_gap_with_short_dip_does_not_close_when_voice_resumes():
    # gap内の0.2秒未満の瞬間的な谷(ビブラート・トレモロ)では閉口せず、gap全体を継続する
    # (gap走査の連続要件)。
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 1.0),
        seg("vowel", 1.0, 1.2, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(1.2, 0.8)
    # gap中央に0.1秒だけの谷(その後gap内で発声が再開する)。
    for i in range(len(rms.times_sec)):
        if 0.5 <= rms.times_sec[i] < 0.6:
            rms.values[i] = 0.02
    mouth_events, _diag = confirm(segments, rms)
    # 谷では閉じず、gapがA継続のまま前後のAと連結して1イベントになる。
    assert [e.shape for e in mouth_events] == [MouthShape.A]


def test_gap_stays_closed_after_scan_close_even_if_voice_returns():
    # 0.2秒以上の無音で一度閉じたgap内では、後から音量が戻っても再度開かない
    # (無音を挟んで戻る発声は直前母音の継続ではない)。
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 1.4),
        seg("vowel", 1.4, 1.6, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(1.6, 0.8)
    # gap前半(0.4〜0.8秒)を無音レベルにし、gap後半(0.8〜1.4秒)で音量が戻る。
    for i in range(len(rms.times_sec)):
        if 0.4 <= rms.times_sec[i] < 0.8:
            rms.values[i] = 0.02
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.A, MouthShape.SILENCE, MouthShape.I]
    # 無音は谷の開始(0.4秒)からgap終端(1.4秒)まで続く(途中で再開しない)。
    assert mouth_events[1].start == pytest.approx(0.4 * FRAME_RATE, abs=0.5)
    assert mouth_events[1].end == pytest.approx(1.4 * FRAME_RATE, abs=0.5)


def test_gap_short_quiet_tail_reaching_gap_end_closes_without_debounce():
    # gap終端まで達する下降側しきい値以下の連続には0.2秒の連続要件を適用しない
    # (その先で発声が再開しないため)。
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 1.0),
        seg("vowel", 1.0, 1.2, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(1.2, 0.8)
    # gap末尾の0.1秒(0.9〜1.0秒)だけ無音レベル(0.2秒未満だがgap終端に達する)。
    for i in range(len(rms.times_sec)):
        if 0.9 <= rms.times_sec[i] < 1.0:
            rms.values[i] = 0.02
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.A, MouthShape.SILENCE, MouthShape.I]
    assert mouth_events[1].start == pytest.approx(0.9 * FRAME_RATE, abs=0.5)


def test_low_dynamics_suppresses_gap_silence_except_leading_trailing():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="a", confidence=0.9),
        seg("gap", 0.6, 0.7),
    ]
    rms = flat_rms(0.7, 0.5, dynamic_range_db=5.0)  # 低ダイナミクス(閾値12dB未満)
    # 中間gap(0.2〜0.4)は低RMSでも低ダイナミクスなら無音化を抑制。
    idx_lo = int(0.2 / 0.010)
    idx_hi = int(0.4 / 0.010)
    for i in range(idx_lo, idx_hi + 1):
        rms.values[i] = 0.01
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    # 中間gapは継続(無音化されない)ので A が連結、末尾gapは(先頭・末尾規則により)常に無音。
    assert shapes == [MouthShape.A, MouthShape.SILENCE]


# --- 母音区間の無音補正 ------------------------------------------------------


def test_vowel_segment_with_silent_rms_is_overridden_to_silence():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.95)]  # 高信頼度でも
    rms = flat_rms(0.3, 0.01)  # 無音しきい値(0.06)以下
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.SILENCE


def test_low_dynamics_suppresses_vowel_silence_override():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.95)]
    rms = flat_rms(0.3, 0.01, dynamic_range_db=5.0)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.A


def _loud_consonant_quiet_core_fixture():
    """伸ばして歌う発声の大部分が先行子音トークンへ割り当てられたモーラを再現する
    (モーラ代表RMSの、モーラ区間全体の窓が要る事例)。

    子音 n [0.0, 0.9](発声の実体。高RMS)+ 母音 i [0.9, 0.92](狭い母音核。低RMS)。
    母音核の中央60%だけを見ると無音しきい値以下になるが、モーラ区間全体では大音量。
    """
    segments = [
        seg("consonant", 0.0, 0.9, phoneme="n"),
        seg("vowel", 0.9, 0.92, phoneme="i", confidence=0.9),
    ]
    hop = 0.010
    n = 93
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [0.9 if t <= 0.85 else 0.02 for t in times]
    return segments, rms_env(times, values)


def test_vowel_with_quiet_core_but_loud_mora_span_is_not_silenced():
    segments, rms = _loud_consonant_quiet_core_fixture()
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.I


def test_vowel_with_quiet_core_but_loud_mora_span_opens_from_the_louder_window():
    segments, rms = _loud_consonant_quiet_core_fixture()
    mouth_events, _diag = confirm(
        segments, rms, intensity_curve=1.0, open_lo=0.0, open_hi=1.0, open_max=1.0)
    # モーラ区間全体[0.0,0.92]の中央60%はほぼ0.9。母音核だけの窓(≈0.02)に引きずられない。
    assert mouth_events[0].open_amount == pytest.approx(0.9, abs=0.05)


def test_vowel_with_absorbed_consonant_still_silenced_when_whole_mora_is_quiet():
    # 二窓のどちらも無音しきい値以下なら閉口する(誤検出母音対策)。
    segments = [
        seg("consonant", 0.0, 0.2, phoneme="n"),
        seg("vowel", 0.2, 0.3, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.3, 0.01)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].shape == MouthShape.SILENCE


# --- 同母音連結 --------------------------------------------------------------


def test_adjacent_same_vowel_segments_merge_into_one_event():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("vowel", 0.2, 0.4, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.4, 0.8)
    mouth_events, diag = confirm(segments, rms)
    assert len(mouth_events) == 1
    assert mouth_events[0].shape == MouthShape.A
    assert mouth_events[0].start == pytest.approx(0.0)
    assert mouth_events[0].end == pytest.approx(0.4 * FRAME_RATE)
    assert diag.merged_morae == 1  # 2区間→1イベントで1回統合


def test_three_adjacent_same_vowel_segments_merge_with_merged_count_two():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("vowel", 0.2, 0.4, phoneme="a", confidence=0.9),
        seg("vowel", 0.4, 0.6, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.8)
    mouth_events, diag = confirm(segments, rms)
    assert len(mouth_events) == 1
    assert diag.merged_morae == 2  # 3区間→1イベントで2回統合


def test_adjacent_different_vowels_do_not_merge():
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("vowel", 0.2, 0.4, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.4, 0.8)
    mouth_events, diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.I]
    assert diag.merged_morae == 0


@pytest.mark.parametrize("phoneme", ["t", "k", "s"])
def test_repeated_consonant_between_same_vowels_does_not_merge(phoneme):
    # 「たた」: 間の子音が前後で同じ音素でも、モーラの区切り(子音の再構音)は実在するため統合しない
    # (lipsyncのモーラ境界の谷で区別できるようにする)。
    segments = [
        seg("consonant", 0.0, 0.05, phoneme=phoneme),
        seg("vowel", 0.05, 0.25, phoneme="a", confidence=0.9),
        seg("consonant", 0.25, 0.30, phoneme=phoneme),
        seg("vowel", 0.30, 0.50, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.50, 0.8)
    mouth_events, diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.A]
    assert diag.merged_morae == 0
    assert mouth_events[0].end == pytest.approx(0.25 * FRAME_RATE)
    assert mouth_events[1].start == pytest.approx(0.25 * FRAME_RATE)


def test_different_intervening_consonants_between_same_vowels_do_not_merge():
    # 「たか」: 間の子音が前後で異なる場合はもとより統合しない(上の「たた」と対称の回帰)。
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="t"),
        seg("vowel", 0.05, 0.25, phoneme="a", confidence=0.9),
        seg("consonant", 0.25, 0.30, phoneme="k"),
        seg("vowel", 0.30, 0.50, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.50, 0.8)
    mouth_events, diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.A]
    assert diag.merged_morae == 0


def test_adjacent_silence_segments_merging_does_not_count_as_merged_morae():
    # 無音(閉口)区間同士の統合はモーラの併合ではないため merged_morae に数えない。
    segments = [
        seg("gap", 0.0, 0.2),
        seg("gap", 0.2, 0.4),
    ]
    rms = flat_rms(0.4, 0.01)
    mouth_events, diag = confirm(segments, rms)
    assert len(mouth_events) == 1
    assert mouth_events[0].shape == MouthShape.SILENCE
    assert diag.merged_morae == 0


# --- RMSオンセット補正 -------------------------------------------------------


def test_vowel_onset_shifts_to_nearby_rms_rise():
    # トークン境界は0.30sだが、実際の立ち上がりはその手前(補正窓±60ms以内)にある。
    hop = 0.010
    n = 60
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [0.05] * n
    rise_index = round((0.28 - _FRAME_CENTER_OFFSET_SEC) / hop)
    rise_time = times[rise_index]  # フレーム中心時刻ベースの実際の立ち上がり時刻
    for i in range(rise_index, n):
        values[i] = 0.8
    rms = rms_env(times, values)
    segments = [
        seg("gap", 0.0, 0.30),
        seg("vowel", 0.30, 0.59, phoneme="a", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, rms)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    # トークン境界(0.30s)そのままにはならず、実際の立ち上がり近傍へ寄る。
    assert a_event.start < 0.30 * FRAME_RATE
    assert a_event.start == pytest.approx(rise_time * FRAME_RATE, abs=0.03 * FRAME_RATE)


def test_vowel_onset_keeps_token_boundary_when_no_rise_nearby():
    rms = flat_rms(0.6, 0.5)  # 立ち上がりが無い(終始一定)
    segments = [
        seg("gap", 0.0, 0.30),
        seg("vowel", 0.30, 0.59, phoneme="a", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, rms)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    assert a_event.start == pytest.approx(0.30 * FRAME_RATE)


# --- 低信頼・無声母音判定 -----------------------------------------------------


def test_low_confidence_and_low_rms_vowel_is_weakened():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.2)]
    rms = flat_rms(0.3, 0.25)  # 無音しきい値は超えるが弱い判定のRMS閾値(0.3)未満
    mouth_events, _diag = confirm(segments, rms)
    expected_base = min(max(0.25 ** 0.6, 0.30), 0.75)
    assert mouth_events[0].open_amount == pytest.approx(expected_base * 0.5)


def test_vowel_not_weakened_when_confidence_is_high_even_with_low_rms():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.3, 0.25)  # RMS<0.3 だが信頼度が高いので弱判定にならない
    mouth_events, _diag = confirm(segments, rms)
    expected = min(max(0.25 ** 0.6, 0.30), 0.75)
    assert mouth_events[0].open_amount == pytest.approx(expected)


def test_vowel_not_weakened_when_rms_is_high_even_with_low_confidence():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.2)]
    rms = flat_rms(0.3, 0.5)  # RMS>=0.3 なので信頼度が低くても弱判定にならない
    mouth_events, _diag = confirm(segments, rms)
    expected = min(max(0.5 ** 0.6, 0.30), 0.75)
    assert mouth_events[0].open_amount == pytest.approx(expected)


def test_low_rms_without_confidence_uses_lower_weak_threshold():
    # 信頼度を出さないバックエンド(confidence=None)ではRMS<0.2(0.3ではなく)で弱判定になる。
    weak_segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=None)]
    weak_events, _ = confirm(weak_segments, flat_rms(0.3, 0.15))  # < 0.2
    expected_weak_base = min(max(0.15 ** 0.6, 0.30), 0.75)
    assert weak_events[0].open_amount == pytest.approx(expected_weak_base * 0.5)

    strong_segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=None)]
    strong_events, _ = confirm(strong_segments, flat_rms(0.3, 0.25))  # >= 0.2, 弱判定にならない
    expected_strong = min(max(0.25 ** 0.6, 0.30), 0.75)
    assert strong_events[0].open_amount == pytest.approx(expected_strong)


# --- 開き量の決定(RMS→開き量写像) -------------------------------------------


def test_open_amount_uses_intensity_curve_and_clamps_to_style_range():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.3, 0.5)
    mouth_events, _diag = confirm(segments, rms, open_lo=0.30, open_hi=0.75, intensity_curve=0.6)
    expected = min(max(0.5 ** 0.6, 0.30), 0.75)
    assert mouth_events[0].open_amount == pytest.approx(expected)


def test_open_amount_respects_open_max_upper_bound():
    segments = [seg("vowel", 0.0, 0.3, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.3, 0.99)
    mouth_events, _diag = confirm(
        segments, rms, open_lo=0.30, open_hi=0.95, open_max=0.5, intensity_curve=1.0)
    assert mouth_events[0].open_amount == pytest.approx(0.5)


def test_bilabial_and_silence_have_zero_open_amount():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="m"),
        seg("vowel", 0.05, 0.2, phoneme="a", confidence=0.01),
    ]
    rms = flat_rms(0.2, 0.01)  # 母音区間も無音補正でSILENCEになる
    mouth_events, _diag = confirm(segments, rms)
    assert all(e.open_amount == 0.0 for e in mouth_events)


def test_open_amount_uses_middle_60_percent_of_vowel_segment():
    # 母音区間[0,1.0]の中央60%([0.2,0.8])だけ高RMS、両端(子音トランジェント相当)は低RMSにする
    # (モーラ代表RMS)。中央60%平均(≈0.8)を使うはずで、両端に引きずられる
    # 区間全体平均より明らかに大きくなる。
    hop = 0.010
    n = 101
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [0.8 if 0.2 <= t <= 0.8 else 0.05 for t in times]
    rms = rms_env(times, values)
    segments = [seg("vowel", 0.0, 1.0, phoneme="a", confidence=0.9)]
    mouth_events, _diag = confirm(
        segments, rms, intensity_curve=1.0, open_lo=0.0, open_hi=1.0, open_max=1.0)
    naive_full_average = sum(values) / len(values)
    assert mouth_events[0].open_amount > naive_full_average + 0.1
    assert mouth_events[0].open_amount == pytest.approx(0.8, abs=0.05)


def test_n_mora_open_amount_is_derived_from_rms():
    # 撥音「ん」も母音的口形として区間代表RMSから開き量を決める。
    segments = [
        seg("vowel", 0.0, 0.1, phoneme="a", confidence=0.9),
        seg("consonant", 0.1, 0.3, phoneme="ɴ"),
    ]
    rms = flat_rms(0.3, 0.5)
    mouth_events, _diag = confirm(
        segments, rms, open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)
    n_event = next(e for e in mouth_events if e.shape == MouthShape.N)
    assert n_event.open_amount == pytest.approx(0.5, abs=1e-6)


# --- 全体被覆・時間順・非重複(lipsyncの入力契約) -----------------------------


def test_events_are_time_ordered_contiguous_and_cover_full_range():
    segments = [
        seg("gap", 0.0, 0.05),
        seg("consonant", 0.05, 0.10, phoneme="k"),
        seg("vowel", 0.10, 0.30, phoneme="a", confidence=0.9),
        seg("consonant", 0.30, 0.35, phoneme="m"),
        seg("vowel", 0.35, 0.55, phoneme="i", confidence=0.9),
        seg("gap", 0.55, 0.65),
    ]
    rms = flat_rms(0.65, 0.6)
    mouth_events, _diag = confirm(segments, rms)
    assert mouth_events[0].start == pytest.approx(0.0)
    assert mouth_events[-1].end == pytest.approx(0.65 * FRAME_RATE)
    for event in mouth_events:
        assert event.start <= event.end  # 負長イベントを作らない
    for prev, nxt in zip(mouth_events, mouth_events[1:]):
        assert prev.end == pytest.approx(nxt.start)  # 隙間なく連続(非重複も兼ねる)
        assert prev.start <= nxt.start  # 時間順(開始時刻が後退しない)


def test_empty_segment_list_returns_empty_events():
    rms = flat_rms(0.1, 0.5)
    mouth_events, diag = confirm([], rms)
    assert mouth_events == []
    assert diag.merged_morae == 0


# --- 決定論(同じ入力から同じ出力) --------------------------------------------


def test_confirmation_is_deterministic():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="k"),
        seg("vowel", 0.05, 0.30, phoneme="a", confidence=0.9),
        seg("gap", 0.30, 0.40),
        seg("vowel", 0.40, 0.60, phoneme="i", confidence=0.4),
    ]
    rms = flat_rms(0.60, 0.5)
    first, _ = confirm(segments, rms)
    second, _ = confirm(segments, rms)
    assert first == second
