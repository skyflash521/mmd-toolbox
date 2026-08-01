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
from song2vmd import events
from vocal_analysis import AudioPcm, RmsEnvelope, Segment
from vocal_analysis import rms as _va_rms

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
    result = events.confirm_mouth_events(segments, rms, **kw)
    return result[0], result[1]  # mouth_events, EventDiagnostics(3件目のモーラ分割数列は無視)


# --- モーラ代表RMSの声量レンジ再正規化を検証するための較正ヘルパー ------------------
#
# 開き量の決定は、生のモーラ代表RMSではなく、曲全体の全モーラ集合を
# パーセンタイル(p10/p90)線形正規化した値を使う。対象モーラが1件だけの
# フィクスチャでは、その1件だけでp10==p90となり縮退して常に0.5に丸まるため、生RMSへの
# 依存性を検証できない。そこで下記アンカー2件(下限・上限)を対象モーラの前後に追加し、
# 曲全体のモーラ集合を3件(下アンカー・対象・上アンカー)に固定する。3件・p10/p90の線形補間
# では、下アンカー<対象<上アンカーの並びのとき正規化後の値が
# (対象の生RMS - 下アンカー) / (上アンカー - 下アンカー) という厳密な線形写像になる
# (p10・p90の補間位置がちょうど打ち消し合うため)。下アンカーは無音しきい値(既定0.06)を
# 上回る0.10、上アンカーは1.00とし、対象の生RMSはこの開区間(0.10, 1.00)に収まる値を使う。

_ANCHOR_LO_RMS = 0.10
_ANCHOR_HI_RMS = 1.00


def _anchor_normalized(raw_rms):
    """上記アンカー較正のもとでの再正規化後RMS(既知の線形写像)。"""
    return (raw_rms - _ANCHOR_LO_RMS) / (_ANCHOR_HI_RMS - _ANCHOR_LO_RMS)


def confirm_flat_calibrated(target_rms_value, *, kind="vowel", vowel_phoneme="a",
                             confidence=0.9, mora_dur=0.3, **kw):
    """対象モーラ1件(区間内RMS一定)を下アンカー・上アンカーの母音モーラで挟んでconfirmし、
    対象モーラのイベントだけを返す(_anchor_normalizedの前提を満たす3モーラ構成)。
    """
    lo_end = mora_dur
    target_end = 2 * mora_dur
    hi_end = 3 * mora_dur
    if kind == "vowel":
        target_seg = seg("vowel", lo_end, target_end, phoneme=vowel_phoneme, confidence=confidence)
    else:  # kind == "n"
        target_seg = seg("consonant", lo_end, target_end, phoneme="ɴ")
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        target_seg,
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [
        _ANCHOR_LO_RMS if t < lo_end else (target_rms_value if t < target_end else _ANCHOR_HI_RMS)
        for t in times
    ]
    rms = rms_env(times, values)
    mouth_events, diag = confirm(segments, rms, **kw)
    return mouth_events[1], diag  # [下アンカー, 対象, 上アンカー] の中央が対象


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


def test_moraic_nasal_is_silence_when_use_n_morph_omitted():
    # confirm_mouth_events 自身の use_n_morph 既定値(off)を直接検証する(confirm() ヘルパーは
    # _DEFAULT_KW で常に True を明示するため、この既定値の回帰は検出できない)。
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("consonant", 0.2, 0.35, phoneme="ɴ"),
    ]
    rms = flat_rms(0.35, 0.8)
    kw = {k: v for k, v in _DEFAULT_KW.items() if k != "use_n_morph"}
    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **kw)
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
    # 走査で分割し、発声が終わった時点から無音にする(gap走査)。継続後のE区間長は長時間モーラの
    # サブウィンドウ分割閾値(1.0秒)未満(0.6秒)に保つ。
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="e̞", confidence=0.9),
        seg("gap", 0.2, 1.4),
        seg("vowel", 1.4, 1.6, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(1.6, 0.8)
    # gapの前半(0.2〜0.6秒)は発声継続(0.8のまま)、後半(0.6秒〜)を無音レベルにする。
    for i in range(len(rms.times_sec)):
        if 0.6 <= rms.times_sec[i] < 1.4:
            rms.values[i] = 0.02
    mouth_events, _diag = confirm(segments, rms)
    shapes = [e.shape for e in mouth_events]
    assert shapes == [MouthShape.E, MouthShape.SILENCE, MouthShape.A]
    # E(継続込み)は発声が終わる0.6秒付近まで、無音はそこからgap終端(1.4秒)まで。
    assert mouth_events[0].end == pytest.approx(0.6 * FRAME_RATE, abs=0.5)
    assert mouth_events[1].end == pytest.approx(1.4 * FRAME_RATE, abs=0.5)


def test_gap_with_short_dip_does_not_close_when_voice_resumes():
    # gap内の0.2秒未満の瞬間的な谷(ビブラート・トレモロ)では閉口せず、gap全体を継続する
    # (gap走査の連続要件)。連結後のA区間長は長時間モーラのサブウィンドウ分割閾値(1.0秒)未満
    # (0.9秒)に保つ。
    segments = [
        seg("vowel", 0.0, 0.15, phoneme="a", confidence=0.9),
        seg("gap", 0.15, 0.75),
        seg("vowel", 0.75, 0.9, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.9, 0.8)
    # gap中央に0.075秒だけの谷(その後gap内で発声が再開する)。
    for i in range(len(rms.times_sec)):
        if 0.375 <= rms.times_sec[i] < 0.45:
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


def _loud_consonant_quiet_core_fixture_calibrated():
    """_loud_consonant_quiet_core_fixture を0.3秒後ろへずらし、アンカー較正
    (_anchor_normalized の前提)の下アンカー・上アンカーで挟む。"""
    offset = 0.3
    hi_end = offset + 0.92 + offset
    segments = [
        seg("vowel", 0.0, offset, phoneme="ɯ", confidence=0.9),
        seg("consonant", offset, offset + 0.9, phoneme="n"),
        seg("vowel", offset + 0.9, offset + 0.92, phoneme="i", confidence=0.9),
        seg("vowel", offset + 0.92, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = []
    for t in times:
        if t < offset:
            values.append(_ANCHOR_LO_RMS)
        elif t <= offset + 0.85:
            values.append(0.9)
        elif t < offset + 0.92:
            values.append(0.02)
        else:
            values.append(_ANCHOR_HI_RMS)
    return segments, rms_env(times, values)


def test_vowel_with_quiet_core_but_loud_mora_span_opens_from_the_louder_window():
    segments, rms = _loud_consonant_quiet_core_fixture_calibrated()
    mouth_events, _diag = confirm(
        segments, rms, intensity_curve=1.0, open_lo=0.0, open_hi=1.0, open_max=1.0)
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    # モーラ区間全体の中央60%窓平均はほぼ0.9(母音核だけの窓≈0.02に引きずられない)。単独
    # モーラでは再正規化が縮退して常に0.5になりこの窓ロジックを開き量から検証できないため、
    # アンカー2件で曲全体のモーラ集合(3件)を模し、_anchor_normalized(0.9) に厳密一致させる。
    assert i_event.open_amount == pytest.approx(_anchor_normalized(0.9), abs=1e-6)


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


def test_gap_continuation_fragment_does_not_count_as_merged_mora():
    # gapが直前母音の発声継続になった断片は、1つの発声を時間で切った断片でモーラではないので、
    # 直前区間へまとまっても併合したモーラ数には数えない。末尾のgapはRMSに依らず閉口するので、
    # 継続の経路へ乗せるために後続の母音を置く。
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.5)  # 無音しきい値(0.06)を上回る一定音量(gap全体が発声継続になる)
    mouth_events, diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A, MouthShape.I]
    assert diag.merged_morae == 0


def test_gap_continuation_between_same_vowels_counts_only_the_real_mora():
    # 実モーラ2つの間にgap継続の断片が挟まって1イベントにまとまる場合、数えるのは実モーラの統合だけ。
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="a", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.5)
    mouth_events, diag = confirm(segments, rms)
    assert [e.shape for e in mouth_events] == [MouthShape.A]
    assert diag.merged_morae == 1


def test_adjacent_moraic_nasal_segments_merge_with_merged_count_one():
    # 撥音どうしの統合は実モーラの統合なので数える(継続断片の除外を撥音の一律除外にしない)。
    segments = [
        seg("consonant", 0.0, 0.2, phoneme="ɴ"),
        seg("consonant", 0.2, 0.4, phoneme="ɴ"),
    ]
    rms = flat_rms(0.4, 0.5)
    mouth_events, diag = confirm(segments, rms, use_n_morph=True)
    assert [e.shape for e in mouth_events] == [MouthShape.N]
    assert diag.merged_morae == 1


def test_gap_continuation_fragment_of_moraic_nasal_does_not_count():
    # 撥音の区間を継続したgapの断片も同じ扱いにする(母音側だけの除外にしない)。
    segments = [
        seg("consonant", 0.0, 0.2, phoneme="ɴ"),
        seg("gap", 0.2, 0.4),
        seg("vowel", 0.4, 0.6, phoneme="i", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.5)
    mouth_events, diag = confirm(segments, rms, use_n_morph=True)
    assert [e.shape for e in mouth_events] == [MouthShape.N, MouthShape.I]
    assert diag.merged_morae == 0


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


def step_rms(duration_sec, rise_sec, low=0.2, high=0.9, hop_sec=0.010):
    """rise_sec 以降だけ音量が上がる包絡。平滑化の窓ぶん、立ち上がり近傍の数フレームが候補になる。"""
    n = max(2, round(duration_sec / hop_sec) + 1)
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop_sec for i in range(n)]
    values = [high if t >= rise_sec else low for t in times]
    return rms_env(times, values)


_ONE_FRAME_SEC = 1.0 / FRAME_RATE


# 補正されるのは母音区間の開始だけなので、検証したい境界以外の母音の開始は、立ち上がりから
# 補正窓(±60ms)より遠ざけて波及を断つ。直前区間の長さを詰めたい場合は、補正対象外の両唇閉鎖を
# 直前へ置いて母音どうしの間隔の制約を避ける。


def test_onset_keeps_one_frame_for_the_preceding_unit():
    # 立ち上がりが直前区間の開始側にあっても、直前区間を1フレームより短くしない。
    segments = [
        seg("vowel", 0.0, 0.25, phoneme="a", confidence=0.9),
        seg("consonant", 0.25, 0.28, phoneme="m"),
        seg("vowel", 0.28, 0.60, phoneme="i", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, step_rms(0.60, 0.24))
    bilabial = next(e for e in mouth_events if e.shape == MouthShape.BILABIAL)
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert bilabial.end == pytest.approx((0.25 + _ONE_FRAME_SEC) * FRAME_RATE)
    assert i_event.start == bilabial.end
    assert bilabial.end - bilabial.start == pytest.approx(_ONE_FRAME_SEC * FRAME_RATE)


def test_onset_keeps_one_frame_for_the_corrected_vowel():
    # 立ち上がりが当該母音の終了側にあっても、その母音を1フレームより短くしない。
    segments = [
        seg("gap", 0.0, 0.30),
        seg("vowel", 0.30, 0.32, phoneme="a", confidence=0.9),
        seg("consonant", 0.32, 0.45, phoneme="m"),
        seg("vowel", 0.45, 0.60, phoneme="i", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, step_rms(0.60, 0.34))
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    assert a_event.start == pytest.approx((0.32 - _ONE_FRAME_SEC) * FRAME_RATE)
    assert a_event.end - a_event.start == pytest.approx(_ONE_FRAME_SEC * FRAME_RATE)


def test_onset_lower_bound_follows_the_corrected_start_of_the_preceding_vowel():
    # 直前が母音でその開始も補正で動く場合、下限は動いた後の開始を基準にする(補正前の開始を
    # 基準にすると、直前の母音が1フレームより短くなる)。
    segments = [
        seg("vowel", 0.20, 0.30, phoneme="a", confidence=0.9),
        seg("vowel", 0.30, 0.60, phoneme="i", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, step_rms(0.60, 0.24))
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert i_event.start == a_event.end
    assert a_event.end - a_event.start == pytest.approx(_ONE_FRAME_SEC * FRAME_RATE)


def test_onset_is_not_applied_when_the_allowed_range_is_empty():
    # 直前区間にも当該母音にも1フレームを残せないときは補正せず、トークン境界をそのまま使う。
    segments = [
        seg("vowel", 0.0, 0.25, phoneme="a", confidence=0.9),
        seg("consonant", 0.25, 0.28, phoneme="m"),
        seg("vowel", 0.28, 0.29, phoneme="i", confidence=0.9),
        seg("consonant", 0.29, 0.33, phoneme="m"),
        seg("vowel", 0.33, 0.60, phoneme="e̞", confidence=0.9),
    ]
    mouth_events, _diag = confirm(segments, step_rms(0.60, 0.24))
    i_event = next(e for e in mouth_events if e.shape == MouthShape.I)
    assert i_event.start == pytest.approx(0.28 * FRAME_RATE)


# --- 低信頼・無声母音判定 -----------------------------------------------------


def test_low_confidence_and_low_rms_vowel_is_weakened():
    # 単独モーラでは再正規化が縮退し常に0.5になり生RMS依存の弱判定を開き量から検証できない
    # ため、アンカー較正(confirm_flat_calibrated)で曲全体のモーラ集合を模す。
    # 無音しきい値は超えるが弱判定のRMS閾値(0.3)未満
    target_event, _diag = confirm_flat_calibrated(0.25, confidence=0.2)
    normalized = _anchor_normalized(0.25)
    expected_base = min(max(normalized ** 0.6, 0.30), 0.75)
    assert target_event.open_amount == pytest.approx(expected_base * 0.5)


def test_vowel_not_weakened_when_confidence_is_high_even_with_low_rms():
    target_event, _diag = confirm_flat_calibrated(0.25, confidence=0.9)  # RMS<0.3 だが信頼度が高いので弱判定にならない
    normalized = _anchor_normalized(0.25)
    expected = min(max(normalized ** 0.6, 0.30), 0.75)
    assert target_event.open_amount == pytest.approx(expected)


def test_vowel_not_weakened_when_rms_is_high_even_with_low_confidence():
    # RMS>=0.3 なので信頼度が低くても弱判定にならない
    target_event, _diag = confirm_flat_calibrated(0.5, confidence=0.2)
    normalized = _anchor_normalized(0.5)
    expected = min(max(normalized ** 0.6, 0.30), 0.75)
    assert target_event.open_amount == pytest.approx(expected)


def test_low_rms_without_confidence_uses_lower_weak_threshold():
    # 信頼度を出さないバックエンド(confidence=None)ではRMS<0.2(0.3ではなく)で弱判定になる。
    weak_event, _ = confirm_flat_calibrated(0.15, confidence=None)  # < 0.2
    weak_normalized = _anchor_normalized(0.15)
    expected_weak_base = min(max(weak_normalized ** 0.6, 0.30), 0.75)
    assert weak_event.open_amount == pytest.approx(expected_weak_base * 0.5)

    strong_event, _ = confirm_flat_calibrated(0.25, confidence=None)  # >= 0.2, 弱判定にならない
    strong_normalized = _anchor_normalized(0.25)
    expected_strong = min(max(strong_normalized ** 0.6, 0.30), 0.75)
    assert strong_event.open_amount == pytest.approx(expected_strong)


# --- 開き量の決定(RMS→開き量写像) -------------------------------------------


def test_open_amount_uses_intensity_curve_and_clamps_to_style_range():
    # 単独モーラでは再正規化が縮退し常に0.5になり生RMS依存を開き量から検証できないため、
    # アンカー較正(confirm_flat_calibrated)で曲全体のモーラ集合を模す。
    target_event, _diag = confirm_flat_calibrated(0.5, open_lo=0.30, open_hi=0.75, intensity_curve=0.6)
    normalized = _anchor_normalized(0.5)
    expected = min(max(normalized ** 0.6, 0.30), 0.75)
    assert target_event.open_amount == pytest.approx(expected)


def test_open_amount_respects_open_max_upper_bound():
    # open_maxのクランプは、入力(母音代表RMSの写像値)がopen_hiを上回っていれば、その具体的な
    # 値によらずopen_maxで一律に飽和する。
    target_event, _diag = confirm_flat_calibrated(
        0.99, open_lo=0.30, open_hi=0.95, open_max=0.5, intensity_curve=1.0)
    assert target_event.open_amount == pytest.approx(0.5)


def test_bilabial_and_silence_have_zero_open_amount():
    segments = [
        seg("consonant", 0.0, 0.05, phoneme="m"),
        seg("vowel", 0.05, 0.2, phoneme="a", confidence=0.01),
    ]
    rms = flat_rms(0.2, 0.01)  # 母音区間も無音補正でSILENCEになる
    mouth_events, _diag = confirm(segments, rms)
    assert all(e.open_amount == 0.0 for e in mouth_events)


def test_open_amount_uses_middle_60_percent_of_vowel_segment():
    # 母音区間の中央60%だけ高RMS、両端(子音トランジェント相当)は低RMSにする(モーラ代表RMS)。
    # 中央60%平均(≈0.8)を使うはずで、両端に引きずられる区間全体平均より明らかに大きくなる。
    # 単独モーラでは再正規化が縮退し常に0.5になりこの窓ロジックを開き量から検証できないため、
    # アンカー2件(下限・上限)を対象モーラの前後に置き、曲全体のモーラ集合(3件)を模す。
    offset = 0.3
    target_dur = 0.9  # 長時間モーラのサブウィンドウ分割閾値(1.0秒)未満に保つ
    hi_end = offset + target_dur + offset
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = []
    target_values = []
    for t in times:
        if t < offset:
            values.append(_ANCHOR_LO_RMS)
        elif t < offset + target_dur:
            v = 0.8 if 0.2 * target_dur <= (t - offset) <= 0.8 * target_dur else 0.05
            values.append(v)
            target_values.append(v)
        else:
            values.append(_ANCHOR_HI_RMS)
    rms = rms_env(times, values)
    segments = [
        seg("vowel", 0.0, offset, phoneme="ɯ", confidence=0.9),
        seg("vowel", offset, offset + target_dur, phoneme="a", confidence=0.9),
        seg("vowel", offset + target_dur, hi_end, phoneme="o̞", confidence=0.9),
    ]
    mouth_events, _diag = confirm(
        segments, rms, intensity_curve=1.0, open_lo=0.0, open_hi=1.0, open_max=1.0)
    a_event = next(e for e in mouth_events if e.shape == MouthShape.A)
    naive_full_average = sum(target_values) / len(target_values)
    assert a_event.open_amount > _anchor_normalized(naive_full_average) + 0.1
    assert a_event.open_amount == pytest.approx(_anchor_normalized(0.8), abs=1e-6)


def test_n_mora_open_amount_is_derived_from_rms():
    # 撥音「ん」も母音的口形として区間代表RMSから開き量を決める。単独モーラでは再正規化が
    # 縮退し常に0.5になりRMS依存を検証できないため、アンカー較正で曲全体のモーラ集合を模す。
    target_event, _diag = confirm_flat_calibrated(
        0.7, kind="n", open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)
    assert target_event.shape == MouthShape.N
    assert target_event.open_amount == pytest.approx(_anchor_normalized(0.7), abs=1e-6)


# --- モーラ代表RMSの声量レンジ再正規化そのもの -------------------------------


def _renormalized(values):
    """曲全体のモーラ代表RMS集合を、confirm_mouth_events と同じ手順で p10/p90 正規化する。

    実経路は境界の算出(_percentile_bounds)と1件ごとの正規化(_normalize_with_bounds)を
    別々に呼ぶため、集合単位の性質を検証するここでも同じ2段で組む。
    """
    p_lo, p_hi = events._percentile_bounds(values)
    return [events._normalize_with_bounds(v, p_lo, p_hi) for v in values]


def test_renormalize_open_rms_stretches_skewed_distribution():
    # 曲全体のモーラ代表RMSが高い側に偏っていても、パーセンタイル(p10/p90)線形正規化で
    # 0〜1のレンジへ引き伸ばされる。
    values = [0.70, 0.75, 0.80, 0.81, 0.85, 0.89, 0.90, 0.95, 0.98, 1.00]
    result = _renormalized(values)
    arr = np.array(values)
    p10 = np.percentile(arr, 10, method="linear")
    p90 = np.percentile(arr, 90, method="linear")
    expected = np.clip((arr - p10) / (p90 - p10), 0.0, 1.0)
    assert result == pytest.approx(list(expected))
    # 少なくとも1件はレンジ下限(0)近傍・1件は上限(1)近傍まで引き伸ばされている
    # (偏った入力のまま0.5〜1.0付近に固まっていた不具合の再発を防ぐ)。
    assert min(result) < 0.15
    assert max(result) > 0.85


def test_renormalize_open_rms_single_value_is_degenerate_midpoint():
    # モーラが1件だけの曲はp90==p10で縮退し、無音側(0.0)でなく開閉の中間値0.5に倒す
    # (無音判定用の正規化とは異なり、開き量という用途では無音側に倒すと不自然なため)。
    assert _renormalized([0.42]) == [0.5]


def test_renormalize_open_rms_all_identical_values_are_degenerate_midpoint():
    assert _renormalized([0.6, 0.6, 0.6, 0.6]) == [0.5, 0.5, 0.5, 0.5]


def test_renormalize_open_rms_tiny_positive_range_is_not_degenerate():
    # p90とp10の差がnp.iscloseの既定許容誤差(0.5付近でおよそ5e-6)より小さい、完全一致では
    # ない極めて小さい正のレンジ(差は約1e-6)では、縮退扱い(0.5への丸め)にせず通常の線形
    # 正規化(0〜1へ大きく引き伸ばす)を適用する(完全一致のみを縮退とする方針の実装が
    # np.isclose等の許容誤差判定へ後退していないかを検出する)。
    values = [0.500000, 0.500000, 0.5000005, 0.500001]
    result = _renormalized(values)
    arr = np.array(values)
    p10 = np.percentile(arr, 10, method="linear")
    p90 = np.percentile(arr, 90, method="linear")
    assert p90 != p10  # 完全一致ではない(このテストの前提)
    assert np.isclose(p90, p10)  # ただしnp.iscloseの既定許容誤差では縮退と誤判定されうる差
    expected = np.clip((arr - p10) / (p90 - p10), 0.0, 1.0)
    assert result == pytest.approx(list(expected))
    assert result != pytest.approx([0.5] * len(values))  # 縮退扱い(全件0.5)になっていない


def test_renormalize_open_rms_empty_list_returns_empty():
    assert _renormalized([]) == []


def test_renormalize_open_rms_is_invariant_to_uniform_gain():
    # パーセンタイル比の相対計算であるため、入力ゲイン(一様なスケール)に不変。
    values = [0.10, 0.35, 0.62, 0.77, 0.91]
    scaled = [v * 0.4 for v in values]  # 一様なゲイン(同じ相対分布・絶対値だけ異なる)
    assert _renormalized(values) == pytest.approx(
        _renormalized(scaled))


def test_confirm_mouth_events_output_is_invariant_to_uniform_audio_gain():
    # 上記の数値レベルの不変性が、実際の音声振幅からRMSを算出する経路(vocal_analysis.rms.
    # compute_rms)を通しても保たれることを、同一波形を異なる振幅でスケールした2入力で確認する。
    #
    # 6段階の振幅包絡(捨てフロア・下アンカー・対象3モーラ・上アンカー)を使う。曲全体パーセン
    # タイル正規化の性質上、複数モーラを均等な振幅包絡で並べると最も静かなモーラが必ずp10
    # percentileの境界にちょうど乗って厳密に0.0へ丸められる(無音しきい値以下になり無音判定に
    # 落ちることもある)。捨てフロアを対象より一段静かな区間として最下段に追加すると、捨てフロア
    # 自身が無音判定に落ちて開き量決定の母集団(母音的口形のみ)から除外され、残る5区間(下アンカー・
    # 対象3モーラ・上アンカー)のうち下アンカー・上アンカーがその母集団内の新たな最小・最大として
    # 正規化後0.0・1.0にちょうど張り付き、対象3モーラだけがその間の中間パーセンタイルへ収まる。
    sample_rate = 8000
    seg_dur_sec = 0.4
    n_per_seg = int(seg_dur_sec * sample_rate)
    levels = [0.05, 0.20, 0.40, 0.50, 0.60, 0.95]  # 捨てフロア・下アンカー・対象a・対象i・対象u・上アンカー
    n = n_per_seg * len(levels)
    t = np.arange(n) / sample_rate
    envelope = np.concatenate([np.full(n_per_seg, lv) for lv in levels])
    tone = (np.sin(2 * np.pi * 220 * t) * envelope).astype(np.float32).reshape(-1, 1)

    bounds = [i * seg_dur_sec for i in range(len(levels) + 1)]
    segments = [
        seg("vowel", bounds[0], bounds[1], phoneme="o̞", confidence=0.9),  # 捨てフロア
        seg("vowel", bounds[1], bounds[2], phoneme="ɯ", confidence=0.9),  # 下アンカー
        seg("vowel", bounds[2], bounds[3], phoneme="a", confidence=0.9),
        seg("vowel", bounds[3], bounds[4], phoneme="i", confidence=0.9),
        seg("vowel", bounds[4], bounds[5], phoneme="u", confidence=0.9),
        seg("vowel", bounds[5], bounds[6], phoneme="e̞", confidence=0.9),  # 上アンカー
    ]

    def confirm_at_gain(gain):
        pcm = AudioPcm(samples=tone * gain, sample_rate=sample_rate)
        rms_envelope = _va_rms.compute_rms(pcm)
        return confirm(segments, rms_envelope)

    events_full_gain, diag_full_gain = confirm_at_gain(1.0)
    events_low_gain, diag_low_gain = confirm_at_gain(0.3)

    # start/end/consonant_class/aperture_classも含め全フィールドを比較する。オンセット補正・
    # 無音境界もRMSに依存するため、開き量だけの比較では見逃しうる差を検出する。float32音声
    # サンプルの絶対値スケール差に伴う浮動小数点丸め誤差(1e-6程度)は許容し近似比較する。
    assert len(events_full_gain) == len(events_low_gain)
    for e_full, e_low in zip(events_full_gain, events_low_gain, strict=True):
        assert e_full.shape == e_low.shape
        assert e_full.consonant_class == e_low.consonant_class
        assert e_full.aperture_class == e_low.aperture_class
        assert e_full.start == pytest.approx(e_low.start, abs=1e-3)
        assert e_full.end == pytest.approx(e_low.end, abs=1e-3)
        assert e_full.open_amount == pytest.approx(e_low.open_amount, abs=1e-5)
    assert diag_full_gain == diag_low_gain
    # このテストの前提(縮退した比較になっていないこと)を明示的に確認する: 捨てフロアは無音
    # 判定に落ち、対象3モーラ(events_full_gain[2:5])は開き量レンジの上下限へクランプされない
    # 中間値を持ち、かつ互いに異なる値であること。
    assert events_full_gain[0].shape == MouthShape.SILENCE
    target_amounts = [e.open_amount for e in events_full_gain[2:5]]
    assert all(0.30 < a < 0.75 for a in target_amounts)
    assert len(set(target_amounts)) == len(target_amounts)


def test_zero_mora_song_does_not_crash_renormalization():
    # 母音・撥音「ん」が1件も無い曲(gapのみ)では、再正規化処理自体を行わない。
    # 空リストを渡してもクラッシュしないことを確認する。
    segments = [seg("gap", 0.0, 0.5)]
    rms = flat_rms(0.5, 0.01)
    mouth_events, _diag = confirm(segments, rms)
    assert len(mouth_events) == 1
    assert mouth_events[0].shape == MouthShape.SILENCE
    assert mouth_events[0].open_amount == 0.0


def test_degenerate_uniform_morae_all_get_midpoint_open_amount():
    # 全モーラの代表RMSが完全に同一な曲(縮退)では、再正規化により全モーラの開き量が一律に
    # 中間値(0.5をintensity-curveで写像した値)になる(生RMSの値そのもの——ここでは0.73——
    # には依存しない)。
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="a", confidence=0.9),
        seg("vowel", 0.2, 0.4, phoneme="i", confidence=0.9),
        seg("vowel", 0.4, 0.6, phoneme="ɯ", confidence=0.9),
    ]
    rms = flat_rms(0.6, 0.73)
    mouth_events, _diag = confirm(
        segments, rms, open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)
    assert [e.open_amount for e in mouth_events] == pytest.approx([0.5, 0.5, 0.5])


def test_open_amount_uses_whole_song_percentiles_not_a_chunk_subset():
    # 長尺分割時、confirm_mouth_eventsにはチャンク境界をまたいだ結合後の全曲セグメント・
    # RMSが1回だけ渡される。この再正規化は渡された全モーラ集合のp10/p90を使うため、
    # チャンク単位で個別正規化した場合とは異なる値になることを、境界前後で異なる母音の
    # 6モーラ構成で検証する。
    # 各モーラの区間長は長時間モーラのサブウィンドウ分割閾値(1.0秒)未満(0.9秒)に保つ。
    mora_dur = 0.9
    segments = [
        seg("vowel", 0.0 * mora_dur, 1.0 * mora_dur, phoneme="ɯ", confidence=0.9),
        seg("vowel", 1.0 * mora_dur, 2.0 * mora_dur, phoneme="e̞", confidence=0.9),
        seg("vowel", 2.0 * mora_dur, 3.0 * mora_dur, phoneme="a", confidence=0.9),   # 境界直前(第1チャンク相当)
        seg("vowel", 3.0 * mora_dur, 4.0 * mora_dur, phoneme="i", confidence=0.9),   # 境界直後(第2チャンク相当)
        seg("vowel", 4.0 * mora_dur, 5.0 * mora_dur, phoneme="o̞", confidence=0.9),
        seg("vowel", 5.0 * mora_dur, 6.0 * mora_dur, phoneme="ɯ", confidence=0.9),
    ]
    raw_values = [0.10, 0.30, 0.50, 0.60, 0.80, 0.99]
    hop = 0.010
    total = 6.0 * mora_dur
    n = round(total / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [raw_values[min(int(t // mora_dur), 5)] for t in times]
    rms = rms_env(times, values)

    mouth_events, _diag = confirm(
        segments, rms, open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)

    whole_song_arr = np.array(raw_values)
    p10 = np.percentile(whole_song_arr, 10, method="linear")
    p90 = np.percentile(whole_song_arr, 90, method="linear")
    expected_boundary_next = np.clip((0.60 - p10) / (p90 - p10), 0.0, 1.0)

    # チャンク単位(第2チャンク=[0.60, 0.80, 0.99]の3モーラ)だけで個別正規化した場合の値。
    chunk2_arr = np.array([0.60, 0.80, 0.99])
    p10_chunk = np.percentile(chunk2_arr, 10, method="linear")
    p90_chunk = np.percentile(chunk2_arr, 90, method="linear")
    per_chunk_boundary_next = np.clip((0.60 - p10_chunk) / (p90_chunk - p10_chunk), 0.0, 1.0)

    boundary_next_event = mouth_events[3]  # 境界直後モーラ(i)
    assert boundary_next_event.open_amount == pytest.approx(float(expected_boundary_next), abs=1e-6)
    assert boundary_next_event.open_amount != pytest.approx(float(per_chunk_boundary_next), abs=1e-3)


def test_silence_classification_unaffected_by_other_loud_morae_context():
    # 母音区間の無音補正は元のRMSのまま使い、再正規化の対象にしない。対象モーラが1件だけ
    # (曲全体のモーラ集合も1件で縮退し常に0.5になる)でも、他に大声のモーラが同居しても、
    # 無音しきい値以下の対象モーラは常にSILENCEになる(縮退時の0.5を無音判定へ誤って
    # 使う実装だと、単独モーラの場合に無音判定が崩れて検出できる)。
    def build(with_other_mora):
        base = 0.3 if with_other_mora else 0.0
        segments = []
        if with_other_mora:
            segments.append(seg("vowel", 0.0, base, phoneme="i", confidence=0.9))  # 大声の他モーラ
        segments.append(seg("vowel", base, base + 0.3, phoneme="a", confidence=0.95))  # 無音しきい値以下(判定対象)
        hop = 0.010
        n = round((base + 0.3) / hop) + 1
        times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
        values = [0.9 if t < base else 0.01 for t in times]
        rms = rms_env(times, values)
        mouth_events, _diag = confirm(segments, rms)
        return mouth_events[-1].shape

    assert build(with_other_mora=False) == MouthShape.SILENCE
    assert build(with_other_mora=True) == MouthShape.SILENCE


def test_weak_vowel_classification_unaffected_by_other_loud_morae_context():
    # 低信頼/無声の母音判定は元のRMSのまま使い、再正規化の対象にしない。対象モーラが1件だけ
    # (曲全体のモーラ集合も1件で縮退し常に0.5になる)でも、他に大声のモーラが同居しても、
    # 弱判定(0.5倍スケール)の適用有無・スケール量は変わらない(縮退時の0.5を弱判定の
    # RMS閾値判定へ誤って使う実装だと、単独モーラの場合に弱判定が崩れて検出できる)。
    def build(confidence, with_other_mora):
        base = 0.3 if with_other_mora else 0.0
        segments = []
        if with_other_mora:
            segments.append(seg("vowel", 0.0, base, phoneme="i", confidence=0.9))  # 大声の他モーラ
        segments.append(seg("vowel", base, base + 0.3, phoneme="a", confidence=confidence))
        hop = 0.010
        n = round((base + 0.3) / hop) + 1
        times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
        values = [0.9 if t < base else 0.25 for t in times]  # 対象は弱判定のRMS閾値(0.3)未満
        rms = rms_env(times, values)
        mouth_events, _diag = confirm(segments, rms)
        return next(e for e in mouth_events if e.shape == MouthShape.A)

    for with_other_mora in (False, True):
        weak_event = build(confidence=0.2, with_other_mora=with_other_mora)  # 低信頼→弱判定
        strong_event = build(confidence=0.9, with_other_mora=with_other_mora)  # 高信頼→弱判定にならない
        assert weak_event.open_amount == pytest.approx(strong_event.open_amount * 0.5)


def test_low_dynamics_suppression_unaffected_by_other_morae_context():
    # 低ダイナミクス曲での無音化抑制は元のdynamic_range_dbのまま使い、再正規化の対象にしない。
    # 他モーラの有無で抑制の発動有無が変わらないことを、同一の対象区間を他モーラあり/なしの
    # 両方で確認して検証する。
    def build(with_other_mora):
        base = 0.2 if with_other_mora else 0.0
        segments = []
        if with_other_mora:
            segments.append(seg("vowel", 0.0, base, phoneme="i", confidence=0.9))  # 追加の他モーラ
        # gap直前の母音(先頭gapによる強制無音と混同しないよう、他モーラの有無に関わらず常に置く)。
        segments.append(seg("vowel", base, base + 0.2, phoneme="a", confidence=0.9))
        segments.append(seg("gap", base + 0.2, base + 0.4))
        segments.append(seg("vowel", base + 0.4, base + 0.6, phoneme="a", confidence=0.9))
        total = base + 0.6
        rms = flat_rms(total, 0.5, dynamic_range_db=5.0)  # 低ダイナミクス(閾値12dB未満)
        idx_lo = int((base + 0.2) / 0.010)
        idx_hi = int((base + 0.4) / 0.010)
        for i in range(idx_lo, idx_hi + 1):
            rms.values[i] = 0.01  # 低RMSでも低ダイナミクスなら無音化を抑制
        mouth_events, _diag = confirm(segments, rms)
        return [e.shape for e in mouth_events]

    without_other_mora = build(with_other_mora=False)
    with_other_mora = build(with_other_mora=True)
    # gapが継続扱いされ前後の同母音(A)と連結する(SILENCEにならない)。他モーラを足しても
    # 対象部分の判定(末尾)は変わらない。
    assert without_other_mora == [MouthShape.A]
    assert with_other_mora[-len(without_other_mora):] == without_other_mora


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
    for prev, nxt in zip(mouth_events, mouth_events[1:], strict=False):
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


# --- 長時間モーラのサブウィンドウ分割 -----------------------------------------
#
# 区間長が閾値(初期値1.0秒)以上のモーラは、複数の等時間幅サブウィンドウへ分割され、
# 同じ母音的口形を持つ、時間順・隙間なく連続する複数のMouthEventとしてlipsyncへ渡される。
# confirm_mouth_eventsの3件目の戻り値(母音的口形ユニットごとの生成MouthEvent数の列)を
# 直接検証するテストは、confirmヘルパー(2件目までしか返さない)でなく
# events.confirm_mouth_events を直接呼ぶ。


@pytest.mark.parametrize("duration_sec,expected_count", [
    (0.3, 2),   # 0.3/0.3=1.0 -> 四捨五入1 -> 最低2へ引き上げ
    (0.6, 2),   # 0.6/0.3=2.0 -> 2
    (0.9, 3),   # 0.9/0.3=3.0 -> 3
    (1.0, 3),   # 1.0/0.3=3.333... -> 四捨五入3
    (1.05, 4),  # 1.05/0.3=3.5 -> 四捨五入(0.5は切り上げ)で4
    (1.5, 5),   # 1.5/0.3=5.0 -> 5
])
def test_subwindow_count_rounds_and_enforces_minimum_two(duration_sec, expected_count):
    assert events._subwindow_count(duration_sec) == expected_count


def test_split_into_subwindows_are_contiguous_equal_width_and_cover_range():
    start, end = 10.0, 11.5  # duration=1.5, count=5
    bounds = events._split_into_subwindows(start, end)
    assert len(bounds) == 5
    assert bounds[0][0] == pytest.approx(start)
    assert bounds[-1][1] == pytest.approx(end)
    width = (end - start) / 5
    for sub_start, sub_end in bounds:
        assert sub_end - sub_start == pytest.approx(width)
    for (_, e1), (s2, _) in zip(bounds, bounds[1:], strict=False):
        assert e1 == pytest.approx(s2)  # 隙間なく連続


def test_mora_below_threshold_remains_single_event():
    # 区間長が閾値(1.0秒)未満のモーラは分割されず、1つのMouthEventのままである。
    segments = [seg("vowel", 0.0, 0.99, phoneme="a", confidence=0.9)]
    rms = flat_rms(0.99, 0.8)
    mouth_events, _diag = confirm(segments, rms)
    assert len(mouth_events) == 1


def test_long_mora_splits_into_contiguous_events_tracking_rms_changes():
    # 1.5秒の長いモーラ(分割数5)内でRMSを5段階に単調上昇させ、生成される各サブウィンドウの
    # 開き量がRMSの上昇に追従して単調に増加することを検証する。
    anchor_dur = 0.2
    mora_dur = 1.5
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("vowel", lo_end, target_end, phoneme="e̞", confidence=0.9),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    step_values = [0.20, 0.35, 0.50, 0.65, 0.80]
    step_dur = mora_dur / 5
    values = []
    for t in times:
        if t < lo_end:
            values.append(_ANCHOR_LO_RMS)
        elif t < target_end:
            step_idx = min(4, int((t - lo_end) / step_dur))
            values.append(step_values[step_idx])
        else:
            values.append(_ANCHOR_HI_RMS)
    rms = rms_env(times, values)
    # open_lo=0.0・open_hi=1.0・open_max=1.0・intensity_curve=1.0 でRMS→開き量の写像を恒等に近づけ、
    # 既定の開き量レンジ(0.30〜0.75)へのクランプで上位の値が飽和し重複するのを避ける
    # (5段階のRMSがクランプを経ても互いに異なる値のまま保たれることを保証するため)。
    kw = dict(_DEFAULT_KW, open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)
    mouth_events, _diag, group_sizes = events.confirm_mouth_events(segments, rms, **kw)
    e_events = [e for e in mouth_events if e.shape == MouthShape.E]
    assert len(e_events) == 5
    assert group_sizes == [1, 5, 1]  # 下アンカー(1)・対象(5分割)・上アンカー(1)
    assert e_events[0].start == pytest.approx(lo_end * FRAME_RATE)
    assert e_events[-1].end == pytest.approx(target_end * FRAME_RATE)
    for prev, nxt in zip(e_events, e_events[1:], strict=False):
        assert prev.end == pytest.approx(nxt.start)  # 隙間なく連続
    amounts = [e.open_amount for e in e_events]
    assert amounts == sorted(amounts)
    assert len(set(amounts)) == 5  # RMSが高いサブウィンドウほど開き量が大きい(単調・非退化)


def test_mora_at_exactly_threshold_is_split():
    # 区間長がちょうど閾値(1.0秒)のモーラも分割対象になる(1.0/0.3=3.333... -> 四捨五入3分割)。
    anchor_dur = 0.2
    mora_dur = 1.0
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("vowel", lo_end, target_end, phoneme="a", confidence=0.9),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    rms = flat_rms(hi_end, 0.6)
    mouth_events, _diag, group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 3
    assert group_sizes == [1, 3, 1]


def test_split_subwindow_rms_uses_middle_60_percent_not_whole_subwindow_average():
    # 各サブウィンドウの代表RMSは、そのサブウィンドウ区間の中央60%窓平均であり、
    # サブウィンドウ全体の単純平均ではない。3番目のサブウィンドウ(内部、先頭でも末尾でもない)
    # だけ両端(トランジェント相当)を低く・中央を高くしたパターンにし、他のサブウィンドウは
    # 一律の高値(center_value)にする。対象を先頭サブウィンドウにしないのは、母音先頭の
    # オンセット補正がモーラの開始時刻を動かしうるため、その影響を受けない内部の
    # サブウィンドウで弁別する。対象サブウィンドウが正しく中央60%窓平均を使えば、その代表RMSは
    # center_valueのみになり、他のサブウィンドウ(一律center_value)と一致する。誤ってサブ
    # ウィンドウ全体の単純平均を使う実装では、対象の代表RMSが両端の低値に引きずられて明確に
    # 低くなり、この一致が崩れる。
    anchor_dur = 0.2
    mora_dur = 1.2  # 4分割、各サブウィンドウ長0.3秒
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("vowel", lo_end, target_end, phoneme="a", confidence=0.9),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    sub_dur = mora_dur / 4  # 0.3秒
    center_value, edge_value = 0.9, 0.0
    target_idx = 2  # 3番目のサブウィンドウ(0始まり)。モーラ先頭から離れオンセット補正の対象外。
    target_start = target_idx * sub_dur
    target_end_sub = (target_idx + 1) * sub_dur
    target_center_lo = target_start + 0.2 * sub_dur
    target_center_hi = target_start + 0.8 * sub_dur  # 対象サブウィンドウの中央60%窓

    values = []
    for t in times:
        if t < lo_end:
            values.append(_ANCHOR_LO_RMS)
        elif t < target_end:
            offset = t - lo_end
            if target_start <= offset < target_end_sub:
                is_center = target_center_lo <= offset < target_center_hi
                values.append(center_value if is_center else edge_value)
            else:
                values.append(center_value)  # 対象以外のサブウィンドウは全区間center_value
        else:
            values.append(_ANCHOR_HI_RMS)
    rms = rms_env(times, values)
    kw = dict(_DEFAULT_KW, open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)
    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **kw)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 4
    for e in a_events:
        assert e.open_amount == pytest.approx(a_events[0].open_amount, abs=1e-6)


def test_split_head_only_keeps_consonant_and_aperture_class_others_are_none():
    # 先行子音(t: ConsonantClass=NEUTRAL, ApertureClass=FIRM_CLOSURE)を持つ長いモーラを分割し、
    # 先頭サブウィンドウだけが元の先頭子音種別・開口減衰種別を持ち、2番目以降はNONEになる
    # ことを検証する。
    anchor_dur = 0.2
    mora_dur = 1.2  # 4分割
    lo_end = anchor_dur
    cons_end = lo_end + 0.05
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("consonant", lo_end, cons_end, phoneme="t"),
        seg("vowel", cons_end, target_end, phoneme="a", confidence=0.9),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [
        _ANCHOR_LO_RMS if t < lo_end else (0.6 if t < target_end else _ANCHOR_HI_RMS)
        for t in times
    ]
    rms = rms_env(times, values)
    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 4
    assert a_events[0].consonant_class == ConsonantClass.NEUTRAL
    assert a_events[0].aperture_class == ApertureClass.FIRM_CLOSURE
    for e in a_events[1:]:
        assert e.consonant_class == ConsonantClass.NONE
        assert e.aperture_class == ApertureClass.NONE


def test_split_preserves_none_head_classes_when_original_has_no_preceding_consonant():
    # 元のユニットの先頭子音種別・開口減衰種別が元々NONEの場合、全サブウィンドウがNONEの
    # ままであることを確認する。
    anchor_dur = 0.2
    mora_dur = 1.2
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("vowel", lo_end, target_end, phoneme="a", confidence=0.9),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    values = [
        _ANCHOR_LO_RMS if t < lo_end else (0.6 if t < target_end else _ANCHOR_HI_RMS)
        for t in times
    ]
    rms = rms_env(times, values)
    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 4
    for e in a_events:
        assert e.consonant_class == ConsonantClass.NONE
        assert e.aperture_class == ApertureClass.NONE


def test_split_weak_vowel_scaling_applies_uniformly_to_all_subwindows():
    # 低信頼/無声で弱判定となった長いモーラが分割された場合、全サブウィンドウの開き量が
    # 弱判定なしの場合のちょうど0.5倍になることを検証する(弱判定はモーラ全体の代表RMSと
    # 信頼度から一度だけ行い、サブウィンドウごとにやり直さない)。
    anchor_dur = 0.2
    mora_dur = 1.2  # 4分割
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur

    def build(confidence):
        segments = [
            seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
            seg("vowel", lo_end, target_end, phoneme="a", confidence=confidence),
            seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
        ]
        hop = 0.010
        n = round(hi_end / hop) + 1
        times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
        half = mora_dur / 2
        values = []
        for t in times:
            if t < lo_end:
                values.append(_ANCHOR_LO_RMS)
            elif t < target_end:
                offset = t - lo_end
                values.append(0.25 if offset < half else 0.28)  # 弱判定RMS閾値(0.3)未満に収める
            else:
                values.append(_ANCHOR_HI_RMS)
        rms = rms_env(times, values)
        mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
        return [e for e in mouth_events if e.shape == MouthShape.A]

    weak_events = build(confidence=0.2)  # 低信頼 -> モーラ全体の代表RMSも弱判定閾値未満 -> 弱判定
    strong_events = build(confidence=0.9)  # 高信頼 -> 弱判定にならない
    assert len(weak_events) == len(strong_events) == 4
    for w, s in zip(weak_events, strong_events, strict=True):
        assert w.open_amount == pytest.approx(s.open_amount * 0.5)


def test_split_does_not_affect_population_percentiles_for_other_morae():
    # 長いモーラの区間長だけを変える(分割数が変わる)2パターンで、その区間のRMSサンプルを
    # 両パターンで同一の単一定数にする(区間長を変えても中央60%窓・区間全体窓いずれの平均も
    # 変わらないため、モーラ代表RMSは両パターンで完全一致する)。比較対象の短いモーラ2件は、
    # 母集団の最小値・最大値(下限アンカー・上限アンカー)ではなく中間パーセンタイル領域に
    # 来る値にする(比較対象が母集団の外側にありパーセンタイル正規化でクリップされる値だと、
    # 母集団の内訳が変わっても常に同じクリップ値になり、サブウィンドウ混入という不具合を
    # 検出できないため)。この条件のもとで、比較対象の開き量が両パターンで完全一致することを
    # 確認する(サブウィンドウの代表RMSがパーセンタイル母集団に混入していないことの回帰)。
    def build(long_mora_dur):
        segments = [
            seg("vowel", 0.0, 0.2, phoneme="ɯ", confidence=0.9),  # 下限アンカー
            seg("vowel", 0.2, 0.4, phoneme="i", confidence=0.9),  # 比較対象1(中間値)
            seg("vowel", 0.4, 0.4 + long_mora_dur, phoneme="e̞", confidence=0.9),  # 長いモーラ
            seg("vowel", 0.4 + long_mora_dur, 0.6 + long_mora_dur, phoneme="a", confidence=0.9),  # 比較対象2
            seg("vowel", 0.6 + long_mora_dur, 0.8 + long_mora_dur, phoneme="o̞", confidence=0.9),  # 上限アンカー
        ]
        hop = 0.010
        total = 0.8 + long_mora_dur
        n = round(total / hop) + 1
        times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
        values = []
        for t in times:
            if t < 0.2:
                values.append(0.10)
            elif t < 0.4:
                values.append(0.40)
            elif t < 0.4 + long_mora_dur:
                values.append(0.60)  # 長いモーラの区間は両パターンで同一の定数
            elif t < 0.6 + long_mora_dur:
                values.append(0.50)
            else:
                values.append(0.99)
        rms = rms_env(times, values)
        mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
        target1 = next(e for e in mouth_events if e.shape == MouthShape.I)
        target2 = next(e for e in mouth_events if e.shape == MouthShape.A)
        return target1.open_amount, target2.open_amount

    short_split_result = build(1.2)  # 4分割
    long_split_result = build(1.8)  # 6分割
    assert short_split_result == pytest.approx(long_split_result)
    # 比較対象がクランプ(0.30または0.75への飽和)された値になっていないことを確認し、
    # このテストの弁別力が保たれていることを担保する。
    assert 0.30 < short_split_result[0] < 0.75
    assert 0.30 < short_split_result[1] < 0.75


def test_zero_mora_song_group_sizes_is_empty():
    # 母音的口形のユニットが1件も無い曲では、3件目の戻り値(分割数列)も空になる。
    segments = [seg("gap", 0.0, 0.5)]
    rms = flat_rms(0.5, 0.01)
    mouth_events, _diag, group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    assert len(mouth_events) == 1
    assert group_sizes == []


def test_split_subwindows_get_same_open_amount_when_population_is_degenerate():
    # 曲全体のモーラ代表RMSの母集団が縮退(p10==p90)している場合、分割される長時間モーラの
    # サブウィンドウ間で代表RMSに意図的な差を持たせても、全サブウィンドウの開き量が互いに
    # 同一の値になることを検証する(縮退判定は分割前のモーラ代表RMSの母集団に対して行われ、
    # サブウィンドウの生RMSはこの母集団に含まれないため)。
    mora_dur = 1.2  # 4分割
    other_value = 0.5
    delta = 0.3
    segments = [
        seg("vowel", 0.0, 0.2, phoneme="ɯ", confidence=0.9),  # 他のモーラ(代表RMS=other_value)
        seg("vowel", 0.2, 0.2 + mora_dur, phoneme="a", confidence=0.9),  # 長いモーラ(分割対象)
    ]
    hop = 0.010
    total = 0.2 + mora_dur
    n = round(total / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    half = mora_dur / 2
    values = []
    for t in times:
        if t < 0.2:
            values.append(other_value)
        else:
            offset = t - 0.2
            # 対称な値のペア(前半・後半)にすることで、長いモーラの中央60%窓平均を
            # other_valueへちょうど一致させ、曲全体の母集団(2件)を縮退させる。
            values.append(other_value - delta if offset < half else other_value + delta)
    rms = rms_env(times, values)
    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 4
    amounts = [e.open_amount for e in a_events]
    assert amounts == pytest.approx([amounts[0]] * 4)


def test_n_mora_splits_when_long():
    # 長く伸ばす撥音「ん」も母音と同様に分割対象になることを検証する。
    anchor_dur = 0.2
    mora_dur = 1.2  # 4分割
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("consonant", lo_end, target_end, phoneme="ɴ"),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    step_values = [0.2, 0.4, 0.6, 0.8]
    step_dur = mora_dur / 4
    values = []
    for t in times:
        if t < lo_end:
            values.append(_ANCHOR_LO_RMS)
        elif t < target_end:
            idx = min(3, int((t - lo_end) / step_dur))
            values.append(step_values[idx])
        else:
            values.append(_ANCHOR_HI_RMS)
    rms = rms_env(times, values)
    # RMS追従テストと同様に、既定の開き量レンジ(0.30〜0.75)へのクランプで上位の値が
    # 飽和し重複するのを避けるため、写像を恒等に近づけるパラメータを明示する。
    kw = dict(_DEFAULT_KW, open_lo=0.0, open_hi=1.0, open_max=1.0, intensity_curve=1.0)
    mouth_events, _diag, group_sizes = events.confirm_mouth_events(segments, rms, **kw)
    n_events = [e for e in mouth_events if e.shape == MouthShape.N]
    assert len(n_events) == 4
    assert group_sizes == [1, 4, 1]
    amounts = [e.open_amount for e in n_events]
    assert amounts == sorted(amounts)
    assert len(set(amounts)) == 4


# --- 長時間モーラのサブウィンドウ: 連続写像(既存のクランプによる強弱潰れを防ぐ) ----------
#
# 分割された長時間モーラの各サブウィンドウの開き量は、_map_open_amount_continuous による連続
# 写像を使う(分割されないモーラの開き量決定は _map_open_amount によるクランプ式を使う)。

@pytest.mark.parametrize("normalized", [0.0, 0.25, 0.5, 0.75, 1.0])
@pytest.mark.parametrize("open_max,cap_side", [(0.60, "open_max"), (0.90, "open_hi")])
def test_map_open_amount_continuous_follows_formula_without_clamping(normalized, open_max, cap_side):
    open_lo, open_hi, intensity_curve = 0.30, 0.75, 0.6
    cap = min(open_hi, open_max)
    floor = min(open_lo, cap)
    expected = floor + (cap - floor) * (normalized ** intensity_curve)
    actual = events._map_open_amount_continuous(
        normalized, open_lo=open_lo, open_hi=open_hi, open_max=open_max, intensity_curve=intensity_curve,
    )
    assert actual == pytest.approx(expected)
    assert cap == pytest.approx(open_max if cap_side == "open_max" else open_hi)


def test_map_open_amount_continuous_degenerates_to_constant_when_open_max_below_open_lo():
    # 逆単調にならないことの回帰。
    open_lo, open_hi, open_max, intensity_curve = 0.30, 0.75, 0.10, 0.6
    amounts = [
        events._map_open_amount_continuous(
            v, open_lo=open_lo, open_hi=open_hi, open_max=open_max, intensity_curve=intensity_curve,
        )
        for v in (0.0, 0.3, 0.7, 1.0)
    ]
    assert amounts == pytest.approx([open_max] * 4)


def test_map_open_amount_continuous_degenerates_to_constant_when_open_max_equals_open_lo():
    open_lo, open_hi, open_max, intensity_curve = 0.30, 0.75, 0.30, 0.6
    amounts = [
        events._map_open_amount_continuous(
            v, open_lo=open_lo, open_hi=open_hi, open_max=open_max, intensity_curve=intensity_curve,
        )
        for v in (0.0, 0.3, 0.7, 1.0)
    ]
    assert amounts == pytest.approx([open_lo] * 4)


def test_split_subwindow_open_amounts_are_distinct_in_clamp_saturation_range():
    anchor_dur = 0.20
    mora_dur = 1.02  # 目標サブウィンドウ長0.3秒に対し _subwindow_count が3を返す区間長
    lo_end = anchor_dur
    target_end = lo_end + mora_dur
    hi_end = target_end + anchor_dur
    segments = [
        seg("vowel", 0.0, lo_end, phoneme="ɯ", confidence=0.9),
        seg("vowel", lo_end, target_end, phoneme="a", confidence=0.9),
        seg("vowel", target_end, hi_end, phoneme="o̞", confidence=0.9),
    ]
    hop = 0.010
    n = round(hi_end / hop) + 1
    times = [_FRAME_CENTER_OFFSET_SEC + i * hop for i in range(n)]
    sub_dur = mora_dur / 3
    sub_values = [0.73, 0.856, 0.955]
    values = []
    for t in times:
        if t < lo_end:
            values.append(_ANCHOR_LO_RMS)
        elif t < target_end:
            offset = t - lo_end
            idx = min(2, int(offset / sub_dur))
            values.append(sub_values[idx])
        else:
            values.append(_ANCHOR_HI_RMS)
    rms = rms_env(times, values)

    mora_representative = events._mora_rms(rms, lo_end, target_end)
    p_lo, p_hi = events._percentile_bounds([_ANCHOR_LO_RMS, mora_representative, _ANCHOR_HI_RMS])
    normalized_values = [events._normalize_with_bounds(v, p_lo, p_hi) for v in sub_values]

    cap = min(_DEFAULT_KW["open_hi"], _DEFAULT_KW["open_max"])
    saturation_threshold = cap ** (1.0 / _DEFAULT_KW["intensity_curve"])
    assert all(v >= saturation_threshold for v in normalized_values)

    expected_amounts = [
        events._map_open_amount_continuous(
            v, open_lo=_DEFAULT_KW["open_lo"], open_hi=_DEFAULT_KW["open_hi"],
            open_max=_DEFAULT_KW["open_max"], intensity_curve=_DEFAULT_KW["intensity_curve"],
        )
        for v in normalized_values
    ]

    mouth_events, _diag, _group_sizes = events.confirm_mouth_events(segments, rms, **_DEFAULT_KW)
    a_events = [e for e in mouth_events if e.shape == MouthShape.A]
    assert len(a_events) == 3
    actual_amounts = [e.open_amount for e in a_events]
    assert actual_amounts == pytest.approx(expected_amounts)
    assert actual_amounts == sorted(actual_amounts)
    assert len(set(actual_amounts)) == 3
