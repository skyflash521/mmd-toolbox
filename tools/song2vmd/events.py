"""song2vmd 口形イベント確定(入口)。

vocal_analysis の音素セグメント列(母音/子音/gap)と相対正規化RMSから、`lipsync` へ渡す口形イベント列
(`MouthEvent`)と各モーラの開き量を確定する。適用順は次の6段階に従う:
(1) IPA写像・両唇閉鎖判定・撥音「ん」判定、(2) gap解決・母音区間の無音補正、(3) 同母音連結、
(4) 母音境界のRMSオンセット補正、(5) 低信頼・無声判定、(6) 先頭子音種別の付与。

音素記号は採用構成(G2P強制アライメント)が実際に出力する記号に基づく:
撥音「ん」は専用記号 ɴ、ま行/ば行/ぱ行頭子音は m/mʲ・b/bʲ・p/pʲ。
"""

from dataclasses import dataclass, replace

import numpy as np

from lipsync import ApertureClass, ConsonantClass, MouthEvent, MouthShape
from vocal_analysis.phonemes import espeak_ipa_to_vowel

FRAME_RATE = 30.0  # 30fps基準。

# ま行・ば行・ぱ行頭子音(採用構成の音素記号)。ɸ(ふ)は唇を丸める子音で
# 閉鎖しないため含めない。
_BILABIAL_PHONEMES = frozenset({"m", "mʲ", "b", "bʲ", "p", "pʲ"})

# 撥音「ん」専用の音素記号(採用構成ではG2Pの N に対応する ɴ)。頭子音の鼻音(n・ɲ・m・mʲ等)とは
# 記号が分かれるため、後続母音の有無を観測しなくても記号だけで一意に判定できる。
_MORAIC_NASAL_PHONEME = "ɴ"

# 先頭子音種別(ConsonantClass)の判定表。
_ROUNDED_PHONEMES = frozenset({"ɸ", "w"})
_SPREAD_PHONEMES = frozenset({"ɕ", "tɕ", "dʑ", "ɲ", "ç"})

# 開口減衰種別(ApertureClass)の判定表。ConsonantClassとは独立な軸で、区切りから
# その母音までの子音列のうち最も強いクラスを付与する(_strongest_aperture_class)。
_FIRM_CLOSURE_PHONEMES = frozenset({"t", "d", "n", "ts", "ɲ"})
_NARROW_CHANNEL_PHONEMES = frozenset({"s", "z", "ɕ", "tɕ", "dʑ", "ç", "j"})
_SLIGHT_CLOSURE_PHONEMES = frozenset({"k", "ɡ", "ɾ", "kʲ", "ɡʲ"})
_APERTURE_RANK = {
    ApertureClass.NONE: 0,
    ApertureClass.SLIGHT_CLOSURE: 1,
    ApertureClass.NARROW_CHANNEL: 2,
    ApertureClass.FIRM_CLOSURE: 3,
}

_VOWEL_SHAPES = {"a": MouthShape.A, "i": MouthShape.I, "u": MouthShape.U, "e": MouthShape.E, "o": MouthShape.O}

_LOW_DYNAMICS_THRESHOLD_DB = 12.0  # 低ダイナミクス判定のしきい値(初期値)。
_MORA_CENTER_FRACTION = 0.6  # 代表RMSの中央60%窓の比率(母音核・モーラ全体の二窓共通)。
_ONSET_WINDOW_SEC = 0.06  # 母音境界のRMSオンセット補正窓(初期値)。
_ONSET_SLOPE_PER_10MS = 0.15  # オンセット判定の傾きしきい値(初期値)。
_ONSET_SMOOTH_WINDOW_SEC = 0.03  # オンセット検出前の平滑化窓(初期値)。
_WEAK_CONFIDENCE_THRESHOLD = 0.5  # 低信頼判定のしきい値(初期値)。
_WEAK_RMS_WITH_CONFIDENCE = 0.3
_WEAK_RMS_WITHOUT_CONFIDENCE = 0.2
_WEAK_SCALE = 0.5


def _consonant_class(phoneme):
    """先頭子音種別(ConsonantClass)を音素記号から判定する。"""
    if phoneme is None:
        return ConsonantClass.NONE
    if phoneme in _ROUNDED_PHONEMES:
        return ConsonantClass.ROUNDED
    if phoneme in _SPREAD_PHONEMES or phoneme.endswith("ʲ"):
        return ConsonantClass.SPREAD
    return ConsonantClass.NEUTRAL


def _aperture_class_of_phoneme(phoneme):
    """単一子音音素の開口減衰種別。表に無い子音はNONE。"""
    if phoneme in _FIRM_CLOSURE_PHONEMES:
        return ApertureClass.FIRM_CLOSURE
    if phoneme in _NARROW_CHANNEL_PHONEMES:
        return ApertureClass.NARROW_CHANNEL
    if phoneme in _SLIGHT_CLOSURE_PHONEMES:
        return ApertureClass.SLIGHT_CLOSURE
    return ApertureClass.NONE


def _strongest_aperture_class(consonant_run):
    """区切りからその母音までに連続して現れる子音列のうち最も強いクラス。"""
    strongest = ApertureClass.NONE
    for phoneme in consonant_run:
        cls = _aperture_class_of_phoneme(phoneme)
        if _APERTURE_RANK[cls] > _APERTURE_RANK[strongest]:
            strongest = cls
    return strongest


@dataclass
class _Unit:
    """口形イベント確定の中間表現(確定前の口形区間。時刻は秒)。

    start_sec/end_sec は口形イベントとして表示する区間(吸収した子音区間を含みうる)。
    content_start_sec/content_end_sec は「子音区間を除いた」母音核区間(vowel/nのみ意味を持つ。
    未指定なら start_sec/end_sec をそのまま使う)。代表RMSは母音核区間とモーラ区間全体の
    中央60%平均の大きい方(モーラ代表RMS)。
    """

    kind: str  # "vowel" | "bilabial" | "n" | "gap" | "silence"
    start_sec: float
    end_sec: float
    letter: str | None = None  # "vowel" のみ意味を持つ(a/i/u/e/o)
    confidence: float | None = None  # "vowel" のみ意味を持つ
    consonant_ipa: str | None = None  # "vowel" のみ意味を持つ(先頭子音種別判定用)
    # "vowel" のみ意味を持つ(開口減衰種別判定用)。直前の区切り(母音・gap・撥音「ん」・両唇閉鎖・
    # 列先頭)からこの母音までに連続して現れた子音(両唇・撥音を除く)の音素列。
    aperture_ipas: tuple[str, ...] = ()
    content_start_sec: float | None = None
    content_end_sec: float | None = None

    def content_span(self):
        start = self.content_start_sec if self.content_start_sec is not None else self.start_sec
        end = self.content_end_sec if self.content_end_sec is not None else self.end_sec
        return start, end


@dataclass(frozen=True)
class EventDiagnostics:
    """口形イベント確定の診断。"""

    weak_vowels: int  # 低信頼・無声判定で開き量を弱めたモーラ数
    low_dynamics: bool  # 低ダイナミクス抑制が働いたか
    merged_morae: int  # 段階(3)の同母音連結で統合された(vowel/n)モーラ数


def _nearest_index(times, t):
    idx = int(np.searchsorted(times, t))
    if idx <= 0:
        return 0
    if idx >= len(times):
        return len(times) - 1
    before, after = times[idx - 1], times[idx]
    return idx - 1 if (t - before) <= (after - t) else idx


def _rms_window_average(rms, start_sec, end_sec):
    """[start_sec, end_sec) に落ちるRMSサンプルの平均。サンプルが無ければ最近傍1点で代表する。"""
    times, values = rms.times_sec, rms.values
    if len(times) == 0:
        return 0.0
    lo = int(np.searchsorted(times, start_sec, side="left"))
    hi = int(np.searchsorted(times, end_sec, side="left"))
    if hi > lo:
        return float(np.mean(values[lo:hi]))
    return float(values[_nearest_index(times, (start_sec + end_sec) / 2.0)])


def _mora_rms(rms, start_sec, end_sec):
    """区間の中央60%の平均RMS。"""
    span = end_sec - start_sec
    margin = span * (1.0 - _MORA_CENTER_FRACTION) / 2.0
    return _rms_window_average(rms, start_sec + margin, end_sec - margin)


def _mora_representative_rms(rms, unit):
    """モーラ代表RMS(母音核区間と、吸収した先行子音を含むモーラ区間全体の、中央60%平均の
    大きい方)。

    強制アライメントは伸ばして歌う発声の大部分を先行子音トークンへ割り当て、母音核を数十msまで
    狭めることがある。その狭い窓だけで判定すると、発声中のモーラを無音補正で閉口させたり開き量を
    過小にするため、モーラ区間全体の窓とも比べて大きい方を採る。
    """
    content_start, content_end = unit.content_span()
    value = _mora_rms(rms, content_start, content_end)
    if (unit.start_sec, unit.end_sec) != (content_start, content_end):
        value = max(value, _mora_rms(rms, unit.start_sec, unit.end_sec))
    return value


def _classify_phonetic(segments, use_n_morph):
    """段階(1): IPA→5母音写像・両唇閉鎖判定・撥音「ん」判定(音素由来、RMS非依存)。

    独立イベントを作らない子音(両唇閉鎖・撥音「ん」以外)は、次に現れる母音的口形イベントの開始時刻を
    その子音の開始まで前へ寄せることで吸収する(前後母音の協調調音は lipsync が扱う)。
    """
    units = []
    pending_start = None
    last_consonant_ipa = None
    aperture_run = []  # 区切りからの累積子音列(開口減衰種別判定用)
    for seg in segments:
        if seg.type == "vowel":
            start = pending_start if pending_start is not None else seg.start_sec
            letter = espeak_ipa_to_vowel(seg.phoneme)
            if letter is None:
                units.append(_Unit("gap", start, seg.end_sec))
                aperture_run = []
            else:
                # content_start/end_sec は吸収した子音区間を含めない元のセグメント境界。
                units.append(_Unit(
                    "vowel", start, seg.end_sec, letter=letter,
                    confidence=seg.confidence, consonant_ipa=last_consonant_ipa,
                    aperture_ipas=tuple(aperture_run),
                    content_start_sec=seg.start_sec, content_end_sec=seg.end_sec,
                ))
                aperture_run = []
            pending_start = None
            last_consonant_ipa = None
        elif seg.type == "consonant":
            last_consonant_ipa = seg.phoneme
            if seg.phoneme in _BILABIAL_PHONEMES:
                start = pending_start if pending_start is not None else seg.start_sec
                units.append(_Unit("bilabial", start, seg.end_sec))
                pending_start = None
                aperture_run = []  # 両唇閉鎖は区切り
            elif seg.phoneme == _MORAIC_NASAL_PHONEME:
                start = pending_start if pending_start is not None else seg.start_sec
                kind = "n" if use_n_morph else "silence"
                units.append(_Unit(
                    kind, start, seg.end_sec,
                    content_start_sec=seg.start_sec, content_end_sec=seg.end_sec,
                ))
                pending_start = None
                # 撥音「ん」自身は独立イベントとして表示済みなので、次の母音の先頭子音種別には
                # 使わない(先行隣接子音なし=NONE)。撥音自身も区切り。
                last_consonant_ipa = None
                aperture_run = []
            else:
                if pending_start is None:
                    pending_start = seg.start_sec
                aperture_run.append(seg.phoneme)
        else:  # gap
            start = pending_start if pending_start is not None else seg.start_sec
            units.append(_Unit("gap", start, seg.end_sec))
            pending_start = None
            last_consonant_ipa = None
            aperture_run = []
    if pending_start is not None and segments:
        units.append(_Unit("gap", pending_start, segments[-1].end_sec))
    return units


_GAP_CLOSE_RUN_SEC = 0.2  # gap走査で発声終了とみなす、下降側しきい値以下の最小連続長


def _gap_close_time(rms, start_sec, end_sec, silence_on):
    """gap内の正規化RMSを先頭から走査し、発声終了時刻を返す(gap走査)。

    silence_on以下が _GAP_CLOSE_RUN_SEC 以上連続した最初の連続、または gap 終端まで続く連続の
    開始時刻を返す(連続要件はビブラート・トレモロの瞬間的な谷での早期閉口を防ぐ余裕で、終端まで
    達した連続にはその先の発声再開が無いため適用しない)。しきい値以下の連続が無ければ None
    (gap全体で発声が継続)。gap内の最初のフレームから始まる連続は start_sec を返す(そのフレームの
    RMS窓はgap開始時刻を覆っており、gap先頭から無音として全体閉口に倒す。フレーム中心時刻を返すと
    幅十数msの継続断片が生じ、直前母音への併合がモーラ併合の診断値を実体なく水増しするため)。
    """
    times, values = rms.times_sec, rms.values
    lo = int(np.searchsorted(times, start_sec, side="left"))
    hi = int(np.searchsorted(times, end_sec, side="left"))
    run_start_index = None
    for i in range(lo, hi):
        if values[i] <= silence_on:
            if run_start_index is None:
                run_start_index = i
            if times[i] - times[run_start_index] >= _GAP_CLOSE_RUN_SEC:
                break
        else:
            run_start_index = None
    if run_start_index is None:
        return None
    return start_sec if run_start_index == lo else float(times[run_start_index])


def _resolve_silence(units, rms, silence_on, low_dynamics):
    """段階(2): gap解決・母音区間の無音補正。

    gapの無音/継続は、直前に確定した口形が母音的(母音・撥音「ん」)かどうかと開始(下降側)しきい値
    だけで決める。gap全体をひとまとめに判定せず、RMSを先頭から走査して発声が終わった時点でgapを
    分割し、前半は直前の母音的口形の継続・残りは無音にする(一度閉じたgap内では
    再度開かない)。先頭・末尾のgapはRMSに依らず常に無音。
    低ダイナミクス曲では、この音量に基づく無音化(gap無音化・母音区間の無音補正)を抑制する
    (先頭・末尾の閉口は対象外)。
    """
    result = []
    open_kind = None  # None(閉) または (kind, letter) で直前の母音的口形を保持
    n = len(units)
    for i, u in enumerate(units):
        if u.kind == "gap":
            if i == 0 or i == n - 1 or open_kind is None:
                result.append(_Unit("silence", u.start_sec, u.end_sec))
                open_kind = None
                continue
            close_at = None if low_dynamics else _gap_close_time(rms, u.start_sec, u.end_sec, silence_on)
            kind, letter = open_kind
            if close_at is None:
                result.append(_Unit(kind, u.start_sec, u.end_sec, letter=letter))
            elif close_at <= u.start_sec:
                result.append(_Unit("silence", u.start_sec, u.end_sec))
                open_kind = None
            else:
                result.append(_Unit(kind, u.start_sec, close_at, letter=letter))
                result.append(_Unit("silence", close_at, u.end_sec))
                open_kind = None
        elif u.kind == "vowel":
            if not low_dynamics and _mora_representative_rms(rms, u) <= silence_on:
                result.append(_Unit("silence", u.start_sec, u.end_sec))
                open_kind = None
            else:
                result.append(u)
                open_kind = ("vowel", u.letter)
        elif u.kind == "n":
            result.append(u)
            open_kind = ("n", None)
        else:  # "bilabial" | "silence"(--no-n-morphの撥音等)
            result.append(u)
            open_kind = None
    return result


_MORA_KINDS = frozenset({"vowel", "n"})  # merged_morae に数える対象(閉口・無音は含めない)


def _merge_adjacent(units):
    """段階(3): 連続する同一口形区間を1つのイベントへまとめる。

    content_start_sec/content_end_sec(代表RMS算出用の区間)は、先頭ユニットの開始から
    末尾ユニットの終了までへ広げる(同一口形が続く区間はすべて母音的内容とみなす)。

    戻り値は (統合後のユニット列, 統合回数)。統合回数は母音/撥音「ん」区間の統合
    (merged_morae)だけを数え、閉口・無音区間の統合は含めない。
    """
    merged = []
    merged_morae = 0
    for u in units:
        # 母音どうしは、間に子音を挟まない(u.consonant_ipa is None、認識上の分割による観測上の
        # 連続)場合だけ統合する。前後の子音が同一音素であっても(例:「たた」)モーラの区切り
        # (子音の再構音)は実在するため統合しない。
        can_merge = (
            merged and merged[-1].kind == u.kind
            and (u.kind != "vowel" or (merged[-1].letter == u.letter and u.consonant_ipa is None))
        )
        if can_merge:
            prev = merged[-1]
            if u.kind in _MORA_KINDS:
                merged_morae += 1
            prev_content_start, _ = prev.content_span()
            _, u_content_end = u.content_span()
            merged[-1] = replace(
                prev, end_sec=u.end_sec,
                content_start_sec=prev_content_start, content_end_sec=u_content_end,
                aperture_ipas=prev.aperture_ipas + u.aperture_ipas,
            )
        else:
            merged.append(u)
    return merged, merged_morae


def _smoothed_rms(rms):
    """オンセット検出用に約30ms移動平均で平滑化したRMSを返す。"""
    times, values = rms.times_sec, rms.values
    if len(times) < 2:
        return times, values
    hop = float(np.median(np.diff(times)))
    if hop <= 0:
        return times, values
    window_samples = max(1, round(_ONSET_SMOOTH_WINDOW_SEC / hop))
    half_window = max(1, (window_samples - 1) // 2)
    kernel = np.ones(2 * half_window + 1) / (2 * half_window + 1)
    # 端をゼロ埋めする mode="same" は平坦な信号の端に偽の傾きを作るため、端の値を複製してから
    # "valid" 畳み込みで同じ長さに戻す(端で不自然な立ち上がりを作らない)。
    padded = np.pad(values, half_window, mode="edge")
    return times, np.convolve(padded, kernel, mode="valid")


def _find_onset(times, smoothed, center_sec):
    """探索窓内で上昇の傾きがしきい値を超える最初の点を探し、元境界に最も近いものを返す。"""
    if len(times) < 2:
        return None
    lo, hi = center_sec - _ONSET_WINDOW_SEC, center_sec + _ONSET_WINDOW_SEC
    candidates = []
    for i in range(1, len(times)):
        if times[i] < lo or times[i] > hi:
            continue
        dt = times[i] - times[i - 1]
        if dt <= 0:
            continue
        slope = (smoothed[i] - smoothed[i - 1]) / (dt / 0.010)
        if slope > _ONSET_SLOPE_PER_10MS:
            candidates.append(times[i])
    if not candidates:
        return None
    return min(candidates, key=lambda t: abs(t - center_sec))


def _refine_onsets(units, rms):
    """段階(4): 母音境界のRMSオンセット補正。

    見つかった立ち上がり時刻は、直前ユニット自身の開始と当該母音区間の終了の範囲へクランプし、
    負長イベントを作らない(lipsync.types.MouthEvent の隙間なし・非重複契約)。
    """
    times, smoothed = _smoothed_rms(rms)
    result = list(units)
    for i, u in enumerate(result):
        if u.kind != "vowel":
            continue
        onset = _find_onset(times, smoothed, u.start_sec)
        if onset is None:
            continue
        lower_bound = result[i - 1].start_sec if i > 0 else float("-inf")
        onset = min(max(onset, lower_bound), u.end_sec)
        if onset == u.start_sec:
            continue
        if i > 0:
            prev = result[i - 1]
            _, prev_content_end = prev.content_span()
            result[i - 1] = replace(prev, end_sec=onset, content_end_sec=min(prev_content_end, onset))
        result[i] = replace(u, start_sec=onset, content_start_sec=onset)
    return result


def _map_open_amount(rms_value, *, open_lo, open_hi, open_max, intensity_curve):
    """RMS→開き量の写像(累乗則・スタイルレンジ・open_maxへのクランプ)。"""
    raw = rms_value ** intensity_curve
    clamped = min(max(raw, open_lo), open_hi)
    return min(clamped, open_max)


def _is_weak_vowel(confidence, rms_value):
    """段階(5): 低信頼・無声母音判定。"""
    if confidence is not None:
        return confidence < _WEAK_CONFIDENCE_THRESHOLD and rms_value < _WEAK_RMS_WITH_CONFIDENCE
    return rms_value < _WEAK_RMS_WITHOUT_CONFIDENCE


_OPEN_RENORM_P_LO = 10.0
_OPEN_RENORM_P_HI = 90.0


def _renormalize_open_rms(values):
    """開き量決定にだけ使う、曲全体モーラ代表RMS集合のパーセンタイル線形正規化。

    p10→0・p90→1、範囲外はクリップ。p90とp10が完全一致する場合(縮退。モーラ1件・全モーラ
    同値を含む)は、無音側でなく開閉の中間値0.5へ一律に倒す。
    """
    if not values:
        return []
    arr = np.array(values, dtype=float)
    p_lo = np.percentile(arr, _OPEN_RENORM_P_LO, method="linear")
    p_hi = np.percentile(arr, _OPEN_RENORM_P_HI, method="linear")
    if p_hi == p_lo:
        return [0.5] * len(values)
    return list(np.clip((arr - p_lo) / (p_hi - p_lo), 0.0, 1.0))


def confirm_mouth_events(segments, rms, *, open_lo, open_hi, open_max, intensity_curve, silence_on,
                          use_n_morph=True):
    """音素セグメント列とRMSから口形イベント列(MouthEvent)と開き量を確定する。

    段階(6)の先頭子音種別付与とフレーム変換(30fps)は本関数内で行い、`lipsync` へ渡す最終形を返す。
    """
    if not segments:
        return [], EventDiagnostics(weak_vowels=0, low_dynamics=False, merged_morae=0)

    low_dynamics = rms.dynamic_range_db < _LOW_DYNAMICS_THRESHOLD_DB
    units = _classify_phonetic(segments, use_n_morph)
    units = _resolve_silence(units, rms, silence_on, low_dynamics)
    units, merged_morae_1 = _merge_adjacent(units)
    units = _refine_onsets(units, rms)
    units, merged_morae_2 = _merge_adjacent(units)  # オンセット補正で隣接区間が同一境界に揃うケースを再連結する
    merged_morae = merged_morae_1 + merged_morae_2

    mora_rms_raw = [
        _mora_representative_rms(rms, u) if u.kind in ("vowel", "n") else None for u in units
    ]
    open_rms_iter = iter(_renormalize_open_rms([v for v in mora_rms_raw if v is not None]))

    mouth_events = []
    weak_vowels = 0
    for u, mora_rms in zip(units, mora_rms_raw):
        if u.kind in ("vowel", "n"):
            open_amount = _map_open_amount(
                next(open_rms_iter), open_lo=open_lo, open_hi=open_hi, open_max=open_max,
                intensity_curve=intensity_curve,
            )
            if u.kind == "vowel":
                if _is_weak_vowel(u.confidence, mora_rms):
                    open_amount *= _WEAK_SCALE
                    weak_vowels += 1
                shape = _VOWEL_SHAPES[u.letter]
                consonant_class = _consonant_class(u.consonant_ipa)
                aperture_class = _strongest_aperture_class(u.aperture_ipas)
            else:
                shape = MouthShape.N
                consonant_class = ConsonantClass.NONE
                aperture_class = ApertureClass.NONE
        else:  # "bilabial" | "silence"
            shape = MouthShape.BILABIAL if u.kind == "bilabial" else MouthShape.SILENCE
            open_amount = 0.0
            consonant_class = ConsonantClass.NONE
            aperture_class = ApertureClass.NONE
        mouth_events.append(MouthEvent(
            shape=shape, start=u.start_sec * FRAME_RATE, end=u.end_sec * FRAME_RATE,
            open_amount=open_amount, consonant_class=consonant_class, aperture_class=aperture_class,
        ))

    return mouth_events, EventDiagnostics(
        weak_vowels=weak_vowels, low_dynamics=low_dynamics, merged_morae=merged_morae)
