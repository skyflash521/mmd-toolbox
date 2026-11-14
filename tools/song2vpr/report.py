def run_fields(*, output, tempo, duration_sec, separated, backends,
               split, annotation, build) -> dict:
    return {
        "output": output,
        "notes": build.note_count,
        "tempo_bpm": tempo.bpm,
        "tempo_source": tempo.tempo_source,
        "time_signature": f"{tempo.numerator}/{tempo.denominator}",
        "time_signature_source": tempo.time_signature_source,
        "resolution": tempo.resolution,
        "duration_sec": duration_sec,
        "separated": separated,
        "backends": backends,
        "vowel_undetermined_notes": annotation.undetermined_vowel_notes,
        "moraic_nasal_notes": annotation.moraic_nasal_notes,
        "fallback_lyric_notes": annotation.notes_beyond_morae,
        "dropped_morae": annotation.discarded_morae,
        "short_notes": build.short_notes,
        "suppressed_notes": split.suppressed_notes + annotation.suppressed_notes,
        "pitch_clamped_notes": build.pitch_clamped_notes,
        "quantized_merged_notes": build.quantized_merged_notes,
        "quantized_stretched_notes": build.quantized_stretched_notes,
        "no_phoneme_notes": annotation.no_phoneme_notes,
    }


def inspect_fields(*, sample_rate, channels, **run) -> dict:
    fields = run_fields(output=None, **run)
    fields.update(input_kind="audio", sample_rate=sample_rate, channels=channels)
    return fields


_HUMAN_BACKEND_LABELS = {
    "separator": "ボーカル分離",
    "recognizer": "内容認識モデル",
    "recognizer_revision": "内容認識モデルのリビジョン",
    "forced_aligner": "強制アライメント",
    "english_katakana_method": "英語カタカナ化",
}

_HUMAN_TEMPO_SOURCE_LABELS = {"option": "指定値", "estimated": "推定値", "default": "仮置き"}
_HUMAN_TIME_SIGNATURE_SOURCE_LABELS = {"option": "指定値", "default": "既定値"}

_HUMAN_FIELD_LABELS = (
    ("duration_sec", "入力の尺(秒)"),
    ("notes", "音符数"),
    ("moraic_nasal_notes", "撥音「ん」を入れた音符"),
    ("vowel_undetermined_notes", "母音が得られず「あ」を入れた音符"),
    ("no_phoneme_notes", "音素列が空の音符"),
    ("short_notes", "配り直しでも短いまま残った音符"),
    ("suppressed_notes", "音節と結びつかず出力しなかった音符"),
    ("pitch_clamped_notes", "音高を丸めた音符"),
    ("quantized_merged_notes", "まとめられて消えた音符"),
    ("quantized_stretched_notes", "長さを 1 tick へ伸ばした音符"),
)

_HUMAN_LYRICS_ONLY_FIELD_LABELS = (
    ("fallback_lyric_notes", "表示歌詞を音声から決めた音符"),
    ("dropped_morae", "破棄した余剰モーラ"),
)


def report_text(fields: dict, *, lyrics_given: bool = False) -> str:
    lines = [f"{_HUMAN_BACKEND_LABELS.get(name, name)}: {value}"
             for name, value in (fields.get("backends") or {}).items() if value is not None]
    lines.append(
        f"テンポ: {fields['tempo_bpm']} ({_HUMAN_TEMPO_SOURCE_LABELS[fields['tempo_source']]})")
    lines.append(f"拍子: {fields['time_signature']} "
                 f"({_HUMAN_TIME_SIGNATURE_SOURCE_LABELS[fields['time_signature_source']]})")
    lines.append(f"分解能(tick/四分音符): {fields['resolution']}")
    lines.append(f"ボーカル分離の実施: {'あり' if fields['separated'] else 'なし'}")
    labels = _HUMAN_FIELD_LABELS + (_HUMAN_LYRICS_ONLY_FIELD_LABELS if lyrics_given else ())
    lines += [f"{label}: {fields[key]}" for key, label in labels if key in fields]
    return "\n".join(lines)
