"""song2vpr のレポート/診断の構造化。

各段が返した診断を、機械モードの result イベント(`mode:"run"`/`"inspect"`)のフィールドへ対応付ける。
音声処理も外部呼び出しも行わず、確定済みの値を並べるだけにする(契約のキー集合と、どの段の診断が
どのキーになるかを1か所に閉じるため)。

件数はいずれも、それを判定した段の診断から取る。tick へ写す段が判定するものは写した後の音符列を、
写す前の段が判定するものは判定した時点の音符列を数えた値になる。
"""


def run_fields(*, output, tempo, duration_sec, separated, backends,
               split, annotation, build) -> dict:
    """result(`mode:"run"`)のフィールド。キーは常に全部載せる。"""
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
        "short_notes": split.short_notes,
        # 抑制の規則は3つあり、担当する段が分かれている。利用者へ出すのはその合計。段は直列なので
        # 同じ音符が二度数えられることはない。
        "suppressed_notes": split.suppressed_notes + annotation.suppressed_notes,
        "pitch_clamped_notes": build.pitch_clamped_notes,
        "quantized_merged_notes": build.quantized_merged_notes,
        "quantized_stretched_notes": build.quantized_stretched_notes,
        "no_phoneme_notes": annotation.no_phoneme_notes,
    }


def inspect_fields(*, sample_rate, channels, **run) -> dict:
    """result(`mode:"inspect"`)のフィールド。run の全キーに入力のメタ情報を足す。"""
    fields = run_fields(output=None, **run)
    fields.update(input_kind="audio", sample_rate=sample_rate, channels=channels)
    return fields


# 人間向けの表示で使う、バックエンド選択の設定のラベル。
_BACKEND_LABELS = {
    "separator": "ボーカル分離",
    "recognizer": "内容認識モデル",
    "recognizer_revision": "内容認識モデルのリビジョン",
    "forced_aligner": "強制アライメント",
    "english_katakana_method": "英語カタカナ化",
}

# 採用値の出どころ。機械モードの値は契約の語なので、人間向けには日本語で出す。
_SOURCE_LABELS = {"option": "指定値", "estimated": "推定値", "default": "仮置き"}

# 人間向けの表示で使うラベル。並びは利用者向けの診断が挙げる項目の順にそろえる
# (result のキーの並びとは別。読み手が仕様の記述と突き合わせられるようにするため)。
_LABELS = (
    ("duration_sec", "入力の尺(秒)"),
    ("notes", "音符数"),
    ("moraic_nasal_notes", "撥音「ん」を入れた音符"),
    ("vowel_undetermined_notes", "母音が得られず「あ」を入れた音符"),
    ("no_phoneme_notes", "音素列が空の音符"),
    ("short_notes", "短いまま残った音符"),
    ("suppressed_notes", "音節と結びつかず出力しなかった音符"),
    ("pitch_clamped_notes", "音高を丸めた音符"),
    ("quantized_merged_notes", "まとめられて消えた音符"),
    ("quantized_stretched_notes", "長さを 1 tick へ伸ばした音符"),
)

# --lyrics 指定時だけ出す項目(未指定の実行では常に 0 で、読み手の手がかりにならない)。
_LYRICS_LABELS = (
    ("fallback_lyric_notes", "表示歌詞を音声から決めた音符"),
    ("dropped_morae", "破棄した余剰モーラ"),
)


def report_text(fields: dict, *, lyrics_given: bool = False) -> str:
    """`--dry-run`・`--verbose` の人間向け診断を整形する。

    件数は result と同じ値を使う(利用者が見る数と機械利用が受け取る数を一致させるため)。表記は
    人間向けに写す——渡さなかったバックエンド選択の設定は出さず、出どころと分離の有無は日本語で出す。
    書き出し先は出さない(書く実行では完了行が示し、書かない実行では示すものが無い)。
    """
    lines = [f"{_BACKEND_LABELS.get(name, name)}: {value}"
             for name, value in (fields.get("backends") or {}).items() if value is not None]
    lines.append(f"テンポ: {fields['tempo_bpm']} ({_SOURCE_LABELS[fields['tempo_source']]})")
    lines.append(
        f"拍子: {fields['time_signature']} ({_SOURCE_LABELS[fields['time_signature_source']]})")
    lines.append(f"分解能(tick/四分音符): {fields['resolution']}")
    lines.append(f"ボーカル分離の実施: {'あり' if fields['separated'] else 'なし'}")
    labels = _LABELS + (_LYRICS_LABELS if lyrics_given else ())
    lines += [f"{label}: {fields[key]}" for key, label in labels if key in fields]
    return "\n".join(lines)
