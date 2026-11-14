from cli_resource_watch import warning_texts as _resource_warning_texts


def _no_notes(_fields):
    message = "音符が1つも得られませんでした"
    return message, f"{message}。入力音声か、ボーカル分離の選択を見直してください"


def _tempo_defaulted(_fields):
    message = "テンポを推定できませんでした"
    return message, f"{message}。120 BPM を仮置きします。--tempo で指定できます"


def _kana_reading_ineffective(fields):
    message = "歌詞のかな読みへの変換が効いていないおそれがあります"
    return message, (
        f"{message}(かなへ変換されるべきだった文字 {fields['unconverted_chars']} / "
        f"判定に使った文字 {fields['counted_chars']})。表示歌詞が実際の発声と食い違うことがあります"
    )


def _forced_split(_fields):
    message = "長尺の自動分割で、無音が見つからず目標境界で強制分割しました"
    return message, f"{message}。境界付近の音符が途切れることがあります"


_BUILDERS = {
    "no_notes": _no_notes,
    "tempo_defaulted": _tempo_defaulted,
    "kana_reading_ineffective": _kana_reading_ineffective,
    "forced_split": _forced_split,
}


def warning_texts(code, fields):
    """戻り値は (機械モードの warning イベント本文, 人間向け1行)。"""
    builder = _BUILDERS.get(code)
    if builder is None:
        return _resource_warning_texts(code, fields)
    return builder(fields)
