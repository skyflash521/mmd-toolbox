"""song2vpr が出す警告の文言。

警告のコードと観測値から、機械モードの warning イベント本文と、観測値を併記した人間向け1行を
組み立てる。文言をツール側に置くのは、勧める対処が song2vpr のオプション名に依存するため。
資源逼迫・GPU 構成について何を観測しどの事実で警告が成立するかは共有の cli_resource_watch が持ち、
それ以外の警告は song2vpr 自身が処理の結果から判定する。
"""


def _gpu_memory_oversubscribed(fields):
    message = "GPUメモリの要求量が空き容量を超過しました"
    return message, (
        f"{message}(要求 {fields['reserved_mib']}MiB / "
        f"開始時空き {fields['free_at_start_mib']}MiB)。超過分は主記憶へ退避され処理が大幅に低速化します。"
        f"--device cpu での実行を検討してください"
    )


def _swap_detected(fields):
    message = "メモリ不足によるスワップを検出しました"
    return message, (
        f"{message}(スワップ増加 {fields['swap_increase_mib']}MiB・"
        f"本プロセス使用 {fields['process_rss_mib']}MiB)。処理が大幅に低速化します。"
        f"他のアプリケーションを終了してメモリを確保してください"
    )


def _cpu_only_torch(fields):
    message = "導入されている torch では GPU を扱えません"
    return message, f"{message}(torch {fields['torch_version']} は CPU 専用版)。CPU で処理します"


def _cuda_unavailable(fields):
    message = "この GPU で使えない CUDA 版の torch が入っています"
    return message, (
        f"{message}(torch {fields['torch_version']})。別の CUDA のバージョンで入れ直してください"
    )


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
    "gpu_memory_oversubscribed": _gpu_memory_oversubscribed,
    "swap_detected": _swap_detected,
    "cpu_only_torch": _cpu_only_torch,
    "cuda_unavailable": _cuda_unavailable,
}


def warning_texts(code, fields):
    """警告のコードと観測値から (機械モードの本文, 人間向け1行) を組み立てる。"""
    return _BUILDERS[code](fields)
