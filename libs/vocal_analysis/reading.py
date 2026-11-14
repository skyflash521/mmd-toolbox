def to_kana_reading(text: str) -> str:
    """失敗はすべて RecognitionError で送出する。text 全体を1つのまとまりとして変換するので、分けて
    呼ぶと語の切れ目が変わり読みが変わりうる。"""
    from .phonemes import RecognitionError

    try:
        from .recognizer import convert_with_g2p

        return convert_with_g2p(text, kana=True)
    except RecognitionError:
        raise
    except Exception as e:
        raise RecognitionError(
            f"かな読みへの変換に失敗しました: {type(e).__name__}: {e}") from e
