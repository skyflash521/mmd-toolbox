"""手組みのプロジェクトを書き出すときの骨組み。

読んだプロジェクトは生の JSON を基礎にできるが、手組みのプロジェクトには基礎が無い。形式が要求する
キーだけを持つ最小の骨組みをここに置き、公開モデルの値で埋めて書き出す。

置くのは形式を成立させるために必要な構造だけで、利用先の判断に属する値(どの歌手を使うか・どんな
歌い方かなど)は持たない。それらは公開モデルが運ぶ。
"""


def sequence() -> dict:
    """トップレベルの骨組み。"""
    return {
        "version": {"major": 6, "minor": 5, "revision": 1},
        "vender": "Yamaha Corporation",
        "title": "",
        "masterTrack": {
            "samplingRate": 44100,
            "tempo": {"isFolded": False, "height": 0.0,
                      "global": {"isEnabled": False, "value": 12000},
                      "ara": {"isEnabled": True}, "events": []},
            "timeSig": {"isFolded": False, "events": []},
            "volume": {"isFolded": True, "height": 0.0, "events": [{"pos": 0, "value": 0}]},
        },
        "voices": [],
        "tracks": [],
    }


def singing_track() -> dict:
    """歌唱トラックの骨組み。"""
    return {
        "type": 2,
        "name": "",
        "color": 0,
        "busNo": 0,
        "isFolded": False,
        "height": 0.0,
        "volume": {"isFolded": True, "height": 0.0, "events": [{"pos": 0, "value": 0}]},
        "panpot": {"isFolded": True, "height": 0.0, "events": [{"pos": 0, "value": 0}]},
        "isMuted": False,
        "isSoloMode": False,
        "parts": [],
    }


def part() -> dict:
    """歌唱パートの骨組み。"""
    return {"name": "", "pos": 0, "duration": 0, "notes": [], "controllers": []}


def note() -> dict:
    """音符の骨組み。読み手が全音符での存在を当てにしてよい6キーと、音素の保護を持つ。

    音素の保護は形式では省略可だが、公開モデルが必ず値を持つので骨組みにも置く。
    """
    return {"pos": 0, "duration": 0, "number": 60, "lyric": "", "phoneme": "", "velocity": 64,
            "isProtected": False}


def lang_ids() -> list:
    """パートの音声バンク参照が持つ言語指定の骨組み。

    値の意味は未解析なので、解析に用いた実 vpr が持っていた形をそのまま置く。
    """
    return [{"langID": 0}, {"langID": 1}]


def controller() -> dict:
    return {"name": "", "events": []}


def tempo_event() -> dict:
    return {"pos": 0, "value": 12000}


def time_signature_event() -> dict:
    return {"bar": 0, "numer": 4, "denom": 4}
