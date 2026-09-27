from .constants import SINGING_TRACK_TYPE


def sequence() -> dict:
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
    return {
        "type": SINGING_TRACK_TYPE,
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
    return {"name": "", "pos": 0, "duration": 0, "notes": [], "controllers": []}


def note() -> dict:
    return {"pos": 0, "duration": 0, "number": 60, "lyric": "", "phoneme": "", "velocity": 64,
            "isProtected": False}


def lang_ids() -> list:
    return [{"langID": 0}, {"langID": 1}]


def controller() -> dict:
    return {"name": "", "events": []}


def tempo_event() -> dict:
    return {"pos": 0, "value": 12000}


def time_signature_event() -> dict:
    return {"bar": 0, "numer": 4, "denom": 4}
