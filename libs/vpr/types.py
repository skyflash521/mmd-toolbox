from dataclasses import dataclass, field


class VprFormatError(Exception):
    """path は ZIP エントリ名または JSON パス(例 tracks[2].parts[0].notes[4].pos)。"""

    def __init__(self, message: str, *, path=None, key=None, value=None) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.key = key
        self.value = value


@dataclass
class VprWarning:
    """tick はプロジェクト絶対 tick(パート相対ではない)。"""

    code: str
    message: str
    track_index: int | None = None
    part_index: int | None = None
    note_index: int | None = None
    related_note_index: int | None = None
    tick: int | None = None


@dataclass
class VibratoPoint:
    """pos はプロジェクト絶対 tick(ビブラート区間始端からの相対位置ではない)。"""

    pos: int
    value: int


@dataclass
class NoteVibrato:
    """duration は区間長で、単位は tick。"""

    type: int
    duration: int
    depths: list[VibratoPoint] = field(default_factory=list)
    rates: list[VibratoPoint] = field(default_factory=list)


@dataclass
class NoteAiExpression:
    vibrato_leading_depth: float
    vibrato_following_depth: float


@dataclass
class Note:
    """start_tick はプロジェクト絶対 tick(パート相対ではない)。pitch は MIDI ノート番号。"""

    start_tick: int
    duration_tick: int
    pitch: int
    lyric: str
    velocity: int
    phonemes: list[str] = field(default_factory=list)
    is_protected: bool = field(default=False, kw_only=True)
    vibrato: NoteVibrato | None = None
    ai_expression: NoteAiExpression | None = None


@dataclass
class ControllerEvent:
    """tick はプロジェクト絶対 tick(パート相対ではない)。"""

    tick: int
    value: int


@dataclass
class ControllerCurve:
    name: str
    events: list[ControllerEvent] = field(default_factory=list)


@dataclass
class Part:
    name: str
    start_tick: int
    duration_tick: int = 0
    voice: "VoiceBank | None" = None
    notes: list[Note] = field(default_factory=list)
    controllers: list[ControllerCurve] = field(default_factory=list)


@dataclass
class Track:
    name: str
    parts: list[Part] = field(default_factory=list)


@dataclass
class VoiceBank:
    comp_id: str
    name: str


@dataclass
class TempoEvent:
    tick: int
    bpm: float


@dataclass
class TimeSignature:
    tick: int
    numerator: int
    denominator: int


@dataclass
class VprProject:
    """resolution の単位は tick/四分音符。entries は Project/sequence.json を含めない。"""

    resolution: int
    tempos: list[TempoEvent] = field(default_factory=list)
    time_signatures: list[TimeSignature] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    title: str = ""
    raw_sequence: dict | None = None
    entries: dict[str, bytes] | None = None
