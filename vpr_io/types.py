"""vpr_io データモデル(vpr_io.md §2)。

VOCALOID プロジェクト(vpr)の音楽情報を、特定の CLI に依らない正規化モデルとして公開する。
時刻は vpr ネイティブの tick(整数)で保持し、音符の時刻はプロジェクト絶対 tick で正規化する。
秒・フレーム・拍への変換は持たない(呼び出し側の責務)。休符は専用型を持たず、同一トラック内の
発音区間の和集合の補集合として導出する。
"""

from dataclasses import dataclass, field


class VprFormatError(Exception):
    """vpr の構造異常(vpr_io.md §3.1)。

    原因特定のため path(ZIP エントリ名または JSON パス)・key(欠落/型不一致の対象キー)・
    value(問題になった実値)を持つ。
    """

    def __init__(self, message: str, *, path=None, key=None, value=None) -> None:
        super().__init__(message)
        self.message = message
        self.path = path
        self.key = key
        self.value = value


@dataclass
class VprWarning:
    """続行可能な事象の構造化報告(vpr_io.md §3.2)。

    ロケータ(添字・tick)は公開データモデルの階層に対応する。VMD 固有の section/frame は持たない。
    """

    code: str
    message: str
    track_index: int | None = None  # VprProject.tracks の添字
    part_index: int | None = None  # Track.parts の添字
    note_index: int | None = None  # Part.notes の添字
    related_note_index: int | None = None  # 2音符の関係(重なり等)で相手側 Part.notes の添字
    tick: int | None = None  # 対象位置のプロジェクト絶対 tick


@dataclass
class Note:
    """音符。start_tick はプロジェクト絶対 tick、duration_tick は tick 長。"""

    start_tick: int
    duration_tick: int
    pitch: int  # MIDI ノート番号
    lyric: str  # 表示歌詞
    velocity: int  # 0〜127 の生値
    phonemes: list[str] = field(default_factory=list)  # 音符内の音素列(空可)


@dataclass
class Part:
    """歌唱区間。start_tick はパートの開始位置(プロジェクト絶対 tick)。"""

    name: str
    start_tick: int
    notes: list[Note] = field(default_factory=list)  # start_tick の昇順


@dataclass
class Track:
    name: str
    parts: list[Part] = field(default_factory=list)


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
    resolution: int  # tick/四分音符
    tempos: list[TempoEvent] = field(default_factory=list)
    time_signatures: list[TimeSignature] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
