"""vpr データモデル。

VOCALOID プロジェクト(vpr)の音楽情報を、特定の CLI に依らない正規化モデルとして公開する。
時刻は vpr ネイティブの tick(整数)で保持し、音符の時刻はプロジェクト絶対 tick で正規化する。
秒・フレーム・拍への変換は持たない(呼び出し側の責務)。休符は専用型を持たず、同一トラック内の
発音区間の和集合の補集合として導出する。
"""

from dataclasses import dataclass, field


class VprFormatError(Exception):
    """vpr の構造異常。

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
    """続行可能な事象の構造化報告。

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
class VibratoPoint:
    """ビブラートの自動化曲線の1点。

    pos は他のコントローラ点と同じプロジェクト絶対 tick(ファイル上のビブラート区間相対値では
    ない)、value はファイル格納の生値。
    """

    pos: int
    value: int


@dataclass
class NoteVibrato:
    """音符のビブラート。フィールドと値の意味は形式仕様が定める。"""

    type: int  # ビブラートの種別
    duration: int  # ビブラート区間長(tick)。0 はビブラート無し
    depths: list[VibratoPoint] = field(default_factory=list)  # 深さの自動化曲線
    rates: list[VibratoPoint] = field(default_factory=list)  # 速さの自動化曲線


@dataclass
class NoteAiExpression:
    """音符単位の表現パラメータのうち、ビブラートの深さ包絡。

    2つの値はどちらも必須で、値の不在は外側の Note.ai_expression が None であることだけで表す
    (片方だけを持つ状態を作れないようにして、読みと書きの対称性を保つ)。
    """

    vibrato_leading_depth: float
    vibrato_following_depth: float


@dataclass
class Note:
    """音符。start_tick はプロジェクト絶対 tick、duration_tick は tick 長。"""

    start_tick: int
    duration_tick: int
    pitch: int  # MIDI ノート番号
    lyric: str  # 表示歌詞
    velocity: int  # 0〜127 の生値
    phonemes: list[str] = field(default_factory=list)  # 音符内の音素列(空可)
    # 音素列の保護。値の意味付け・選別は利用先が持つ。既存フィールドの間に挟まる位置なので、
    # 位置引数の並びを動かさないようキーワード専用にする(位置で渡した vibrato がここへ入ると、
    # 真偽値でない値が形式へ書かれてしまう)。
    is_protected: bool = field(default=False, kw_only=True)
    vibrato: NoteVibrato | None = None
    ai_expression: NoteAiExpression | None = None


@dataclass
class ControllerEvent:
    """連続コントローラ曲線の1点。tick はプロジェクト絶対 tick、value はファイル格納の生値。"""

    tick: int
    value: int


@dataclass
class ControllerCurve:
    """パート単位の連続コントローラ曲線(声量 dynamics・表情 s5Expression・音色 等)。

    vpr の各パートが持つ連続パラメータ自動化を生値のまま保持する。どれが声量かの選別・値域の正規化・
    開き量への写像は呼び出し側(各CLI)の責務で、vpr は形式の事実(名前と (tick, 生値) 列)だけ
    公開する。
    """

    name: str  # vpr の controller 名(例 "dynamics"・"s5Expression")
    events: list[ControllerEvent] = field(default_factory=list)  # tick の昇順


@dataclass
class Part:
    """歌唱区間。start_tick はパートの開始位置(プロジェクト絶対 tick)。"""

    name: str
    start_tick: int
    duration_tick: int = 0  # パート長
    voice: "VoiceBank | None" = None  # このパートが使うボイスバンク
    notes: list[Note] = field(default_factory=list)  # start_tick の昇順
    controllers: list[ControllerCurve] = field(default_factory=list)  # 連続コントローラ曲線


@dataclass
class Track:
    name: str
    parts: list[Part] = field(default_factory=list)


@dataclass
class VoiceBank:
    """歌唱に使うボイスバンクの指定。

    どの歌手を使うかは形式の事実でなく利用先の判断なので、vpr は与えられたものを直列化するだけとし、
    値を選ばない。
    """

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
    resolution: int  # tick/四分音符
    tempos: list[TempoEvent] = field(default_factory=list)
    time_signatures: list[TimeSignature] = field(default_factory=list)
    tracks: list[Track] = field(default_factory=list)
    title: str = ""  # 曲名
    # 未解釈データのロスレス保持。read は sequence.json 全体を保持し、手組み時は None。
    raw_sequence: dict | None = None
    # Project/sequence.json 以外の ZIP エントリ。read が保持し、write がそのまま書き戻す。
    entries: dict[str, bytes] | None = None
