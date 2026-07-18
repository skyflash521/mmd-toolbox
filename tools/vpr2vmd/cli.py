"""vpr2vmd CLI。

vpr を入力に、口形イベント列と開き量を作って lipsync に渡し、口パク VMD を 1 コマンドで
出力する薄いラッパー。引数解析・検証・出力先解決を担い、vpr 読み込みから口パク VMD
ドキュメント生成までの変換パイプライン(vpr 読み込み → 口形イベント確定 → 開き量 → lipsync)は
_build() が束ねる。--dry-run は出力VMDを書かず、処理計画と診断を表示する。

終了コード: 0 正常 / 1 入力不正(入力 vpr の欠落・非vpr 等) /
2 引数エラー(未知オプション・範囲不正・上書きガード) / 3 出力書き込み失敗 / 130 協調的な中断(Ctrl-C 等)。

--machine / --describe は構造化出力モード。標準出力を JSON Lines のイベント
ストリーム専用にし、失敗も error イベントで理由を返す。既定(非機械)の表示・終了コードは変えない。
"""

import argparse
import inspect
import math
import os
import sys
from dataclasses import dataclass, replace

from cli_events import (
    ArgumentParseError,
    EventEmitter,
    MachineArgumentParser,
    argparse_error_field,
    error_event,
)
from lipsync import generate_morph_keys
from vmd import VmdDocument, ensure_frame0_neutral_keys, normalize, write_file
from vpr import VprFormatError, read

from . import __version__, loudness, openness, presets, timing
from .events import (
    EventDiagnostics,
    OverlapDiagnostics,
    build_mouth_events,
    resolve_overlaps,
)
from .io import TrackSelectionError, collect_notes, select_track
from .tempo_correction import apply_tempo_correction

# 口パクスタイルプリセット名。具体値の解決は presets.resolve が担う。
STYLE_NAMES = ("pop", "ballad", "powerful", "whisper", "rap")

# VMD ヘッダのモデル名は固定 20 バイト・Shift-JIS。
_MODEL_NAME_MAX_BYTES = 20

# CLI 未指定時に効く固定既定を、実際に使う呼び出し先の関数シグネチャから 1 か所で取る
# (--describe / --machine の inspect が報告する既定を実挙動と一致させ、値の二重管理を避ける)。
_DEFAULT_LEGATO_MAX = inspect.signature(build_mouth_events).parameters["legato_max_frames"].default
_DEFAULT_REF_BPM = inspect.signature(apply_tempo_correction).parameters["ref_bpm"].default
_DEFAULT_TEMPO_SCALE_MIN = inspect.signature(apply_tempo_correction).parameters["s_min"].default


def _model_name(text: str) -> str:
    """--model-name を検証する。cp932 で表現でき、20 バイト以内であること。"""
    try:
        encoded = text.encode("cp932")
    except UnicodeEncodeError:
        raise argparse.ArgumentTypeError(
            f"モデル名は Shift-JIS(cp932)で表現できる文字のみ: {text!r}"
        )
    if len(encoded) > _MODEL_NAME_MAX_BYTES:
        raise argparse.ArgumentTypeError(
            f"モデル名は cp932 で {_MODEL_NAME_MAX_BYTES} バイト以内"
            f"({len(encoded)} バイト): {text!r}"
        )
    return text


def _open_amount(text: str) -> float:
    """開き量(--open-max / --default-open)を検証する。0.0〜1.0 の有限 float。

    開き量は lipsync の口形開き量(0〜1)・開き量上限(open_cap, 0〜1)に対応する量なので、
    範囲外(負値・1 超)・inf/nan は引数エラーにする。
    """
    v = float(text)  # 非数値は ValueError → argparse が exit 2 にする
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"有限な数値が必要: {text!r}")
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"開き量は 0.0〜1.0 の範囲: {text!r}")
    return v


def _finite_float(text: str) -> float:
    """有限な float へ変換する(inf/nan を弾く)。範囲チェックは呼び出し側の検証関数で行う。"""
    v = float(text)  # 非数値は ValueError → argparse が exit 2 にする
    if not math.isfinite(v):
        raise argparse.ArgumentTypeError(f"有限な数値が必要: {text!r}")
    return v


def _positive_float(text: str) -> float:
    """正の有限 float(--legato-max・--ref-bpm)。フレーム数・BPM は 0 以下になり得ない。"""
    v = _finite_float(text)
    if v <= 0.0:
        raise argparse.ArgumentTypeError(f"正の数値が必要: {text!r}")
    return v


def _unit_float(text: str) -> float:
    """0.0〜1.0 の有限 float(--valley-shallow・--valley-deep)。谷係数は母音高さに対する割合。"""
    v = _finite_float(text)
    if not 0.0 <= v <= 1.0:
        raise argparse.ArgumentTypeError(f"0.0〜1.0 の範囲が必要: {text!r}")
    return v


def _nonneg_float(text: str) -> float:
    """0 以上の有限 float(--valley-slope)。間隙長あたりの谷係数の減少量は負にならない。"""
    v = _finite_float(text)
    if v < 0.0:
        raise argparse.ArgumentTypeError(f"0 以上の数値が必要: {text!r}")
    return v


def _scale_min(text: str) -> float:
    """テンポ補正の下げ止まり係数(--tempo-scale-min)。0 超〜1.0 の有限 float。"""
    v = _finite_float(text)
    if not 0.0 < v <= 1.0:
        raise argparse.ArgumentTypeError(f"0 超〜1.0 の範囲が必要: {text!r}")
    return v


def _nonneg_int(text: str) -> int:
    """0 以上の整数(--anticipation)。0 は先行準備を無効化する。"""
    v = int(text)  # 非整数は ValueError → argparse が exit 2 にする
    if v < 0:
        raise argparse.ArgumentTypeError(f"0 以上の整数が必要: {text!r}")
    return v


def _positive_int(text: str) -> int:
    """1 以上の整数(--coartic-overlap)。協調調音の重なり=基準長は 1 フレーム以上。"""
    v = int(text)
    if v < 1:
        raise argparse.ArgumentTypeError(f"1 以上の整数が必要: {text!r}")
    return v


def _build_parser(machine: bool = False) -> argparse.ArgumentParser:
    # 構造化出力モード(--machine / --describe)は使用法エラーを error イベントへ振り替えるため、
    # SystemExit の代わりに ArgumentParseError を送出する MachineArgumentParser を使う(--help/--version は
    # error() を経由しないので影響を受けず、従来どおり SystemExit で短絡する)。
    # allow_abbrev=False: 仕様外の前置き省略形を受理しない(未知/省略形は exit 2)。
    # help= は各オプションの人間向け説明。
    cls = MachineArgumentParser if machine else argparse.ArgumentParser
    p = cls(prog="vpr2vmd", allow_abbrev=False)
    # input は nargs="?"(--describe を入力無しで成立させるため)。describe 以外の実行では main() が欠落を検査する。
    p.add_argument("input", nargs="?", help="入力 vpr ファイル")
    p.add_argument("-o", "--output", help="出力 VMD(既定: <入力名>.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先の既存ファイルへの上書きを許可する(未指定で出力先に既存ファイルがあるとエラー)")
    # --track は整数なら 0-based INDEX、非整数なら Track.name。解釈・解決は
    # io.select_track が行うため、ここでは生文字列のまま保持する(type=str)。
    p.add_argument("--track",
                   help="口パク対象の歌唱トラック。整数は 0-based の INDEX、非整数は Track 名"
                        "(既定: 先頭トラック)")
    p.add_argument("--model-name", dest="model_name", type=_model_name,
                   default=f"vpr2vmd {__version__}",
                   help="VMD に格納するモデル名(最大 20 バイト・Shift-JIS)")
    p.add_argument("--style", choices=STYLE_NAMES, default="pop",
                   help="口パクスタイルプリセット(開き量レンジ・タイミング・誇張を切り替える)")
    # --n-morph / --no-n-morph は既定 on の対。dest=n_morph を共有する。
    p.add_argument("--n-morph", dest="n_morph", action="store_true", default=True,
                   help="撥音「ん」に「ん」モーフを使う(既定 on)。--no-n-morph の対の明示形")
    p.add_argument("--no-n-morph", dest="n_morph", action="store_false",
                   help="撥音「ん」に「ん」モーフを使わず無音(閉口)に倒す。--n-morph の対")
    # 既定はプリセット値。未指定センチネル(None)は presets.resolve がプリセットから解決する。
    p.add_argument("--open-max", dest="open_max", type=_open_amount,
                   help="口の開き量の上限(0.0〜1.0。既定: プリセット値)")
    p.add_argument("--default-open", dest="default_open", type=_open_amount,
                   help="ベロシティが一様なときの既定開き量(0.0〜1.0。既定: 開き量レンジ中央)")
    # 視覚で詰める調整パラメータ(未指定 None はプリセット/既定値を使う)。プリセット解決とテンポ補正の
    # 後に最終値として上書きする。各パラメータの意味は lipsync 側が定める。
    p.add_argument("--legato-max", dest="legato_max", type=_positive_float,
                   help="レガート間隙とみなす間隙長の上限(フレーム・正値。既定: 8.0)")
    p.add_argument("--valley-shallow", dest="valley_shallow", type=_unit_float,
                   help="レガート谷の谷係数の上限(浅い側・0.0〜1.0。既定: プリセット値)")
    p.add_argument("--valley-deep", dest="valley_deep", type=_unit_float,
                   help="レガート谷の谷係数の下限(深い側・0.0〜1.0。既定: プリセット値)")
    p.add_argument("--valley-slope", dest="valley_slope", type=_nonneg_float,
                   help="間隙長 1 フレームあたりの谷係数の減少(0 以上。既定: プリセット値)")
    p.add_argument("--coartic-overlap", dest="coartic_overlap", type=_positive_int,
                   help="協調調音の重なり上限(=基準長・1 以上。既定: プリセット値)")
    p.add_argument("--anticipation", dest="anticipation", type=_nonneg_int,
                   help="母音口形の先行準備フレーム数(0 以上・0 で無効。既定: プリセット値)")
    p.add_argument("--ref-bpm", dest="ref_bpm", type=_positive_float,
                   help="テンポ補正の基準テンポ(正値。既定: 120)")
    p.add_argument("--tempo-scale-min", dest="tempo_scale_min", type=_scale_min,
                   help="テンポ補正の下げ止まり係数(0 超〜1.0。既定: 0.5)")
    p.add_argument("--dry-run", dest="dry_run", action="store_true",
                   help="出力せず処理計画と診断を表示する")
    p.add_argument("-v", "--verbose", dest="verbose", action="store_true",
                   help="通常実行でも処理計画と診断を標準出力へ表示する(出力 VMD は書く)")
    p.add_argument("--machine", action="store_true",
                   help="出力を JSON Lines のイベントストリームにする(標準出力=イベント専用・"
                        "標準エラー=人間向けログ)。既定の人間向け表示・終了コードは変えない")
    p.add_argument("--describe", action="store_true",
                   help="オプション定義とスタイルプリセット一覧の result を出して終了する"
                        "(vpr を読まない・入力不要の自己記述)")
    p.add_argument("--version", action="version", version=f"vpr2vmd {__version__}",
                   help="バージョンを表示して終了する")
    return p


def _default_output(input_path: str) -> str:
    # 既定出力は <入力名(拡張子なし)>.vmd。元の拡張子に依らず常に .vmd。
    base, _ = os.path.splitext(input_path)
    return base + ".vmd"


# --describe の型/制約表。dest → (type, constraint)。help/default は parser の各 action から取り、
# 固定既定(legato_max/ref_bpm/tempo_scale_min)だけ _DESCRIBE_DEFAULT で上書きする。メタ/モード操作
# (describe/version/help/machine)は _D_TYPE に無いので options から除外される。
_D_UNIT = {"min": 0, "max": 1, "exclusive_min": False}       # 0〜1(開き量・谷係数)
_D_POS = {"min": 0, "max": None, "exclusive_min": True}      # 0 超(--legato-max・--ref-bpm)
_D_NONNEG = {"min": 0, "max": None, "exclusive_min": False}  # 0 以上(--valley-slope・--anticipation)
_D_INT1 = {"min": 1, "max": None, "exclusive_min": False}    # 1 以上(--coartic-overlap)
_D_UNIT_EXCL = {"min": 0, "max": 1, "exclusive_min": True}   # 0 超〜1(--tempo-scale-min)

_D_TYPE = {
    "input": ("str", None),
    "output": ("str", None),
    "overwrite": ("flag", None),
    "track": ("str", None),
    "model_name": ("str", None),
    "style": ("enum", {"choices": list(STYLE_NAMES)}),
    "n_morph": ("flag", None),
    "open_max": ("float", _D_UNIT),
    "default_open": ("float", _D_UNIT),
    "legato_max": ("float", _D_POS),
    "valley_shallow": ("float", _D_UNIT),
    "valley_deep": ("float", _D_UNIT),
    "valley_slope": ("float", _D_NONNEG),
    "coartic_overlap": ("int", _D_INT1),
    "anticipation": ("int", _D_NONNEG),
    "ref_bpm": ("float", _D_POS),
    "tempo_scale_min": ("float", _D_UNIT_EXCL),
    "dry_run": ("flag", None),
    "verbose": ("flag", None),
}

# 固定既定を持つオプションの default(argparse は None センチネルなので、実効既定を呼び出し先から取る)。
_DESCRIBE_DEFAULT = {
    "legato_max": _DEFAULT_LEGATO_MAX,
    "ref_bpm": _DEFAULT_REF_BPM,
    "tempo_scale_min": _DEFAULT_TEMPO_SCALE_MIN,
}


def _describe_options(parser):
    """--describe の options を parser 定義から機械導出する。順序は add_argument 順。

    各要素は {name, type, constraint, default, help}(キー 5 つ)。メタ/モード操作(--describe/--version/
    --help/--machine)は _D_TYPE に無いので除外。真偽フラグの否定形(--no-n-morph)は肯定形の長形式で
    既に載るのでスキップする。default は固定既定を持つものは _DESCRIBE_DEFAULT、それ以外は action.default。
    """
    options = []
    for action in parser._actions:
        dest = action.dest
        if dest not in _D_TYPE:
            continue
        type_, constraint = _D_TYPE[dest]
        if dest == "input":
            name = "input"
        else:
            # 肯定形の長形式を採る。--no-* だけの否定形 action はスキップ(肯定形で既に載る)。
            pos = [s for s in action.option_strings if s.startswith("--") and not s.startswith("--no-")]
            if not pos:
                continue
            name = pos[0]
        options.append({
            "name": name,
            "type": type_,
            "constraint": constraint,
            "default": _DESCRIBE_DEFAULT.get(dest, action.default),
            "help": action.help,
        })
    return options


def _describe_presets():
    """--describe の presets を presets モジュールから導出する。

    各要素は {name, values}。name はスタイル名、values は CLI で上書き可能なパラメータの
    プリセット解決値(開き量上限・既定開き量・谷係数・協調調音重なり・先行準備)。
    """
    out = []
    for name in STYLE_NAMES:
        openness_p, gen = presets.resolve(name)
        out.append({
            "name": name,
            "values": {
                "open_max": openness_p.open_max,
                "default_open": openness_p.default_open,
                "valley_shallow": gen.legato_valley_shallow,
                "valley_deep": gen.legato_valley_deep,
                "valley_slope": gen.legato_valley_slope,
                "coartic_overlap": gen.coartic_overlap_max,
                "anticipation": gen.anticipation_frames,
            },
        })
    return out


def _print_plan(args, output: str) -> None:
    """--dry-run の処理計画表示(出力は書かない)。"""
    print(f"input: {args.input}")
    print(f"output: {output}")
    print(f"track: {args.track if args.track is not None else '(先頭トラック)'}")
    print(f"style: {args.style}")
    # 既定は撥音「ん」に「ん」モーフを使う(on)。--no-n-morph 指定時は無音へ倒す(off)。
    print(f"n-morph: {'on (撥音→ん)' if args.n_morph else 'off (撥音→無音)'}")
    print(f"model-name: {args.model_name!r}")
    print(f"open-max: {args.open_max if args.open_max is not None else '(プリセット値)'}")
    print(
        f"default-open: "
        f"{args.default_open if args.default_open is not None else '(プリセット値)'}"
    )
    # 調整パラメータ(未指定はプリセット/既定値を使う)。
    def shown(v):
        return v if v is not None else "(既定)"

    print(f"legato-max: {shown(args.legato_max)}")
    print(
        f"valley(shallow/deep/slope): "
        f"{shown(args.valley_shallow)}/{shown(args.valley_deep)}/{shown(args.valley_slope)}"
    )
    print(f"coartic-overlap: {shown(args.coartic_overlap)}")
    print(f"anticipation: {shown(args.anticipation)}")
    print(f"tempo(ref-bpm/scale-min): {shown(args.ref_bpm)}/{shown(args.tempo_scale_min)}")


@dataclass
class _Diagnostics:
    """--dry-run の診断要約に出す統計と注意事項。"""

    adopted: int  # 採用音符数
    events: int  # 口形イベント数
    morph_keys: int  # モーフキー数
    open_amounts: list[float]  # 採用音符別の開き量(最小/最大/平均の素材)
    overlap: OverlapDiagnostics  # 重複音符の除外・切り詰め件数
    event: EventDiagnostics  # 母音未確定件数・自前イベントを作らない記号


@dataclass
class _Built:
    """_build() の成果。書き込み・診断表示・イベント送出は呼び出し側(_run)が担う。"""

    document: VmdDocument
    diagnostics: _Diagnostics
    track_index: int  # 解決した対象トラックの 0-based INDEX
    track_name: str  # 解決した対象トラック名
    resolved: dict  # inspect の params(スタイル解決→テンポ補正→CLI 上書き適用後の最終値)


def _build(args, emitter, fail):
    """vpr を読み口パク VMD ドキュメントと診断・解決値を組み立てる(書き込みはしない)。

    vpr 解析 → 対象トラック選択 → 重なり解決 → 口形イベント確定 → 開き量 → lipsync → モーフキーまでを
    束ね、`_Built` を返す。読み込み警告は surface し、失敗は fail() で終端して終了コードを返す
    (非vpr・対象トラック皆無は 1、`--track` の不正値は 2)。
    """
    try:
        project, warnings = read(args.input)
    except VprFormatError as e:
        return fail("not_vpr", f"入力を vpr として読めません: {e}", 1, field="input")
    _surface_warnings(warnings, emitter)  # 重なり音符などの構造化警告を surface する
    if not project.tracks:
        return fail("no_tracks", "入力 vpr にトラックがありません", 1, field="input")
    try:
        track = select_track(project, args.track)
    except TrackSelectionError as e:
        return fail("bad_track", f"--track: {e}", 2, field="--track")
    track_index = next(i for i, t in enumerate(project.tracks) if t is track)

    adopted, overlap_diag = resolve_overlaps(collect_notes(track))
    openness_params, gen_params = presets.resolve(args.style, args.open_max, args.default_open)
    # 曲の代表BPMから保持・アタック・リリースを縮める(高速テンポでの短母音消失を防ぐ)。未指定の
    # --ref-bpm/--tempo-scale-min は補正関数の既定を明示適用する(inspect が報告する既定と一致させる)。
    rep_bpm = timing.representative_bpm(adopted, project.tempos, project.resolution)
    ref_bpm = args.ref_bpm if args.ref_bpm is not None else _DEFAULT_REF_BPM
    tempo_scale_min = args.tempo_scale_min if args.tempo_scale_min is not None else _DEFAULT_TEMPO_SCALE_MIN
    gen_params = apply_tempo_correction(gen_params, rep_bpm, ref_bpm=ref_bpm, s_min=tempo_scale_min)
    # CLI 調整(指定された値だけを最終値として上書き。テンポ補正後に効く=適用順 A)。これらは
    # テンポでスケールしないパラメータなので、上書き値がそのまま生成に渡る。
    overrides = {}
    if args.coartic_overlap is not None:
        overrides["coartic_overlap_max"] = args.coartic_overlap
    if args.anticipation is not None:
        overrides["anticipation_frames"] = args.anticipation
    if args.valley_shallow is not None:
        overrides["legato_valley_shallow"] = args.valley_shallow
    if args.valley_deep is not None:
        overrides["legato_valley_deep"] = args.valley_deep
    if args.valley_slope is not None:
        overrides["legato_valley_slope"] = args.valley_slope
    if overrides:
        gen_params = replace(gen_params, **overrides)
    # 開き量(強弱): 声量コントローラ曲線(dynamics/s5Expression)があればモーラ区間平均から写し、
    # 無ければ velocity 由来へフォールバックする。
    open_by_note = loudness.open_amounts_from_loudness(
        track.parts,
        adopted,
        lo=openness_params.lo,
        hi=openness_params.hi,
        open_max=openness_params.open_max,
        gamma=openness_params.gamma,
    )
    if open_by_note is None:
        open_by_note = openness.open_amounts(
            [note.velocity for note in adopted],
            lo=openness_params.lo,
            hi=openness_params.hi,
            open_max=openness_params.open_max,
            default_open=openness_params.default_open,
            gamma=openness_params.gamma,
        )
    # --legato-max は GenerationParams 外(口形イベント確定段の引数)。未指定なら固定既定を明示適用する。
    legato_max = args.legato_max if args.legato_max is not None else _DEFAULT_LEGATO_MAX
    mouth_events, event_diag = build_mouth_events(
        adopted,
        project.tempos,
        project.resolution,
        use_n_morph=args.n_morph,
        open_by_note=open_by_note,
        legato_max_frames=legato_max,
    )
    morph_keys = generate_morph_keys(mouth_events, gen_params)

    document = VmdDocument(
        model_name_raw=args.model_name.encode("cp932").ljust(20, b"\x00"),
        morph=morph_keys,
    )
    # 使用モーフを 0F に中立登録してから(編集・MMD互換規約)フレーム順へ正規化する。
    document = ensure_frame0_neutral_keys(document, sections=("morph",))
    document, _warnings = normalize(document, sections=["morph"])
    diagnostics = _Diagnostics(
        adopted=len(adopted),
        events=len(mouth_events),
        # 実際に書き込まれるキー数(0F 中立登録・normalize 後)を数える。
        morph_keys=len(document.morph),
        open_amounts=list(open_by_note),
        overlap=overlap_diag,
        event=event_diag,
    )
    resolved = {
        "open_max": openness_params.open_max,
        "default_open": openness_params.default_open,
        "legato_max": legato_max,
        "valley_shallow": gen_params.legato_valley_shallow,
        "valley_deep": gen_params.legato_valley_deep,
        "valley_slope": gen_params.legato_valley_slope,
        "coartic_overlap": gen_params.coartic_overlap_max,
        "anticipation": gen_params.anticipation_frames,
        "ref_bpm": ref_bpm,
        "tempo_scale_min": tempo_scale_min,
        "representative_bpm": rep_bpm,
    }
    return _Built(document, diagnostics, track_index, track.name, resolved)


def _print_diagnostics(diag: _Diagnostics) -> None:
    """--dry-run の診断要約を標準出力へ出す。"""
    print("--- 診断 ---")
    print(f"採用音符数: {diag.adopted}")
    print(f"口形イベント数: {diag.events}")
    print(f"モーフキー数: {diag.morph_keys}")
    if diag.open_amounts:
        lo = min(diag.open_amounts)
        hi = max(diag.open_amounts)
        avg = sum(diag.open_amounts) / len(diag.open_amounts)
        print(f"開き量(最小/最大/平均): {lo:.3f}/{hi:.3f}/{avg:.3f}")
    print(f"母音未確定: {diag.event.vowel_undetermined}")
    print(f"重複音符 除外: {diag.overlap.excluded} 切り詰め: {diag.overlap.truncated}")
    symbols = diag.event.non_event_symbols
    if symbols:
        listed = " ".join(f"{sym}({symbols[sym]})" for sym in sorted(symbols))
        print(f"イベント外記号: {listed}")


def _open_amounts_stats(amounts):
    """採用音符別開き量の {min, max, mean}(採用 0 件なら None。inspect 用)。"""
    if not amounts:
        return None
    return {"min": min(amounts), "max": max(amounts), "mean": sum(amounts) / len(amounts)}


def _build_inspect(args, built: _Built) -> dict:
    """`--machine --dry-run` の inspect result ペイロードを組む。VMD は書かない。"""
    diag = built.diagnostics
    return {
        "output": None,
        "input_kind": "vpr",
        "track_index": built.track_index,
        "track_name": built.track_name,
        "style": args.style,
        "n_morph": args.n_morph,
        "model_name": args.model_name,
        "params": built.resolved,
        "adopted_notes": diag.adopted,
        "mouth_events": diag.events,
        "morph_keys": diag.morph_keys,
        "open_amounts": _open_amounts_stats(diag.open_amounts),
        "vowel_undetermined": diag.event.vowel_undetermined,
        "overlap_excluded": diag.overlap.excluded,
        "overlap_truncated": diag.overlap.truncated,
        "non_event_symbols": diag.event.non_event_symbols,
    }


def _valley_bounds_inverted(args) -> bool:
    """解決後の谷係数が下限(deep)>上限(shallow)で退化するか。

    谷係数の不変条件(下限≤上限)は vpr 内容に依らずプリセット既定と CLI 上書きだけで定まるので、
    入力 vpr を読む前(dry-run を含む)に判定できる。テンポ補正は谷係数を変えないため、ここで
    プリセット値と上書きだけから解決して判定してよい。
    """
    _, gen = presets.resolve(args.style, args.open_max, args.default_open)
    shallow = args.valley_shallow if args.valley_shallow is not None else gen.legato_valley_shallow
    deep = args.valley_deep if args.valley_deep is not None else gen.legato_valley_deep
    return deep > shallow


def _surface_warnings(warnings, emitter) -> None:
    """vpr 読み込みが返す構造化警告を surface する。

    機械モードは 1 警告 1 イベント(vpr 内の位置キー付き・section は null)、非機械は code・message の
    同一組を 1 行に集約して標準エラーへ出す。
    """
    if emitter is not None:
        for w in warnings:
            emitter.warning(
                code=w.code, message=w.message, section=None,
                track_index=w.track_index, part_index=w.part_index, note_index=w.note_index,
                related_note_index=w.related_note_index, tick=w.tick,
            )
        return
    seen = set()
    for w in warnings:
        key = (w.code, w.message)
        if key in seen:
            continue
        seen.add(key)
        print(f"warning: {w.code}: {w.message}", file=sys.stderr)


def main(argv=None) -> int:
    """CLI エントリポイント。終了コードを返す(0/1/2/3/130)。"""
    # 人間向け標準エラーはロケール符号化で表せない文字でも UnicodeEncodeError で落とさない。
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    if argv is None:
        argv = sys.argv[1:]

    # 構造化出力モード判定。解析前に argv で先取り(引数エラー時も出力チャネルを決めるため)。
    # --describe は --machine を要さない独立メタ操作。どちらかがあれば emitter を用意し、
    # MachineArgumentParser で使用法エラーも error イベントへ振り替える。emitter はバイナリ stdout へ
    # UTF-8 で書く(ロケール符号化非依存)。どちらも無ければ None(従来の人間向け経路)。
    machine = "--machine" in argv
    describe = "--describe" in argv
    emitter = EventEmitter(sys.stdout.buffer) if (machine or describe) else None

    def fail(code, message, exit_code, *, field=None, path=None):
        """失敗を報告して終了コードを返す。構造化出力モードは error イベントでストリームを終端し、
        それ以外は理由を標準エラーへ 1 行出す(トレースバックは出さない)。"""
        if emitter is not None:
            emitter.error(**error_event(
                code=code, message=message, exit_code=exit_code, field=field, path=path))
        else:
            print(f"error: {message}", file=sys.stderr)
        return exit_code

    parser = _build_parser(machine or describe)
    try:
        args = parser.parse_args(argv)
    except ArgumentParseError as e:
        # 構造化出力モードの MachineArgumentParser は使用法エラーで例外を送出する(SystemExit の代わり)。
        return fail("bad_argument", e.message, 2, field=argparse_error_field(e.message))
    except SystemExit as e:
        # 非機械の使用法エラー(argparse が stderr へ出力済み・code 2)と、両モードの --help/--version
        # (メタ操作・code 0)。例外を握って終了コードへ変換する。
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)

    # 自己記述。vpr を読まず options/presets の result を出して終了する独立メタ操作。
    if args.describe:
        emitter.result(mode="describe", options=_describe_options(parser),
                       presets=_describe_presets())
        return 0
    # input は nargs="?"(--describe を入力無しで成立させるため)。describe 以外の実行では必須。
    if args.input is None:
        return fail("bad_argument", "入力 vpr(input)が必要です", 2, field="input")

    # 引数解析後の本体。KeyboardInterrupt(Ctrl-C 等)は協調的な中断(cancelled/130)として畳み、それ以外の
    # 想定外例外はトレースバックを漏らさず internal_error(理由 1 行 + 終了コード 1)へ畳む。
    # 書き込みは全計算後に 1 回だけ起きるため、中断でも中途半端な出力ファイルは残らない。
    try:
        return _run(args, emitter, fail)
    except KeyboardInterrupt:
        return fail("cancelled", "中断された(Ctrl-C 等)", 130)
    except Exception as e:
        return fail("internal_error", f"{type(e).__name__}: {e}", 1)


def _run(args, emitter, fail) -> int:
    """引数解析済みの本体(検証 → 読み込み → 変換 → 書き込み)。失敗は fail() で終端する。"""
    output = args.output if args.output is not None else _default_output(args.input)

    # 上書きガード: 出力先に既存ファイルがある場合は --overwrite 無しで拒否する。
    if not args.overwrite and os.path.exists(output):
        return fail("output_exists",
                    f"出力先に既存ファイルがあります(--overwrite が必要): {output}", 2, field="--output")

    # 谷係数の不変条件(下限≤上限)は vpr 内容に依らない引数レベルの検証。引数エラー(2)を入力不正(1)より
    # 先に評価する規約に従い、存在確認の前に弾く(dry-run でも弾く)。
    if _valley_bounds_inverted(args):
        return fail("valley_bounds_inverted",
                    "谷係数の下限が上限を超えています(--valley-deep > --valley-shallow)", 2)

    # 入力 vpr の存在確認(欠落は入力不正)。読み込み・形式検証は _build が行う。
    if not os.path.isfile(args.input):
        return fail("input_not_found", f"入力 vpr が見つかりません: {args.input}", 1, field="input")

    # --dry-run でも読み込み・処理は同じく行い(出力VMDだけ書かない)、診断・警告を出せるようにする。
    built = _build(args, emitter, fail)
    if isinstance(built, int):
        return built  # not_vpr(1)・no_tracks(1)・bad_track(2)は _build が fail 済み
    diag = built.diagnostics

    # 対象トラックに有効な発音が無い(採用音符列が空)→ 警告して正常終了。
    if diag.adopted == 0:
        if emitter is not None:
            emitter.warning(
                code="no_adopted_notes", message="対象トラックに有効な発音がありません", section=None,
                track_index=None, part_index=None, note_index=None, related_note_index=None, tick=None,
            )
        else:
            print("warning: no_adopted_notes: 対象トラックに有効な発音がありません", file=sys.stderr)

    # --dry-run / --verbose は処理計画と診断を標準出力へ出す。機械モードは標準出力をイベント専用に保つ
    # ため人間向け表示は出さない。
    if (args.dry_run or args.verbose) and emitter is None:
        _print_plan(args, output)
        _print_diagnostics(diag)

    # --dry-run は出力を書かずに終える。機械モードは入力検査(inspect)の result で終端する。
    if args.dry_run:
        if emitter is not None:
            emitter.result(mode="inspect", **_build_inspect(args, built))
        return 0

    try:
        write_file(built.document, output)
    except OSError as e:
        return fail("write_failed", f"出力の書き込みに失敗: {e}", 3, field="--output", path=output)

    # 書き込み成功後に convert result でストリームを終端する。
    if emitter is not None:
        emitter.result(
            mode="convert", output=output, track_index=built.track_index, track_name=built.track_name,
            morph_keys=diag.morph_keys, adopted_notes=diag.adopted, mouth_events=diag.events,
        )
    return 0
