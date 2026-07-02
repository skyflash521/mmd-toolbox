"""vpr2vmd CLI(vpr2vmd.md §4)。

vpr を入力に、口形イベント列と開き量を作って lipsync に渡し、口パク VMD を 1 コマンドで
出力する薄いラッパー。引数解析・検証・出力先解決を担い、vpr 読み込みから口パク VMD
ドキュメント生成までの変換パイプライン(vpr 読み込み → 口形イベント確定 → 開き量 → lipsync)は
_build() が束ねる。--dry-run は出力VMDを書かず、処理計画と診断を表示する。

終了コード(既存ツール shakevmd 等の規約に倣う):
0 正常 / 1 入力不正(入力 vpr の欠落・非vpr 等) / 2 引数エラー(未知オプション・範囲不正・
上書きガード) / 3 出力書き込み失敗。
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass, replace

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

# 口パクスタイルプリセット名(vpr2vmd.md §4.2)。具体値の解決は presets.resolve が担う。
STYLE_NAMES = ("pop", "ballad", "powerful", "whisper", "rap")

# VMD ヘッダのモデル名は固定 20 バイト・Shift-JIS(vpr2vmd.md §4.2・§5)。
_MODEL_NAME_MAX_BYTES = 20


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


def _build_parser() -> argparse.ArgumentParser:
    # allow_abbrev=False: 仕様外の前置き省略形を受理しない(未知/省略形は exit 2)。
    # help= は各オプションの人間向け説明(vpr2vmd.md §4.2・規約 §6)。
    p = argparse.ArgumentParser(prog="vpr2vmd", allow_abbrev=False)
    p.add_argument("input", help="入力 vpr ファイル")
    p.add_argument("-o", "--output", help="出力 VMD(既定: <入力名>.vmd)")
    p.add_argument("--overwrite", action="store_true",
                   help="出力先が入力と同一パスになる指定を許可する(別パスの既存ファイルは常に上書き)")
    # --track は整数なら 0-based INDEX、非整数なら Track.name(vpr2vmd.md §4.2)。解釈・解決は
    # io.select_track が行うため、ここでは生文字列のまま保持する(type=str)。
    p.add_argument("--track",
                   help="口パク対象の歌唱トラック。整数は 0-based の INDEX、非整数は Track 名"
                        "(既定: 先頭トラック)")
    p.add_argument("--model-name", dest="model_name", type=_model_name, default="",
                   help="VMD に格納するモデル名(最大 20 バイト・Shift-JIS)")
    p.add_argument("--style", choices=STYLE_NAMES, default="pop",
                   help="口パクスタイルプリセット(開き量レンジ・タイミング・誇張を切り替える)")
    # --n-morph / --no-n-morph は既定 on の対(vpr2vmd.md §4.2)。dest=n_morph を共有する。
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
    # 後に最終値として上書きする(vpr2vmd.md §3・§4.2)。lipsync の各パラメータの意味は lipsync.md が正本。
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
    p.add_argument("--version", action="version", version=f"vpr2vmd {__version__}",
                   help="バージョンを表示して終了する")
    return p


def _default_output(input_path: str) -> str:
    # vpr2vmd.md §4.1: 既定出力は <入力名(拡張子なし)>.vmd。元の拡張子に依らず常に .vmd。
    base, _ = os.path.splitext(input_path)
    return base + ".vmd"


def _same_path(a: str, b: str) -> bool:
    """2 パスが同一ファイルを指すか。未存在でも realpath 比較で判定する。"""
    # 実ファイルが同一かを優先(symlink・大小無視 FS でも inode で一致判定)。出力先が
    # 未存在だと samefile が立たないので、symlink 解決した realpath で比較する。
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.realpath(a) == os.path.realpath(b)


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
    """--dry-run の診断要約に出す統計と注意事項(vpr2vmd.md §4.4)。"""

    adopted: int  # 採用音符数
    events: int  # 口形イベント数
    morph_keys: int  # モーフキー数
    open_amounts: list[float]  # 採用音符別の開き量(最小/最大/平均の素材)
    overlap: OverlapDiagnostics  # 重複音符の除外・切り詰め件数
    event: EventDiagnostics  # 母音未確定件数・自前イベントを作らない記号


def _build(args):
    """vpr を読み口パク VMD ドキュメントと診断を組み立てる(書き込みはしない。vpr2vmd.md §3〜§5)。

    vpr 解析 → 対象トラック選択 → 重なり解決 → 口形イベント確定 → 開き量 → lipsync →
    モーフキーまでを束ね、`(VmdDocument, _Diagnostics)` を返す。書き込みと診断表示・警告は
    呼び出し側(main)が担う(`--dry-run` でも処理は同じで、出力VMDだけ書かない)。入力不正
    (非vpr・対象トラック皆無)は終了コード 1、`--track` の不正値は 2 を返す(vpr2vmd.md §4.3)。
    """
    try:
        project, warnings = read(args.input)
    except VprFormatError as e:
        return _fail(f"入力を vpr として読めません: {e}", 1)  # 非vpr=入力不正
    _surface_warnings(warnings)  # 重なり音符などの構造化警告を標準エラーへ出す(§4.4)
    if not project.tracks:
        return _fail("入力 vpr にトラックがありません", 1)  # 対象トラック皆無=入力不正
    try:
        track = select_track(project, args.track)
    except TrackSelectionError as e:
        return _fail(f"--track: {e}", 2)  # 当該入力で有効な選択にならない=引数エラー

    adopted, overlap_diag = resolve_overlaps(collect_notes(track))
    openness_params, gen_params = presets.resolve(args.style, args.open_max, args.default_open)
    # 曲の代表BPMから保持・アタック・リリースを縮める(高速テンポでの短母音消失を防ぐ)。
    # --ref-bpm/--tempo-scale-min はテンポ補正の入力なので未指定なら補正関数の既定に委ねる。
    rep_bpm = timing.representative_bpm(adopted, project.tempos, project.resolution)
    tempo_kwargs = {}
    if args.ref_bpm is not None:
        tempo_kwargs["ref_bpm"] = args.ref_bpm
    if args.tempo_scale_min is not None:
        tempo_kwargs["s_min"] = args.tempo_scale_min
    gen_params = apply_tempo_correction(gen_params, rep_bpm, **tempo_kwargs)
    # CLI 調整(指定された値だけを最終値として上書き。テンポ補正後に効く=適用順 A)。これらは
    # テンポでスケールしないパラメータなので、上書き値がそのまま生成に渡る(vpr2vmd.md §3・§4.2)。
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
    # 無ければ velocity 由来へフォールバックする(vpr2vmd.md §3)。
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
    # --legato-max は GenerationParams 外(口形イベント確定段の引数)。未指定なら build_mouth_events の既定。
    legato_kwargs = {} if args.legato_max is None else {"legato_max_frames": args.legato_max}
    mouth_events, event_diag = build_mouth_events(
        adopted,
        project.tempos,
        project.resolution,
        use_n_morph=args.n_morph,
        open_by_note=open_by_note,
        **legato_kwargs,
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
    return document, diagnostics


def _print_diagnostics(diag: _Diagnostics) -> None:
    """--dry-run の診断要約を標準出力へ出す(vpr2vmd.md §4.4)。"""
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


def _valley_bounds_inverted(args) -> bool:
    """解決後の谷係数が下限(deep)>上限(shallow)で退化するか。

    谷係数の不変条件(下限≤上限)は vpr 内容に依らずプリセット既定と CLI 上書きだけで定まるので、
    入力 vpr を読む前(dry-run を含む)に判定できる。テンポ補正は谷係数を変えないため、ここで
    プリセット値と上書きだけから解決して判定してよい(vpr2vmd.md §4.2)。
    """
    _, gen = presets.resolve(args.style, args.open_max, args.default_open)
    shallow = args.valley_shallow if args.valley_shallow is not None else gen.legato_valley_shallow
    deep = args.valley_deep if args.valley_deep is not None else gen.legato_valley_deep
    return deep > shallow


def _fail(message: str, exit_code: int) -> int:
    """失敗理由を標準エラーへ 1 行出して終了コードを返す(vpr2vmd.md §6・§7.4)。"""
    print(f"error: {message}", file=sys.stderr)
    return exit_code


def _surface_warnings(warnings) -> None:
    """vpr 読み込みが返す構造化警告を標準エラーへ出す(同一 code・message は 1 行に集約。§4.4)。"""
    seen = set()
    for w in warnings:
        key = (w.code, w.message)
        if key in seen:
            continue
        seen.add(key)
        print(f"警告: {w.message} ({w.code})", file=sys.stderr)


def main(argv=None) -> int:
    """CLI エントリポイント。終了コードを返す(0/1/2/3。vpr2vmd.md §4.3)。"""
    # 人間向け標準エラーはロケール符号化で表せない文字でも UnicodeEncodeError で落とさない(規約 §10)。
    if hasattr(sys.stderr, "reconfigure"):
        try:
            sys.stderr.reconfigure(errors="backslashreplace")
        except Exception:
            pass
    if argv is None:
        argv = sys.argv[1:]

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as e:
        # argparse はエラー時 code 2 で sys.exit(--help/--version は 0)。終了コードへ変換する。
        code = e.code
        return code if isinstance(code, int) else (0 if code is None else 2)

    # 引数解析後の本体。想定外例外はトレースバックを漏らさず理由 1 行 + 終了コード 1 へ畳む(規約 §5)。
    try:
        return _run(args)
    except Exception as e:
        return _fail(f"{type(e).__name__}: {e}", 1)


def _run(args) -> int:
    """引数解析済みの本体(検証 → 読み込み → 変換 → 書き込み)。失敗は _fail で理由 1 行 + 終了コードを返す。"""
    output = args.output if args.output is not None else _default_output(args.input)

    # 上書きガード(vpr2vmd.md §4.2): 出力先が入力と同一パスになる指定だけを --overwrite 無しで拒否する。
    # 別パスの既存出力ファイルは対象にしない。同一パス判定を存在確認より先に置く(未存在でも入力上書きは弾く)。
    if not args.overwrite and _same_path(output, args.input):
        return _fail(f"出力先が入力と同一パスです(--overwrite が必要): {output}", 2)

    # 谷係数の不変条件(下限≤上限)は vpr 内容に依らない引数レベルの検証。引数エラー(2)を入力不正(1)より
    # 先に評価する規約に従い、存在確認の前に弾く(dry-run でも弾く)。
    if _valley_bounds_inverted(args):
        return _fail("谷係数の下限が上限を超えています(--valley-deep > --valley-shallow)", 2)

    # 入力 vpr の存在確認(欠落は入力不正)。読み込み・形式検証は _build が行う。
    if not os.path.isfile(args.input):
        return _fail(f"入力 vpr が見つかりません: {args.input}", 1)

    # --dry-run でも読み込み・処理は同じく行い(出力VMDだけ書かない)、診断・警告を出せるようにする。
    built = _build(args)
    if isinstance(built, int):
        return built  # 入力不正(1)・--track の不正値(2)は _build が _fail 済み
    document, diagnostics = built

    # 対象トラックに有効な発音が無い(採用音符列が空)→ 警告して正常終了(vpr2vmd.md §4.3・§4.4)。
    if diagnostics.adopted == 0:
        print("警告: 対象トラックに有効な発音がありません", file=sys.stderr)

    # --dry-run / --verbose は処理計画と診断を標準出力へ出す。--dry-run は書かず、--verbose は書く。
    if args.dry_run or args.verbose:
        _print_plan(args, output)
        _print_diagnostics(diagnostics)
    if args.dry_run:
        return 0

    try:
        write_file(document, output)
    except OSError as e:
        return _fail(f"出力の書き込みに失敗: {e}", 3)  # 出力VMDの書き込み失敗
    return 0
