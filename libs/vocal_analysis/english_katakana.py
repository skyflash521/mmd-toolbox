"""英語カタカナ化フォールバック。

pyopenjtalk-plusが1形態素ノードで完結する英単語を正しく読めない場合(品詞がフィラーと
判定される場合)に、CMUdictで発音記号(ARPAbet)を引き、変換方式(既定: arpakanaによるルール
ベース変換。選択式: ENGLISH_KATAKANA_MODELの生成モデルによる変換)でカタカナへ補完変換
する。CMUdictに無い語・変換結果がひらがな/カタカナ以外を含む場合は変換せず元のテキストのまま
残す(安全側フォールバック)。
"""

import unicodedata
from collections.abc import Callable

from .config import (
    DEFAULT_ENGLISH_KATAKANA_METHOD,
    ENGLISH_KATAKANA_MODEL,
    EnglishKatakanaMethod,
)
from .quiet import silence_third_party_output, suppress_native_stderr

_FULLWIDTH_OFFSET = 0xFEE0
_FULLWIDTH_LO = 0xFF01
_FULLWIDTH_HI = 0xFF5E

_KANA_RANGES = (
    (0x3040, 0x309F),  # ひらがな
    (0x30A0, 0x30FF),  # カタカナ・長音符
)

_PROMPT_TEMPLATE = """英語とその単語単位の音素から、リエゾンを考慮したカタカナと繋がった音素列を生成してください。

英語: take it easy
単語音素: [T EY1 K] [IH1 T] [IY1 Z IY0]
カタカナ: テイキットイージー
繋がった音素: T EY1 K IH1 T IY1 Z IY0

英語: {word}
単語音素: [{phonemes}]
カタカナ: """


def _is_target_node(surface_normalized: str, pos: str) -> bool:
    """NFKC正規化後の表層形・品詞から対象ノードかどうかを判定する。

    ASCII英字のみ(1文字以上)で構成され、かつ品詞がフィラー(pyopenjtalk-plusが正しい読みを
    持たない語)であるノードだけを対象とする。品詞が名詞(既に正しく変換済み)・記号(複数
    ノードへ分裂した語の断片)のノードは対象にしない。
    """
    if not surface_normalized or not surface_normalized.isascii() or not surface_normalized.isalpha():
        return False
    return pos == "フィラー"


def _find_target_words(text: str) -> list[str]:
    """テキストを形態素解析し、対象語(NFKC正規化後の表層形)を出現順に列挙する(重複除去しない)。"""
    import pyopenjtalk

    targets = []
    with suppress_native_stderr():
        nodes = pyopenjtalk.run_frontend(text)
    for node in nodes:
        surface = unicodedata.normalize("NFKC", node["string"])
        if _is_target_node(surface, node["pos"]):
            targets.append(surface)
    return targets


_cmudict_cache = None


def _get_cmudict_entries():
    """CMUdict辞書全体をプロセス内キャッシュする(nltkのdict()は呼び出すたびに辞書全体を
    再構築するため、_g2p経由で頻繁に呼ばれるこの関数の呼び出しごとに再構築させない)。
    未取得の場合はnltkの標準キャッシュへ自動取得する(他の学習済みモデルの初回取得と同様、
    利用者に手動コマンドを要求しない)。
    """
    global _cmudict_cache
    if _cmudict_cache is None:
        import nltk
        from nltk.corpus import cmudict

        try:
            _cmudict_cache = cmudict.dict()
        except LookupError:
            nltk.download("cmudict", quiet=True)
            _cmudict_cache = cmudict.dict()
    return _cmudict_cache


def _lookup_cmudict_phonemes(word: str) -> str | None:
    """CMUdictで英単語の発音記号(ARPAbet)を引く。複数発音があれば先頭を使う。未収録ならNone。"""
    entries = _get_cmudict_entries().get(word.lower())
    if not entries:
        return None
    return " ".join(entries[0])


def _is_valid_katakana(text: str) -> bool:
    """生成結果がひらがな・カタカナのみで構成されるかを判定する(出力妥当性検証)。"""
    if not text:
        return False
    return all(any(lo <= ord(ch) <= hi for lo, hi in _KANA_RANGES) for ch in text)


def _select_katakana_model_device() -> str:
    """変換モデルの実行デバイスを環境から自動選択する(GPUが利用可能ならGPUを使う)。

    変換モデルは1.1Bパラメータの生成モデルで、CPU実行では1回の生成呼び出しに数秒かかる。
    既存のS2内容認識モデル(recognizer._select_device)と同じ考え方で
    GPUを優先する。
    """
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


_katakana_model_cache = None


def _hf_snapshot_download(repo_id, *, revision=None, tqdm_class=None):
    """huggingface_hub.snapshot_download への薄いラッパー(モンキーパッチの受け口)。"""
    from huggingface_hub import snapshot_download

    return snapshot_download(repo_id, revision=revision, tqdm_class=tqdm_class)


def _prefetch_with_progress(repo_id: str, revision: str | None, on_progress: Callable[[str], None]) -> bool:
    """repo_id のファイル群を事前フェッチし、実際にバイト転送が発生した区間だけ
    on_progress(f"ダウンロード中: {repo_id} {percent}%") を呼ぶ。戻り値は実際にダウンロードが
    発生したか。recognizer._prefetch_with_progress と同じ仕組み(huggingface_hub の
    tqdm_class 差し込み口)で、変換モデルの取得元(huggingface_hub)がrecognizer側と同じ
    ため同型だが、recognizer側を再利用すると循環import(recognizer→本モジュール)になるため
    ここで個別に持つ。
    """
    from huggingface_hub.utils import tqdm as hf_tqdm

    state = {"shown": False}

    class _RelayTqdm(hf_tqdm):
        def __init__(self, *args, **kwargs):
            self._relay_unit = kwargs.get("unit")
            self._relay_n = kwargs.get("initial") or 0
            super().__init__(*args, **kwargs)

        def update(self, n=1):
            result = super().update(n)
            if self._relay_unit == "B" and self.total:
                self._relay_n += n or 0
                state["shown"] = True
                percent = min(100, int(self._relay_n * 100 / self.total))
                on_progress(f"ダウンロード中: {repo_id} {percent}%")
            return result

    _hf_snapshot_download(repo_id, revision=revision, tqdm_class=_RelayTqdm)
    return state["shown"]


def _load_katakana_model(on_progress: Callable[[str], None] | None = None):
    """変換モデルをロードする(プロセス内キャッシュ)。実行デバイスは環境から自動選択する。

    on_progress はモデルの初回取得が実際にネットワークダウンロードを要した区間だけダウンロード
    進捗文言を渡して呼ぶことに加え、ロード開始時(ダウンロードの有無に関わらず、キャッシュ済みで
    ディスクから読み込むだけの場合を含む)にも1回呼び、ロード完了までの区間を無通知にしない。
    """
    global _katakana_model_cache
    if _katakana_model_cache is not None:
        return _katakana_model_cache

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    silence_third_party_output()

    downloaded = False
    if on_progress is not None:
        downloaded = _prefetch_with_progress(
            ENGLISH_KATAKANA_MODEL.model_id, ENGLISH_KATAKANA_MODEL.model_revision, on_progress)

    device = _select_katakana_model_device()

    try:
        if on_progress is not None:
            on_progress(f"カタカナ生成モデル読み込み中: {ENGLISH_KATAKANA_MODEL.model_id}")
        tokenizer = AutoTokenizer.from_pretrained(
            ENGLISH_KATAKANA_MODEL.model_id, revision=ENGLISH_KATAKANA_MODEL.model_revision
        )
        # GPU実行時はfp16でロードし、fp32(約4.4GB)に対して重みのVRAM使用量を半減させる
        # (実曲の対象語でfp32と変換結果が一致することを確認して採用)。CPU実行時はfp16未対応の
        # ためfp32のまま。
        dtype = torch.float16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            ENGLISH_KATAKANA_MODEL.model_id,
            revision=ENGLISH_KATAKANA_MODEL.model_revision,
            dtype=dtype,
        )
    finally:
        # ロードが完了した時点で通知を終える(ダウンロードが実際に発生した場合のみ)。
        # 例外時もライブ表示側の後始末に合わせクリアする。
        if downloaded:
            on_progress("")
    model.to(device)
    model.eval()
    _katakana_model_cache = (tokenizer, model)
    return _katakana_model_cache


def _generate_katakana_tinyllama(
    word: str, phonemes: str, on_progress: Callable[[str], None] | None = None
) -> str:
    """英単語・発音記号からカタカナを生成する(変換モデル呼び出し。method="tinyllama-katakana-
    converter"選択時のみ使う)。

    _load_katakana_model()が送出する例外(未導入時のインポート例外・モデル取得失敗のOSError)は
    ここで捕捉せず呼び出し元(_g2p)まで伝播させる(環境不備として明確に停止させる。
    _generate_katakana_arpakanaと同じ設計)。一方、モデルのロード後に生成処理自体が送出する
    例外は、他の変換失敗経路(CMUdict未収録・出力不正)と同様に扱えるよう、空文字列
    (_is_valid_katakanaがFalseと判定する値)を返す。
    """
    import torch

    tokenizer, model = _load_katakana_model(on_progress=on_progress)
    try:
        prompt = _PROMPT_TEMPLATE.format(word=word, phonemes=phonemes)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        generated = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return generated.split("\n")[0].strip()
    except Exception:
        return ""


def _generate_katakana_arpakana(word: str, phonemes: str) -> str:
    """ARPAbet発音記号をルールベースでカタカナへ変換する(arpakanaライブラリ。生成モデル・
    GPU不要。method="arpakana"(既定値)選択時に使う)。wordは使わない(音素列だけから決まる)。

    arpakanaライブラリ未導入時のインポート例外はここで捕捉せず呼び出し元(_g2p)まで伝播させる
    (CMUdict未収録・出力不正と異なり、ライブラリ不足は語ごとの変換失敗でなく環境不備のため、
    安全側フォールバックせず明確に停止させる)。一方、変換処理自体が送出する例外(想定外の音素列
    構成など)は、他の変換失敗経路(CMUdict未収録・出力不正)と同様に扱えるよう、空文字列
    (_is_valid_katakanaがFalseと判定する値)を返す。
    """
    from arpakana import arpabet_to_kana

    try:
        return arpabet_to_kana(phonemes)
    except Exception:
        return ""


def _generate_katakana(
    word: str, phonemes: str, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """英単語・発音記号からカタカナを生成する(methodで変換方式を選択する)。

    未知のmethod値は、無言でいずれかの方式へ流さずValueErrorにする(EnglishKatakanaMethodは
    型検査上の制約に過ぎず実行時には任意の文字列を渡せるため、誤字や将来の値追加が意図せず
    tinyllama-katakana-converter(GPU・生成モデル)側で処理されることを防ぐ)。
    """
    if method == "arpakana":
        return _generate_katakana_arpakana(word, phonemes)
    if method == "tinyllama-katakana-converter":
        return _generate_katakana_tinyllama(word, phonemes, on_progress=on_progress)
    raise ValueError(f"未知の変換方式です: {method!r}")


def _to_halfwidth(text: str) -> str:
    """全角ラテン/数字/記号ブロック(Unicode fullwidth)を半角化する(1文字ずつの固定オフセット変換)。"""
    return "".join(
        chr(ord(ch) - _FULLWIDTH_OFFSET) if _FULLWIDTH_LO <= ord(ch) <= _FULLWIDTH_HI else ch
        for ch in text
    )


def _to_fullwidth(text: str) -> str:
    """半角ASCII英字を全角(Unicode fullwidth)へ変換する(_to_halfwidthの逆変換)。"""
    return "".join(chr(ord(ch) + _FULLWIDTH_OFFSET) for ch in text)


def _locate_and_replace(text: str, target_words: list[str], converted: dict[str, str]) -> str:
    """対象語を元テキスト中で特定し、変換に成功した語(convertedに値がある語)だけを置換した
    文字列を返す。

    各対象語は、半角表記(target_wordsはNFKC正規化済みのため常に半角)・全角表記(元テキスト自体が
    全角の場合に対応する)の両方でcursor以降を検索し、両方見つかった場合はより手前(cursorに近い)
    側を採る(元テキスト中に半角表記と全角表記の出現が混在する場合、常に半角を優先すると、cursorの
    直後にある全角の出現を飛び越して後方の半角の出現を誤って拾うため)。位置が特定できた対象語
    (converted に無い語も含む)は次の対象語の検索開始位置(cursor)を前進させる(位置特定できない
    語だけカーソルを進めないと、後続の対象語の検索がその語より前の位置から始まり誤った箇所を
    見つけうるため、変換の成否に関わらず位置特定できた時点でカーソルを進める)。converted に
    値が無い語(CMUdict未収録・生成結果が不正)は置換対象に加えず元のまま残す。まず全対象語の
    (開始位置, 終了位置, 変換後カタカナ)を先に求めてから(この走査中はtextを書き換えない)、
    1回の走査で最終的な文字列を組み立てる。
    """
    spans: list[tuple[int, int, str]] = []
    cursor = 0
    for word in target_words:
        candidates = [
            (index, needle)
            for index, needle in (
                (text.find(word, cursor), word),
                (text.find(_to_fullwidth(word), cursor), _to_fullwidth(word)),
            )
            if index != -1
        ]
        if not candidates:
            continue
        index, needle = min(candidates, key=lambda candidate: candidate[0])
        cursor = index + len(needle)
        katakana = converted.get(word)
        if katakana is not None:
            spans.append((index, index + len(needle), katakana))

    if not spans:
        return text

    result = []
    prev_end = 0
    for start, end, katakana in spans:
        result.append(text[prev_end:start])
        result.append(katakana)
        prev_end = end
    result.append(text[prev_end:])
    return "".join(result)


def release_katakana_model() -> None:
    """変換モデルのプロセス内キャッシュを解放し、GPUのキャッシュ済みメモリを返す。

    変換モデル(1.1Bパラメータ。GPU実行時はfp16で重み約2.2GB)を使い終えた時点で常駐をやめるための解放口。
    次の変換が必要になれば _load_katakana_model が改めてロードする。
    """
    global _katakana_model_cache
    if _katakana_model_cache is None:
        return
    _katakana_model_cache = None
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def uncached_target_words(
    text: str, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD
) -> list[str]:
    """テキスト中の対象語のうち、変換結果がまだ語キャッシュに無いものを出現順(重複なし)で返す。

    呼び出し元が「これから convert_target_words を呼ぶと新たな変換(変換モデルのロードを伴いうる)が
    走るか」を変換前に判定するための照会口。語が返らなければ、後続の変換はすべて語キャッシュに
    当たり、モデルのロードは起きない。
    """
    return [
        word
        for word in dict.fromkeys(_find_target_words(text))
        if (method, word) not in _conversion_cache
    ]


def convert_words(
    words: list[str], method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> None:
    """語のリストをまとめて変換し、結果を語キャッシュへ入れる。

    tinyllama-katakana-converter 方式では、変換後に変換モデルを解放する(まとめて変換する
    呼び出し元は変換モデルの常駐を変換中だけに限定でき、他のGPU常駐モデルとの同時常駐による
    VRAM逼迫を避けられる)。arpakana 方式はモデルを使わないため解放は行わない。on_progress は
    tinyllama-katakana-converter 方式の変換モデルの取得・ロードが実際に発生した区間だけ
    進捗文言を渡して呼ぶ(_load_katakana_model の契約どおり)。
    """
    try:
        for word in words:
            _convert_word(word, method, on_progress=on_progress)
    finally:
        if method == "tinyllama-katakana-converter":
            release_katakana_model()


def convert_target_words(
    text: str, method: EnglishKatakanaMethod = DEFAULT_ENGLISH_KATAKANA_METHOD,
    on_progress: Callable[[str], None] | None = None,
) -> str:
    """テキスト中の対象語(pyopenjtalkが読めない英単語)をカタカナへ変換する。対象語が無ければ
    元のテキストをそのまま返す。methodで変換方式を選択する(既定: arpakana)。on_progress は
    convert_words と同じ契約(tinyllama-katakana-converter 方式の変換モデル取得時のみ呼ぶ)。
    """
    target_words = _find_target_words(text)
    if not target_words:
        return text

    converted: dict[str, str] = {}
    for word in dict.fromkeys(target_words):
        result = _convert_word(word, method, on_progress=on_progress)
        if result is not None:
            converted[word] = result

    return _locate_and_replace(text, target_words, converted)


_conversion_cache: dict[tuple[EnglishKatakanaMethod, str], str | None] = {}


def _convert_word(
    word: str, method: EnglishKatakanaMethod, on_progress: Callable[[str], None] | None = None
) -> str | None:
    """1語をカタカナへ変換する(CMUdict参照→変換→出力妥当性検証)。

    _g2p経由でrecognize()の実行中に同じ語が複数回変換対象になりうる(区間全体の音素密度判定・
    単語単位のG2Pなど、_g2pは同じ内容に対して複数回呼ばれるため)。CMUdict参照・変換はともに
    毎回実行するとコストが大きいため、結果(カタカナ、または変換不可を示すNone)をプロセス内
    キャッシュする(CMUdict・各変換方式はいずれも入力に対して決定論的なため、結果の使い回しは
    安全)。キャッシュキーに変換方式(method)を含めるのは、同じ語でも方式が異なれば結果が
    異なりうるため(arpakanaとtinyllama-katakana-converterの結果を混同しない)。
    """
    cache_key = (method, word)
    if cache_key in _conversion_cache:
        return _conversion_cache[cache_key]

    result = None
    phonemes = _lookup_cmudict_phonemes(word)
    if phonemes is not None:
        katakana = _generate_katakana(word, phonemes, method=method, on_progress=on_progress)
        if _is_valid_katakana(katakana):
            result = katakana
    _conversion_cache[cache_key] = result
    return result
