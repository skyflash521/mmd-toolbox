import re
from collections.abc import Callable

from .config import KANA_PROMPT

HALLUCINATION_PHONEME_RATE = 20.0
ECHO_FRAGMENT_PREFIX_LEN = 5
REPEAT_MIN_COUNT = 3
REPEAT_MIN_TOTAL_CHARS = 8
REPEAT_NORMALIZE_MORA_RATE = 3.5
MORA_G2P_SYMBOLS = frozenset({"a", "i", "u", "e", "o", "I", "U", "N"})


def normalize_sentence(sentence: str) -> str:
    return re.sub(r"[\s、。,.]", "", sentence)


_PROMPT_SENTENCES = frozenset(
    normalized for normalized in (normalize_sentence(part) for part in KANA_PROMPT.split("。")) if normalized
)


def is_hallucinated_phoneme_density(phoneme_count: int, duration_sec: float) -> bool:
    if duration_sec <= 0:
        return False
    return phoneme_count / duration_sec > HALLUCINATION_PHONEME_RATE


def _common_prefix_len(a: str, b: str) -> int:
    n = 0
    for ca, cb in zip(a, b, strict=False):
        if ca != cb:
            break
        n += 1
    return n


def strip_prompt_echo(text: str) -> tuple[str, bool]:
    """戻り値は (除去後のテキスト, 1文以上除去したか)。"""
    parts = re.split(r"(?<=。)", text)
    kept = [part for part in parts if normalize_sentence(part) not in _PROMPT_SENTENCES]
    removed = len(kept) < len(parts)
    if removed and kept:
        tail = normalize_sentence(kept[-1])
        if tail and any(_common_prefix_len(tail, prompt) >= ECHO_FRAGMENT_PREFIX_LEN
                        for prompt in _PROMPT_SENTENCES):
            kept.pop()
    return "".join(kept), removed


def find_suffix_repetition(text: str) -> tuple[str, int, str] | None:
    """戻り値は (反復の単位, 繰り返し数, 反復を除いた先頭部)。末尾に反復が無ければ None。"""
    best: tuple[str, int, str] | None = None
    n = len(text)
    for unit_len in range(1, n // REPEAT_MIN_COUNT + 1):
        unit = text[n - unit_len:]
        count = 1
        while n - (count + 1) * unit_len >= 0 and text[n - (count + 1) * unit_len: n - count * unit_len] == unit:
            count += 1
        total = unit_len * count
        if count >= REPEAT_MIN_COUNT and total >= REPEAT_MIN_TOTAL_CHARS:
            if best is None or total > len(best[0]) * best[1]:
                best = (unit, count, text[: n - total])
    return best


def resolve_transcription(
    duration_sec: float,
    text: str,
    words: list[tuple[str, float, float]] | None,
    *,
    g2p: Callable[[str], list[str]],
    retranscribe: Callable[[], str] | None,
) -> tuple[str, list[tuple[str, float, float]] | None]:
    """戻り値は採用する (テキスト, 単語タイムスタンプ)。テキストを変えたときは単語タイムスタンプを None にする。
    g2p は音素密度の判定に使う。retranscribe が None なら再書き起こしはしない。"""
    text2, echo_removed = strip_prompt_echo(text)
    if echo_removed:
        words = None

    def phoneme_density(candidate: str) -> float:
        if not candidate.strip() or duration_sec <= 0:
            return 0.0
        return len(g2p(candidate)) / duration_sec

    def needs_retry(candidate: str) -> bool:
        if not candidate.strip():
            return echo_removed
        return phoneme_density(candidate) > HALLUCINATION_PHONEME_RATE

    if retranscribe is not None and needs_retry(text2):
        retry_text, _retry_echo_removed = strip_prompt_echo(retranscribe().strip())
        text2, words = retry_text, None

    if text2.strip() and phoneme_density(text2) > HALLUCINATION_PHONEME_RATE:
        repetition = find_suffix_repetition(text2)
        if repetition is not None:
            unit, count, head = repetition
            rescued = text2
            if head.strip():
                rescued = head
            else:
                unit_moras = max(1, sum(1 for symbol in g2p(unit) if symbol in MORA_G2P_SYMBOLS))
                normalized = max(1, round(duration_sec * REPEAT_NORMALIZE_MORA_RATE / unit_moras))
                if normalized < count:
                    rescued = unit * normalized
            if rescued != text2:
                text2 = rescued
                words = None

    return text2, words
