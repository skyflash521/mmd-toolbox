"""S-1 認識ゲートの参照ラベル写像テスト。

歌唱の実音素記号を a/i/u/e/o(母音)・c(子音)・sil(無音/息)へ写像する規則と、モノフォンラベル
形式(秒単位・HTK 100ns単位)のパーサを、実際の音素記号・時刻値を使って決定論的に検証する。
"""

import pytest

pytestmark = pytest.mark.xfail(reason="impl pending: vocal_analysis.gate", strict=True)


@pytest.mark.parametrize(
    "symbol,expected",
    [("a", "a"), ("i", "i"), ("u", "u"), ("e", "e"), ("o", "o")],
)
def test_reference_symbol_to_category_vowels(symbol, expected):
    from vocal_analysis.gate import reference_symbol_to_category

    assert reference_symbol_to_category(symbol) == expected


@pytest.mark.parametrize("symbol", ["pau", "br"])
def test_reference_symbol_to_category_silence(symbol):
    from vocal_analysis.gate import reference_symbol_to_category

    assert reference_symbol_to_category(symbol) == "sil"


@pytest.mark.parametrize(
    "symbol",
    [
        "b", "by", "ch", "cl", "d", "f", "g", "gy", "h", "hy", "j", "k", "ky",
        "m", "my", "n", "N", "ny", "p", "py", "r", "ry", "s", "sh", "t", "ts",
        "v", "w", "y", "z",
    ],
)
def test_reference_symbol_to_category_consonants(symbol):
    from vocal_analysis.gate import reference_symbol_to_category

    # cl(促音の閉鎖)・N(撥音)・上記以外の全子音は一律 c。
    assert reference_symbol_to_category(symbol) == "c"


def test_reference_symbol_to_category_xx_is_excluded():
    from vocal_analysis.gate import reference_symbol_to_category

    # xx(未定義区間)は母音でも子音でもsilでもなく、採点から除外する対象として None を返す。
    assert reference_symbol_to_category("xx") is None


@pytest.mark.parametrize("symbol", ["sy", "ty", "zy", "q", "I", "U", "O", ""])
def test_reference_symbol_to_category_unknown_symbol_raises(symbol):
    from vocal_analysis.gate import UnknownReferenceSymbolError, reference_symbol_to_category

    # 網羅した記号集合に無い想定外記号はエラーで停止する(黙って捨てない)。
    with pytest.raises(UnknownReferenceSymbolError):
        reference_symbol_to_category(symbol)


def test_parse_seconds_monophone_label_reads_seconds_and_maps_category(tmp_path):
    from vocal_analysis.gate import parse_seconds_monophone_label

    # 秒単位のモノフォンラベルは「開始秒 終了秒 音素」形式。
    label_path = tmp_path / "001.lab"
    label_path.write_text(
        "0.0000000 18.6263777 pau\n"
        "18.6263777 19.0916217 br\n"
        "19.0916217 19.1636238 k\n"
        "19.1636238 19.3223705 i\n",
        encoding="utf-8",
    )

    segments = parse_seconds_monophone_label(label_path)

    assert [s.category for s in segments] == ["sil", "sil", "c", "i"]
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(18.6263777)
    assert segments[-1].start_sec == pytest.approx(19.1636238)
    assert segments[-1].end_sec == pytest.approx(19.3223705)


def test_parse_seconds_monophone_label_skips_blank_lines(tmp_path):
    from vocal_analysis.gate import parse_seconds_monophone_label

    label_path = tmp_path / "001.lab"
    label_path.write_text("0.0 1.0 pau\n\n1.0 2.0 a\n", encoding="utf-8")

    segments = parse_seconds_monophone_label(label_path)

    assert len(segments) == 2


def test_parse_htk100ns_monophone_label_converts_units_to_seconds(tmp_path):
    from vocal_analysis.gate import parse_htk100ns_monophone_label

    # HTK 100ns単位のモノフォンラベルは「開始 終了 音素」形式(時刻は整数)。
    # 35841272 * 1e-7 = 3.5841272 秒。
    label_path = tmp_path / "001.lab"
    label_path.write_text(
        "0 35841272 pau\n35841272 36371876 m\n36371876 38893424 a\n",
        encoding="utf-8",
    )

    segments = parse_htk100ns_monophone_label(label_path)

    assert [s.category for s in segments] == ["sil", "c", "a"]
    assert segments[0].start_sec == pytest.approx(0.0)
    assert segments[0].end_sec == pytest.approx(3.5841272)
    assert segments[1].start_sec == pytest.approx(3.5841272)
    assert segments[1].end_sec == pytest.approx(3.6371876)


def test_parse_seconds_monophone_label_xx_category_is_none(tmp_path):
    from vocal_analysis.gate import parse_seconds_monophone_label

    label_path = tmp_path / "001.lab"
    label_path.write_text("0.0 1.0 xx\n", encoding="utf-8")

    segments = parse_seconds_monophone_label(label_path)

    assert segments[0].category is None


def test_parse_seconds_monophone_label_unknown_symbol_raises(tmp_path):
    from vocal_analysis.gate import UnknownReferenceSymbolError, parse_seconds_monophone_label

    label_path = tmp_path / "001.lab"
    label_path.write_text("0.0 1.0 zy\n", encoding="utf-8")

    with pytest.raises(UnknownReferenceSymbolError):
        parse_seconds_monophone_label(label_path)


def test_parse_seconds_monophone_label_delegates_to_reference_symbol_to_category(tmp_path, monkeypatch):
    from vocal_analysis import gate as gate_module

    # パーサが記号写像を独自実装(重複)せず reference_symbol_to_category を呼び出すことを、
    # 差し替えた戻り値がそのままセグメントへ伝播するかで直接検証する(単に同じ例外が出ることの
    # 確認だけでは、パーサ側に独自の写像表を重複実装していても通ってしまうため不十分)。
    calls = []

    def fake_reference_symbol_to_category(symbol):
        calls.append(symbol)
        return "SENTINEL"

    monkeypatch.setattr(gate_module, "reference_symbol_to_category", fake_reference_symbol_to_category)

    label_path = tmp_path / "001.lab"
    label_path.write_text("0.0 1.0 k\n", encoding="utf-8")

    segments = gate_module.parse_seconds_monophone_label(label_path)

    assert calls == ["k"]
    assert segments[0].category == "SENTINEL"


def test_parse_htk100ns_monophone_label_malformed_line_raises_clear_error(tmp_path):
    from vocal_analysis.gate import MonophoneLabelFormatError, parse_htk100ns_monophone_label

    label_path = tmp_path / "001.lab"
    label_path.write_text("0 35841272\n", encoding="utf-8")  # 音素記号が欠落

    with pytest.raises(MonophoneLabelFormatError, match="001.lab"):
        parse_htk100ns_monophone_label(label_path)


def test_parse_htk100ns_monophone_label_delegates_to_reference_symbol_to_category(tmp_path, monkeypatch):
    from vocal_analysis import gate as gate_module

    # HTK 100ns単位のパーサも秒単位のパーサと同じく reference_symbol_to_category に委譲することを、
    # 差し替えた戻り値がそのままセグメントへ伝播するかで検証する(このパーサ自身が独自の写像表を
    # 重複実装していないことの確認)。
    calls = []

    def fake_reference_symbol_to_category(symbol):
        calls.append(symbol)
        return "SENTINEL"

    monkeypatch.setattr(gate_module, "reference_symbol_to_category", fake_reference_symbol_to_category)

    label_path = tmp_path / "001.lab"
    label_path.write_text("0 10000000 k\n", encoding="utf-8")

    segments = gate_module.parse_htk100ns_monophone_label(label_path)

    assert calls == ["k"]
    assert segments[0].category == "SENTINEL"
