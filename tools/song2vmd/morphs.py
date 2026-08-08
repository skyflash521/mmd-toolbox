"""song2vmd モーフ生成・VMD組み立て接続。

口形イベント列(MouthEvent)と生成パラメータを共有モジュール lipsync へ渡してモーフキーを生成し、
モーフキーのみを持つ VmdDocument を組み立てる。ボーン・カメラ・照明・セルフ影・IKの各セクションは
空で出力する。VMDへの書き出しはこのモジュールでは行わない(呼び出し側が
`vmd.io.write_file` を使う)。
"""

from lipsync import generate_morph_keys
from vmd import VmdDocument, ensure_frame0_neutral_keys, normalize

_MODEL_NAME_BYTES = 20


def build_vmd_document(events, params, model_name):
    """口形イベント列からモーフキーのみのVmdDocumentを組み立てる。

    model_name は cp932 で表現できる20バイト以内であることを呼び出し側(cli.py の引数検証)が
    保証済みの前提で、ここでは20バイトへパディングするだけを行う。
    """
    morph_keys = generate_morph_keys(events, params)
    document = VmdDocument(
        model_name_raw=model_name.encode("cp932").ljust(_MODEL_NAME_BYTES, b"\x00"),
        morph=morph_keys,
    )
    document = ensure_frame0_neutral_keys(document, sections=("morph",))
    document, _warnings = normalize(document, sections=["morph"])
    return document
