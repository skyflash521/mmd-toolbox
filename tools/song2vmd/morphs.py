from lipsync import generate_morph_keys
from vmd import VmdDocument, ensure_frame0_neutral_keys, normalize

_VMD_MODEL_NAME_BYTES = 20


def build_vmd_document(events, params, model_name):
    """model_name は cp932 で 20 バイト以内であること(超過は検査しない)。"""
    morph_keys = generate_morph_keys(events, params)
    document = VmdDocument(
        model_name_raw=model_name.encode("cp932").ljust(_VMD_MODEL_NAME_BYTES, b"\x00"),
        morph=morph_keys,
    )
    document = ensure_frame0_neutral_keys(document, sections=("morph",))
    document, _warnings = normalize(document, sections=["morph"])
    return document
