from lipsync import ApertureClass, ConsonantClass, MouthEvent, MouthShape

from .phonemes import PhonemeCategory, aperture_class, categorize, consonant_class, vowel_shape

_BILABIAL_NOMINAL_FRAMES = 3.0
_BILABIAL_MAX_SHARE_OF_NOTE = 0.5


def note_mouth_events(
    phonemes: list[str], start: float, end: float, use_n_morph: bool = False
) -> list[MouthEvent] | None:
    """口形が直前の音符に依存して決まる音符では None。"""
    vowels = [vowel_shape(p) for p in phonemes if categorize(p) is PhonemeCategory.VOWEL]
    if vowels:
        return _vowel_events(phonemes, start, end, vowels)
    if any(categorize(p) is PhonemeCategory.GEMINATE_STOP for p in phonemes):
        return [MouthEvent(MouthShape.SILENCE, start, end)]
    if phonemes and all(categorize(p) is PhonemeCategory.MORAIC_NASAL for p in phonemes):
        shape = MouthShape.N if use_n_morph else MouthShape.SILENCE
        return [MouthEvent(shape, start, end)]
    return None


def _onset_consonant_class(phonemes: list[str], first_vowel_index: int) -> ConsonantClass:
    onset = [
        p for p in phonemes[:first_vowel_index]
        if categorize(p) is PhonemeCategory.OTHER
    ]
    classes = [consonant_class(p) for p in onset]
    if ConsonantClass.ROUNDED in classes:
        return ConsonantClass.ROUNDED
    if ConsonantClass.SPREAD in classes:
        return ConsonantClass.SPREAD
    if classes:
        return ConsonantClass.NEUTRAL
    return ConsonantClass.NONE


def _onset_aperture_class(phonemes: list[str], first_vowel_index: int) -> ApertureClass:
    onset = phonemes[:first_vowel_index]
    last_bilabial = -1
    for i, p in enumerate(onset):
        if categorize(p) is PhonemeCategory.BILABIAL:
            last_bilabial = i
    after_last_bilabial = [
        p for p in onset[last_bilabial + 1:]
        if categorize(p) is PhonemeCategory.OTHER
    ]
    classes = [aperture_class(p) for p in after_last_bilabial]
    if ApertureClass.FIRM_CLOSURE in classes:
        return ApertureClass.FIRM_CLOSURE
    if ApertureClass.NARROW_CHANNEL in classes:
        return ApertureClass.NARROW_CHANNEL
    if ApertureClass.SLIGHT_CLOSURE in classes:
        return ApertureClass.SLIGHT_CLOSURE
    return ApertureClass.NONE


def _vowel_events(
    phonemes: list[str], start: float, end: float, vowels: list[MouthShape]
) -> list[MouthEvent]:
    events: list[MouthEvent] = []
    bilabial_frames = 0.0
    if categorize(phonemes[0]) is PhonemeCategory.BILABIAL:
        bilabial_frames = min(
            _BILABIAL_NOMINAL_FRAMES, (end - start) * _BILABIAL_MAX_SHARE_OF_NOTE
        )
        events.append(MouthEvent(MouthShape.BILABIAL, start, start + bilabial_frames))

    first_vowel_index = next(
        i for i, p in enumerate(phonemes) if categorize(p) is PhonemeCategory.VOWEL
    )
    onset_class = _onset_consonant_class(phonemes, first_vowel_index)
    onset_aperture = _onset_aperture_class(phonemes, first_vowel_index)

    vowel_region_start = start + bilabial_frames
    width = (end - vowel_region_start) / len(vowels)
    last = len(vowels) - 1
    for i, shape in enumerate(vowels):
        seg_start = vowel_region_start + i * width
        seg_end = end if i == last else vowel_region_start + (i + 1) * width
        cc = onset_class if i == 0 else ConsonantClass.NONE
        ac = onset_aperture if i == 0 else ApertureClass.NONE
        events.append(MouthEvent(shape, seg_start, seg_end, consonant_class=cc, aperture_class=ac))
    return events
