import pytest


@pytest.fixture
def real_cmudict():
    from vocal_analysis.english_katakana import ensure_cmudict

    ensure_cmudict()
