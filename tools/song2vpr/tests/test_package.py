import re

import song2vpr


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", song2vpr.__version__)

