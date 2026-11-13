import re

import song2vmd


def test_version_is_semver():
    assert re.fullmatch(r"\d+\.\d+\.\d+", song2vmd.__version__)

