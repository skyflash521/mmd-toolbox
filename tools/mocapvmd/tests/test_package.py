import mocapvmd


def test_package_is_regular_package_with_init_file():
    assert mocapvmd.__file__ is not None
