from mocapvmd.default_profile import (
    build_default_model,
)


def test_name_to_index_points_back_to_every_bone():
    model = build_default_model()
    n = len(model.bones)
    for name, idx in model.name_to_index.items():
        assert 0 <= idx < n
        assert model.bones[idx].name == name
    for b in model.bones:
        assert b.name in model.name_to_index


def test_hierarchy_is_self_contained():
    model = build_default_model()
    n = len(model.bones)
    roots = 0
    for b in model.bones:
        if b.parent is None:
            roots += 1
        else:
            assert 0 <= b.parent < n
    assert roots >= 1
