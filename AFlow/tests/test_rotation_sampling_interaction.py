from scripts.describe_rotation_sampling_interaction import interaction


def test_additive_and_nonadditive_effects():
    assert abs(interaction(.7,.71,.72,.73))<1e-12
    assert abs(interaction(.7,.71,.72,.715)+.015)<1e-12
    assert interaction(0,1,1,1)==-1
