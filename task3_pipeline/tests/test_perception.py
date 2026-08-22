# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU unit tests for the M1 perception boundary (task3_pipeline/
perception.py).

Run: python -m pytest task3_pipeline/tests/test_perception.py -q
"""

from __future__ import annotations

import pytest

from task3_pipeline.perception import (
    GroundTruthPerception,
    PerceivedPose,
    Perception,
    pose_error,
)
from task3_pipeline.world import MockWorld


def test_ground_truth_perception_matches_world_adapter_reads():
    world = MockWorld(seed=0)
    perception = GroundTruthPerception()

    poses = perception.perceive(world, ("cup", "plate2"))

    assert set(poses) == {"cup", "plate2"}
    for name, perceived in poses.items():
        x, y = world.object_xy(name)
        z = world.object_z(name)
        assert perceived.position == (x, y, z)
        assert perceived.confidence == 1.0
        assert perceived.visible is True


def test_ground_truth_perception_satisfies_the_perception_protocol():
    # GroundTruthPerception must be swappable for the future real backend
    # without any caller-side change -- that is the entire point of the
    # boundary (ACTIVE_BRIEF sec 2 rule 3).
    assert isinstance(GroundTruthPerception(), Perception)


def test_pose_error_is_zero_for_a_perfect_match():
    perceived = PerceivedPose(
        position=(1.0, 2.0, 0.5),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        confidence=1.0,
        visible=True,
    )
    assert pose_error(perceived, (1.0, 2.0, 0.5)) == pytest.approx(0.0)


def test_pose_error_measures_real_displacement():
    perceived = PerceivedPose(
        position=(1.0, 2.0, 0.5),
        orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
        confidence=1.0,
        visible=True,
    )
    # 3-4-5 triangle in x/y, z unchanged -> error is exactly 5.
    assert pose_error(perceived, (4.0, 6.0, 0.5)) == pytest.approx(5.0)


@pytest.mark.parametrize("bad_confidence", [-0.01, 1.01])
def test_perceived_pose_rejects_confidence_outside_unit_range(bad_confidence):
    with pytest.raises(ValueError):
        PerceivedPose(
            position=(0.0, 0.0, 0.0),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            confidence=bad_confidence,
            visible=True,
        )


def test_perceived_pose_rejects_invisible_with_nonzero_confidence():
    # An honest "not detected" must report confidence 0.0 -- this is the
    # same failure class ACTIVE_BRIEF sec 3 warns about ("a gate measuring
    # something other than what it claims to"): a dropped detection that
    # still carries a nonzero confidence would silently corrupt M1's
    # measured dropout rate and M3's training noise profile.
    with pytest.raises(ValueError):
        PerceivedPose(
            position=(0.0, 0.0, 0.0),
            orientation_wxyz=(1.0, 0.0, 0.0, 0.0),
            confidence=0.4,
            visible=False,
        )
