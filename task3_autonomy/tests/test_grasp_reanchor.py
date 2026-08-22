# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""REV12 T5: candidates must survive a live scene
(task3_autonomy/grasp_reanchor.py). No Isaac, no GPU."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task3_autonomy.grasp_reanchor import (  # noqa: E402
    FLOOR_DROP_Z_M,
    MAX_SANE_DELTA_M,
    ROUTINE_DRIFT_XY_M,
    ReanchorAction,
    reanchor_candidate,
)

_RECORDED_POSE = (-4.0, -1.5, 0.746)
_CANDIDATE_POS = (-4.02, -1.48, 0.83)  # some grasp point near the object


def test_unchanged_object_translates_by_zero():
    result = reanchor_candidate(_CANDIDATE_POS, _RECORDED_POSE, _RECORDED_POSE)
    assert result.action is ReanchorAction.PROCEED
    assert result.translated_position == _CANDIDATE_POS
    assert result.delta_xy_m == 0.0
    assert result.delta_z_m == 0.0
    assert result.routine_drift is True


def test_routine_drift_translates_rigidly():
    live_pose = (
        _RECORDED_POSE[0] + 0.03,
        _RECORDED_POSE[1] - 0.02,
        _RECORDED_POSE[2],
    )
    result = reanchor_candidate(_CANDIDATE_POS, _RECORDED_POSE, live_pose)
    assert result.action is ReanchorAction.PROCEED
    assert result.routine_drift is True
    expected = (
        _CANDIDATE_POS[0] + 0.03,
        _CANDIDATE_POS[1] - 0.02,
        _CANDIDATE_POS[2],
    )
    for got, want in zip(result.translated_position, expected):
        assert abs(got - want) < 1e-9


def test_large_but_sane_drift_still_translates_and_is_flagged_non_routine():
    live_pose = (_RECORDED_POSE[0] + 0.3, _RECORDED_POSE[1], _RECORDED_POSE[2])
    result = reanchor_candidate(_CANDIDATE_POS, _RECORDED_POSE, live_pose)
    assert result.action is ReanchorAction.PROCEED
    assert result.routine_drift is False
    assert result.delta_xy_m > ROUTINE_DRIFT_XY_M


def test_floor_drop_abandons():
    live_pose = (
        _RECORDED_POSE[0],
        _RECORDED_POSE[1],
        _RECORDED_POSE[2] - FLOOR_DROP_Z_M - 0.01,
    )
    result = reanchor_candidate(_CANDIDATE_POS, _RECORDED_POSE, live_pose)
    assert result.action is ReanchorAction.ABANDON_FLOOR
    assert result.translated_position is None


def test_floor_drop_boundary_does_not_abandon():
    """Exactly at the threshold (not past it) should still proceed --
    guards against an off-by-one on the strict '>' comparison."""
    live_pose = (
        _RECORDED_POSE[0],
        _RECORDED_POSE[1],
        _RECORDED_POSE[2] - FLOOR_DROP_Z_M,
    )
    result = reanchor_candidate(_CANDIDATE_POS, _RECORDED_POSE, live_pose)
    assert result.action is ReanchorAction.PROCEED


def test_absurd_jump_abandons_not_translates():
    """A jump larger than MAX_SANE_DELTA_M but still inside the coarse
    scene bounds -- must be caught by the jump check, not translated. The
    223m spoon2 fling itself lands outside scene bounds entirely and is
    covered by test_out_of_scene_bounds_rejected instead (bounds are
    checked first -- see the module docstring)."""
    live_pose = (
        _RECORDED_POSE[0] + 1.0 + MAX_SANE_DELTA_M,
        _RECORDED_POSE[1],
        _RECORDED_POSE[2],
    )
    result = reanchor_candidate(_CANDIDATE_POS, _RECORDED_POSE, live_pose)
    assert result.action is ReanchorAction.ABANDON_JUMP
    assert result.translated_position is None
    assert result.delta_xy_m > MAX_SANE_DELTA_M


def test_out_of_scene_bounds_rejected():
    live_pose = (500.0, 500.0, 500.0)
    result = reanchor_candidate(_CANDIDATE_POS, _RECORDED_POSE, live_pose)
    assert result.action is ReanchorAction.ABANDON_OUT_OF_BOUNDS
    assert result.translated_position is None


def test_out_of_bounds_checked_before_floor_drop():
    """An out-of-bounds AND low-z live pose must be reported as
    out-of-bounds, not misdiagnosed as a floor drop."""
    live_pose = (500.0, 500.0, -10.0)
    result = reanchor_candidate(_CANDIDATE_POS, _RECORDED_POSE, live_pose)
    assert result.action is ReanchorAction.ABANDON_OUT_OF_BOUNDS


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
