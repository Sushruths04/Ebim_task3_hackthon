# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU tests for R9 T2's rank-mode logic in
scripts/task3/ik_feasibility_sweep.py: the stance-grid generator and
`_solve_pose`'s aggregation (FK-residual lookup, success/failure
bookkeeping), both pure Python once the real solver is stubbed out. The
real Lula solve itself (`_build_solver`, `run_rank` end to end) needs the
Isaac container and is exercised there by
`scripts/task3/gate_t2_bowl2_replay.py` -- GATE T2 -- never here, matching
this project's established split (see test_perception_targets.py's own
docstring for the same rule applied to perception_targets.py).

Imported by file path (importlib), not `import ik_feasibility_sweep`:
scripts/task3/ is not a package, and this keeps the import independent of
sys.path ordering elsewhere in the suite.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def _load_module():
    path = REPO_ROOT / "scripts" / "task3" / "ik_feasibility_sweep.py"
    spec = importlib.util.spec_from_file_location(
        "ik_feasibility_sweep_under_test", path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ranker = _load_module()


def test_rank_stance_grid_includes_extra_stances_additively():
    grid = ranker._rank_stance_grid(
        (-3.9, -0.8), extra_stances=[((-3.7676, -0.816), 3.142)]
    )
    assert ((-3.7676, -0.816), 3.142) in grid
    assert len(grid) == 2 * 4 + 1  # 2 approaches x 4 radius offsets + 1 extra


def test_rank_stance_grid_without_extras_is_just_the_generated_grid():
    grid = ranker._rank_stance_grid((-3.9, -0.8))
    assert len(grid) == 2 * 4


def test_rank_stance_grid_radius_never_collapses_below_floor():
    from task3_autonomy.navigation import STANCE_REACH_RADIUS_M

    assert STANCE_REACH_RADIUS_M > 0.30  # sanity: the floor is a real clamp
    grid = ranker._rank_stance_grid((0.0, 0.0))
    dists = [math.hypot(*xy) for xy, _yaw in grid]
    assert min(dists) >= 0.30 - 1e-6


class _FakeArmSolver:
    def __init__(self, position):
        self._position = np.asarray(position, dtype=np.float64)

    def compute_end_effector_pose(self):
        return self._position, np.eye(3)


class _FakeResult:
    def __init__(self, left_succeeded, right_succeeded):
        self.left_succeeded = left_succeeded
        self.right_succeeded = right_succeeded
        self.left = {"left_fr3v2_joint1": 0.1} if left_succeeded else {}
        self.right = {"right_fr3v2_joint1": 0.1} if right_succeeded else {}


class _FakeDualSolver:
    """Stubs `DualArmLulaIK.solve()` plus the private per-arm solver
    handles `_solve_pose` reaches into -- exercises the real aggregation
    code path (FK residual lookup, `joint_names.index`) without needing
    Isaac."""

    def __init__(self, left_succeeded, right_succeeded, achieved_position):
        self._left_solver = _FakeArmSolver(achieved_position)
        self._right_solver = _FakeArmSolver(achieved_position)
        self._left_succeeded = left_succeeded
        self._right_succeeded = right_succeeded

    def solve(self, **kwargs):
        return _FakeResult(self._left_succeeded, self._right_succeeded)


_JOINT_NAMES = ("left_fr3v2_joint1", "right_fr3v2_joint1")


def _solve(solver, target, achieved=None, spine=0.0):
    # spine=0.0 by default: with no spine offset, `_solve_pose`'s target
    # compensation (see test_solve_pose_compensates_target_by_spine_
    # offset below) is a no-op, so these tests can compare achieved vs.
    # target directly without also modeling the compensation.
    seed = np.zeros(len(_JOINT_NAMES))
    return ranker._solve_pose(
        solver,
        seed,
        _JOINT_NAMES,
        {},
        target=target,
        orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        base_xy=(0.0, 0.0),
        base_yaw=0.0,
        spine=spine,
    )


def test_solve_pose_reports_zero_residual_on_exact_convergence():
    target = (1.0, 2.0, 3.0)
    out = _solve(_FakeDualSolver(True, True, target), target)
    assert out["left"]["succeeded"] is True
    assert out["left"]["position_error_m"] == 0.0
    assert out["right"]["succeeded"] is True
    assert out["right"]["position_error_m"] == 0.0


def test_solve_pose_reports_sentinel_error_on_failure():
    out = _solve(
        _FakeDualSolver(False, False, (0.0, 0.0, 0.0)), (5.0, 5.0, 5.0)
    )
    assert out["left"]["succeeded"] is False
    assert (
        out["left"]["position_error_m"] == ranker.UNREACHABLE_POSITION_ERROR_M
    )
    assert out["right"]["succeeded"] is False
    assert (
        out["right"]["position_error_m"] == ranker.UNREACHABLE_POSITION_ERROR_M
    )


def test_solve_pose_reports_nonzero_residual_when_solved_short_of_target():
    target = (1.0, 0.0, 0.0)
    achieved = (0.9, 0.0, 0.0)  # "solved" but 10cm short of the real target
    out = _solve(_FakeDualSolver(True, False, achieved), target, achieved)
    assert out["left"]["succeeded"] is True
    assert out["left"]["position_error_m"] == 0.1
    assert out["right"]["succeeded"] is False
    assert (
        out["right"]["position_error_m"] == ranker.UNREACHABLE_POSITION_ERROR_M
    )


def test_solve_pose_asymmetric_success_matches_the_real_bowl2_shape():
    """Not a replay of the real numbers (that needs the real solver --
    GATE T2, GPU-side) -- just confirms the aggregation code path can
    represent the documented real shape (plans/VM_B_LOG.md 2026-08-02):
    one side converges near-exactly, the other fails outright."""
    target = (-4.185, -1.692, 0.877)
    out = _solve(_FakeDualSolver(True, False, target), target)
    assert out["left"]["succeeded"] and out["left"]["position_error_m"] < 0.01
    assert not out["right"]["succeeded"]
    assert (
        out["right"]["position_error_m"] == ranker.UNREACHABLE_POSITION_ERROR_M
    )


def test_solve_pose_compensates_target_by_spine_offset():
    """Regression test for a real bug caught live (first GATE T2 replay,
    plans/SYNC.md): `DualArmLulaIK.solve()` internally solves each arm
    against `target - spine_offset_world` (dual_arm_lula.py's
    `_solve_arm`, since the Lula YAML fixes the vertical spine joint at
    zero), so a fake solver that "achieves" the target EXACTLY at the
    COMPENSATED position (not the raw target) must report ~zero residual
    -- not `spine` (an earlier version of this function compared against
    the raw, uncompensated target and reported `ik_margin == -spine` on
    every solved side, a real observed number, not synthetic)."""
    spine = 0.4236
    target = (1.0, 2.0, 3.0)
    achieved = (1.0, 2.0, 3.0 - spine)  # base has no roll/pitch: offset is +Z
    out = _solve(
        _FakeDualSolver(True, True, achieved), target, achieved, spine=spine
    )
    assert out["left"]["succeeded"] is True
    assert out["left"]["position_error_m"] < 1e-6
    assert out["left"]["position_error_m"] != spine
