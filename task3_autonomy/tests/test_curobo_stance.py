# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""S1 (plans/SPRINT_30H_2026-08-02.md sec 1, step 3): CPU-only check of
`CuroboStanceSearch.stance_for()`'s candidate-selection rule, and S2 sub-fix
1's reach-margin term added on top of it.

Runs against the REAL selection code (`_select_closest_reachable`,
extracted verbatim from the inline loop in `stance_for()`, same
`range(1, n_total)` / `practical_ok` / `_segment_clears_island` semantics)
-- not a reimplementation. No torch/cuRobo/GPU import required: importing
`task3_autonomy.curobo_stance` at module scope never touches those (they
are imported lazily inside `CuroboStanceSearch.__init__`).
"""

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from task3_autonomy.curobo_stance import (  # noqa: E402
    _RADII_MAX_M,
    _RADII_MIN_M,
    _select_closest_reachable,
)

# S1's own live GPU finding (GATE S1 CONFIRMED, s1_gpu_confirm_r2): the
# ~0.855m arm reach limit, measured, not a spec value invented for this test.
_MEASURED_REACH_LIMIT_M = 0.855

# Object placed far from KITCHEN_ISLAND_BBOX (x: -4.51..-3.77, y: -2.47..-1.22)
# so every candidate/robot position below trivially clears the island and
# `_select_closest_reachable`'s path-clear check never rejects anything --
# isolates the ranking behavior under test.
_OBJECT_XY = (10.0, 10.0)
_RADII = [_RADII_MIN_M, 0.60, 0.70, _RADII_MAX_M]
_N_ANGLES = 12  # every 30 degrees
_LIVE_XY = (
    _OBJECT_XY[0] + 5.0,
    _OBJECT_XY[1] + 5.0,
)  # 5m away, well outside _RADII_MAX_M


def _annulus_candidates(radii=_RADII, n_angles=_N_ANGLES):
    """(xy list, radius-per-candidate list) on concentric rings around
    `_OBJECT_XY`, index-aligned. Mirrors `stance_for()`'s own candidate
    generation (radius x yaw grid) closely enough for this test's purpose."""
    xy = []
    radius_of = []
    for r in radii:
        for k in range(n_angles):
            theta = 2.0 * math.pi * k / n_angles
            xy.append(
                (
                    _OBJECT_XY[0] + r * math.cos(theta),
                    _OBJECT_XY[1] + r * math.sin(theta),
                )
            )
            radius_of.append(r)
    return xy, radius_of


def test_pre_S2_ranking_preferred_outer_radius_when_robot_is_far_outside():
    """Historical record (S1, GATE CONFIRMED): with margin_k=0 -- pure
    "closest to the robot", the rule this codebase actually ran with
    before S2 -- selection always lands on the OUTERMOST ring when every
    radius is equally practical_ok and the robot approaches from
    outside. This is what S1's live GPU run
    (`outputs/task3_pipeline/s1_gpu_confirm_r2`) observed:
    chosen_radius_m == min_accepted_radius_m == _RADII_MAX_M every time."""
    cand_xy, radius_of = _annulus_candidates()
    all_xy = [(0.0, 0.0)] + cand_xy  # index 0 is the unused fallback slot
    practical_ok = [True] * len(all_xy)

    best_idx, best_dist, n_path_blocked = _select_closest_reachable(
        len(all_xy), practical_ok, all_xy, _LIVE_XY, _OBJECT_XY, margin_k=0.0
    )

    assert best_idx is not None
    assert n_path_blocked == 0
    chosen_radius = radius_of[best_idx - 1]
    assert chosen_radius == _RADII_MAX_M, (
        f"selection picked radius {chosen_radius}, expected the outer "
        f"annulus max {_RADII_MAX_M} with margin_k=0 (pure closest-to-robot)"
    )


def test_S2_margin_term_prefers_inner_radius_when_available():
    """S2 sub-fix 1: with the production default margin_k=1.0, the SAME
    scenario as the historical test above now prefers the INNERMOST
    radius -- a reach-margin term outweighs the (here, radius-independent)
    drive-distance term whenever a smaller radius is available at all."""
    cand_xy, radius_of = _annulus_candidates()
    all_xy = [(0.0, 0.0)] + cand_xy
    practical_ok = [True] * len(all_xy)

    best_idx, best_dist, n_path_blocked = _select_closest_reachable(
        len(all_xy), practical_ok, all_xy, _LIVE_XY, _OBJECT_XY
    )

    assert best_idx is not None
    chosen_radius = radius_of[best_idx - 1]
    assert chosen_radius == _RADII_MIN_M, (
        f"selection picked radius {chosen_radius}, expected the inner "
        f"annulus min {_RADII_MIN_M} with the default reach-margin term"
    )


def test_S2_margin_term_still_trades_off_against_drive_distance():
    """The margin term should not be absolute -- a MUCH closer outer-ring
    candidate can still beat a far-away inner-ring one, since the score is
    a sum, not a lexicographic radius-first ordering. Construct a case
    where the only inner-radius (0.45m) candidate sits far from the robot
    and an outer-radius (0.85m) candidate sits almost at the robot's feet;
    the outer one should win (dist term dominates)."""
    object_xy = (0.0, 0.0)
    live_xy = (0.86, 0.0)  # just outside the outer ring, on the +X axis
    far_inner = (-0.45, 0.0)  # radius 0.45m from object, ~1.31m from robot
    close_outer = (0.85, 0.0)  # radius 0.85m from object, ~0.01m from robot
    all_xy = [(0.0, 0.0), far_inner, close_outer]
    practical_ok = [True, True, True]

    best_idx, best_dist, n_path_blocked = _select_closest_reachable(
        len(all_xy), practical_ok, all_xy, live_xy, object_xy, obstacles=[]
    )

    assert best_idx == 2, "a much closer outer-ring candidate should still win"


def test_selection_falls_back_to_inner_radius_when_outer_is_unreachable():
    """Sanity check: if only the inner rings are practical_ok (e.g. the
    outer ring is geometrically infeasible for this object), selection
    must still work and pick from what's actually available."""
    cand_xy, radius_of = _annulus_candidates()
    all_xy = [(0.0, 0.0)] + cand_xy
    practical_ok = [radius_of[i] < 0.65 for i in range(len(cand_xy))]
    practical_ok = [
        True
    ] + practical_ok  # index 0 unused, never selected anyway

    best_idx, best_dist, n_path_blocked = _select_closest_reachable(
        len(all_xy), practical_ok, all_xy, _LIVE_XY, _OBJECT_XY
    )

    assert best_idx is not None
    chosen_radius = radius_of[best_idx - 1]
    assert chosen_radius < 0.65


def test_S2_subfix2_radii_max_leaves_real_margin_against_reach_ceiling():
    """S2 sub-fix 2 (P4): the annulus's own outer edge, not just the
    ranking term on top of it, must sit meaningfully inside the measured
    ~0.855m reach ceiling -- the old 0.85m value left only ~5mm, which S1
    showed was the actual singularity boundary in practice."""
    assert _RADII_MAX_M < _MEASURED_REACH_LIMIT_M
    margin_m = _MEASURED_REACH_LIMIT_M - _RADII_MAX_M
    assert margin_m >= 0.05, (
        f"_RADII_MAX_M={_RADII_MAX_M} leaves only {margin_m:.3f}m of margin "
        f"against the measured {_MEASURED_REACH_LIMIT_M}m reach ceiling"
    )


def test_selection_returns_none_when_nothing_is_practical_ok():
    cand_xy, _ = _annulus_candidates()
    all_xy = [(0.0, 0.0)] + cand_xy
    practical_ok = [False] * len(all_xy)

    best_idx, best_dist, n_path_blocked = _select_closest_reachable(
        len(all_xy), practical_ok, all_xy, _LIVE_XY, _OBJECT_XY
    )

    assert best_idx is None
    assert best_dist is None


def test_arm_base_pose_for_uses_the_correct_sides_own_offset():
    # Q5 (SYNC 21/25): CuroboStanceSearch used to capture only the RIGHT
    # arm's local offset from the robot root at construction time
    # (curobo_stance.py, originally :313) and every candidate stance was
    # rated against it. This CPU-only test bypasses __init__ (which needs
    # a live Isaac robot + cuRobo/CUDA) and drives the real
    # _arm_base_pose_for() directly with hand-set per-side offsets,
    # proving side="right" and side="left" genuinely read their own
    # stored offset rather than both aliasing to the same one.
    import torch

    from task3_autonomy.curobo_stance import CuroboStanceSearch

    search = object.__new__(CuroboStanceSearch)
    search._torch = torch
    search._root_pos_w = torch.tensor([0.0, 0.0, 0.0])
    search._yaw_quat_correction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    search._offset_pos_local = {
        "right": torch.tensor([0.2, -0.3, 0.9]),
        "left": torch.tensor([0.2, 0.3, 0.9]),
    }
    search._offset_quat_local = {
        "right": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "left": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    }

    xc = torch.tensor([0.0])
    yc = torch.tensor([0.0])
    yawc = torch.tensor([0.0])

    right_pos, _ = search._arm_base_pose_for(xc, yc, yawc, side="right")
    left_pos, _ = search._arm_base_pose_for(xc, yc, yawc, side="left")

    def _round3(t):
        return [round(v, 3) for v in t.squeeze(0).tolist()]

    assert _round3(right_pos) == [0.2, -0.3, 0.9], right_pos
    assert _round3(left_pos) == [0.2, 0.3, 0.9], left_pos
    # The two sides must disagree here (mirrored y offsets) -- if this
    # ever comes back equal, the dict lookup silently collapsed back to
    # one side, which is exactly the bug Q5 fixes.
    assert _round3(right_pos) != _round3(left_pos)


def test_right_base_pose_for_alias_still_matches_the_right_side():
    # Pre-Q5 callers/tests reference _right_base_pose_for by name -- it
    # must still return exactly what side="right" returns, unchanged.
    import torch

    from task3_autonomy.curobo_stance import CuroboStanceSearch

    search = object.__new__(CuroboStanceSearch)
    search._torch = torch
    search._root_pos_w = torch.tensor([0.0, 0.0, 0.0])
    search._yaw_quat_correction = torch.tensor([1.0, 0.0, 0.0, 0.0])
    search._offset_pos_local = {
        "right": torch.tensor([0.2, -0.3, 0.9]),
        "left": torch.tensor([0.2, 0.3, 0.9]),
    }
    search._offset_quat_local = {
        "right": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "left": torch.tensor([1.0, 0.0, 0.0, 0.0]),
    }

    xc = torch.tensor([0.0])
    yc = torch.tensor([0.0])
    yawc = torch.tensor([0.0])

    alias_pos, alias_quat = search._right_base_pose_for(xc, yc, yawc)
    right_pos, right_quat = search._arm_base_pose_for(
        xc, yc, yawc, side="right"
    )
    assert alias_pos.tolist() == right_pos.tolist()
    assert alias_quat.tolist() == right_quat.tolist()
