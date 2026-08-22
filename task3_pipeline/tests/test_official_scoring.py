# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for `task3_pipeline/official_scoring.py`, the thin adapter
onto the vendored `third_party/ebim_grading/` grader (REV19 P0.3).

Run: python -m pytest task3_pipeline/tests/test_official_scoring.py -q
  or: python -B task3_pipeline/tests/test_official_scoring.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from task3_pipeline import official_scoring as scoring  # noqa: E402


def test_default_objects_are_five_including_tray():
    assert scoring.DEFAULT_STAGE1_OBJECTS == (
        "simple_tray",
        "bowl2",
        "spoon2",
        "plate2",
        "cup",
    )
    assert scoring.DEFAULT_UTENSIL_OBJECTS == scoring.DEFAULT_STAGE1_OBJECTS


def test_score_stage1_uses_dining_rectangle_not_seat_exact():
    dining_xy = (-2.85, 1.9)  # TASK3_DINING_AREA centre
    positions = {
        name: (*dining_xy, 0.8) for name in scoring.DEFAULT_STAGE1_OBJECTS
    }
    result = scoring.score_stage1(positions)
    assert result.score == 5
    assert result.max_score == 5


def test_score_stage1_missing_objects_fail_not_crash():
    result = scoring.score_stage1({"cup": (-2.85, 1.9, 0.8)})
    assert result.score == 1
    assert result.max_score == 5
    assert set(result.failed) == {"simple_tray", "bowl2", "spoon2", "plate2"}


def test_score_stage4_aabb_overlap_not_point_containment():
    sink = scoring.TASK3_SINK_REGION
    tabletop_z = sink.tabletop_z
    # Object's AABB straddles the sink boundary (its own point is outside,
    # but the box overlaps) -- must still pass, because the real scorer is
    # an AABB overlap, not point-in-rect.
    outside_point_but_overlapping = {
        "cup": (
            (sink.bounds.x_max - 0.01, sink.bounds.y_max - 0.01),
            (sink.bounds.x_max + 0.20, sink.bounds.y_max + 0.20),
        ),
    }
    result = scoring.score_stage4(
        outside_point_but_overlapping, {"cup": tabletop_z}
    )
    assert result.score == 1, (
        "AABB overlap must count even if centroid is outside"
    )


def test_score_stage4_z_below_tabletop_fails():
    sink = scoring.TASK3_SINK_REGION
    box = {
        "cup": (
            (sink.bounds.x_min, sink.bounds.y_min),
            (sink.bounds.x_max, sink.bounds.y_max),
        )
    }
    result = scoring.score_stage4(box, {"cup": sink.tabletop_z - 0.01})
    assert result.score == 0


def test_bean_recovery_no_partial_credit_below_0_8():
    assert scoring.grading.bean_recovery_score(7, 10) == 0  # ratio 0.7
    assert scoring.grading.bean_recovery_score(8, 10) == 2  # ratio 0.8
    assert scoring.grading.bean_recovery_score(9, 10) == 3  # ratio 0.9
    assert scoring.grading.bean_recovery_score(10, 10) == 4  # ratio 1.0


def test_score_stage2_requires_smooth_and_hold():
    assert (
        scoring.score_stage2(beans_left=4, hold_seconds=3.5, smooth=True) == 4
    )
    assert (
        scoring.score_stage2(beans_left=4, hold_seconds=2.9, smooth=True) == 0
    )
    assert (
        scoring.score_stage2(beans_left=4, hold_seconds=3.5, smooth=False) == 0
    )
    assert (
        scoring.score_stage2(beans_left=6, hold_seconds=3.5, smooth=True) == 4
    )  # capped


def test_bean_on_spoon_radial_and_dz_bounds():
    spoon = (0.0, 0.0, 1.0)
    assert scoring.bean_on_spoon(spoon, (0.05, 0.0, 1.05))  # inside
    assert not scoring.bean_on_spoon(
        spoon, (0.07, 0.0, 1.05)
    )  # outside radius
    assert not scoring.bean_on_spoon(
        spoon, (0.0, 0.0, 0.97)
    )  # dz too negative
    assert scoring.bean_on_spoon(
        spoon, (0.0, 0.0, 1.119)
    )  # just inside +0.12 edge


def test_feed_pose_is_head_plus_017_z():
    p = scoring.feed_pose((1.0, 2.0, 3.0))
    assert (p.x, p.y, p.z) == (1.0, 2.0, 3.17)


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as exc:  # noqa: BLE001
                failures += 1
                print(f"[FAIL] {name}: {exc}")
    summary = (
        "all official_scoring tests passed."
        if not failures
        else f"{failures} FAILED"
    )
    print(f"\n{summary}")
    sys.exit(1 if failures else 0)
