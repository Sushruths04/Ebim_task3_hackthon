# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""CPU-only tests for task3_pipeline/verification.py (REV19 P1.1).

Run: python -m pytest task3_pipeline/tests/test_verification.py -q
  or: python -B task3_pipeline/tests/test_verification.py
"""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from task3_pipeline import verification as v  # noqa: E402


def test_object_follows_ee_delta_true_when_object_tracks_motion():
    ee_start = (0.0, 0.0, 1.0)
    ee_end = (0.0, 0.0, 1.10)  # EE rose 10cm
    obj_start = (0.0, 0.0, 0.9)
    obj_end = (0.0, 0.0, 1.0)  # object rose 10cm too
    assert v.object_follows_ee_delta(ee_start, ee_end, obj_start, obj_end)


def test_object_follows_ee_delta_false_when_object_stationary():
    """The exact failure mode this predicate exists to catch: object sits
    still near the gripper while the EE moves away with nothing in it."""
    ee_start = (0.0, 0.0, 1.0)
    ee_end = (0.0, 0.0, 1.10)
    obj_start = (0.0, 0.0, 0.75)
    obj_end = (0.0, 0.0, 0.75)  # object never moved -- not held
    assert not v.object_follows_ee_delta(ee_start, ee_end, obj_start, obj_end)


def test_object_follows_ee_delta_requires_real_ee_motion():
    """A stationary EE can't be used to judge 'follows' at all -- a static
    object next to a static gripper must not count as held."""
    ee_start = (0.0, 0.0, 1.0)
    ee_end = (0.0, 0.0, 1.001)  # EE barely moved
    obj_start = (0.0, 0.0, 0.75)
    obj_end = (0.0, 0.0, 0.751)
    assert not v.object_follows_ee_delta(ee_start, ee_end, obj_start, obj_end)


def test_object_follows_ee_delta_tolerates_partial_drift():
    ee_start = (0.0, 0.0, 1.0)
    ee_end = (0.0, 0.0, 1.10)
    obj_start = (0.0, 0.0, 0.9)
    obj_end = (0.0, 0.0, 0.995)  # object rose 9.5cm vs EE's 10cm -- close
    assert v.object_follows_ee_delta(ee_start, ee_end, obj_start, obj_end)


def test_three_predicate_hold_requires_all_three():
    base_kwargs = dict(
        ee_pos_start=(0.0, 0.0, 1.0),
        ee_pos_end=(0.0, 0.0, 1.10),
        object_pos_start=(0.0, 0.0, 0.9),
        object_pos_end=(0.0, 0.0, 1.0),
        object_rise_m=0.10,
        min_lift_m=0.05,
    )

    # All three pass.
    good = v.three_predicate_hold(gripper_position_rad=0.075, **base_kwargs)
    assert good.held
    assert (
        good.gripper_in_cage_band
        and good.object_follows_ee
        and good.object_lifted
    )

    # Gripper wide open (the O5 bug this replaces) -- must fail despite the
    # object genuinely moving with the EE.
    open_gripper = v.three_predicate_hold(
        gripper_position_rad=0.9, **base_kwargs
    )
    assert not open_gripper.held
    assert not open_gripper.gripper_in_cage_band

    # Gripper caged but object never actually moved (empty-air close).
    empty_kwargs = dict(base_kwargs)
    empty_kwargs["object_pos_end"] = base_kwargs["object_pos_start"]
    empty = v.three_predicate_hold(gripper_position_rad=0.075, **empty_kwargs)
    assert not empty.held
    assert not empty.object_follows_ee

    # Caged and following, but never actually lifted above min_lift_m.
    no_lift_kwargs = dict(base_kwargs)
    no_lift_kwargs["object_rise_m"] = 0.01
    no_lift = v.three_predicate_hold(
        gripper_position_rad=0.075, **no_lift_kwargs
    )
    assert not no_lift.held
    assert not no_lift.object_lifted


def test_three_predicate_hold_uses_grasp_transport_cage_band_by_default():
    from task3_autonomy.grasp_transport import (
        GRIP_QUALITY_CAGED_MAX_RAD,
        GRIP_QUALITY_CAGED_MIN_RAD,
    )

    assert v.GRIP_QUALITY_CAGED_MIN_RAD == GRIP_QUALITY_CAGED_MIN_RAD
    assert v.GRIP_QUALITY_CAGED_MAX_RAD == GRIP_QUALITY_CAGED_MAX_RAD


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
        "all verification tests passed."
        if not failures
        else f"{failures} FAILED"
    )
    print(f"\n{summary}")
    sys.exit(1 if failures else 0)
