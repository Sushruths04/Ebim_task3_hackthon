# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Adapter: our world state -> the vendored organizers' `grading.py`.

`grading.py` is a **development facilitator, not the official scorer** (its
own docstring says so), but it is the only executable definition of the
scoring rules that exists, and it has drifted from `task3_pipeline/config.py`
on real points (5 Stage-1/4 objects including `simple_tray`, not 4; no
partial credit below an 0.8 bean-recovery ratio). Per REV19 P0: **the grader
wins.** This module is the ONLY file allowed to import the vendored copy at
`third_party/ebim_grading/` (see PROVENANCE.md there for the pinned SHA) --
everything else calls through here so there is exactly one seam to keep in
sync if the upstream spec changes.

Pure Python, CPU-only, no Isaac import. Callers are responsible for reading
whatever live pose/bbox data they have (privileged in DEBUG_ORACLE_MODE,
perception-derived in AUTONOMOUS_MODE) and passing it in as plain tuples.
"""

from __future__ import annotations

import importlib.util
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_GRADING_PATH = (
    _REPO_ROOT
    / "third_party"
    / "ebim_grading"
    / "scripts"
    / "evaluation"
    / "task3"
    / "grading.py"
)


def _load_grading():
    """Import the vendored grading.py by path (it is not a package).

    Must register the module in ``sys.modules`` BEFORE `exec_module`:
    grading.py uses ``from __future__ import annotations``, so its
    dataclasses resolve their deferred type annotations by looking the
    module up by name at first use.
    """
    spec = importlib.util.spec_from_file_location(
        "_ebim_official_grading", _GRADING_PATH
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ebim_official_grading"] = module
    spec.loader.exec_module(module)
    return module


grading = _load_grading()

Point3D = grading.Point3D
Bounds2D = grading.Bounds2D
StageScore = grading.StageScore
FeedHoldState = grading.FeedHoldState

DEFAULT_STAGE1_OBJECTS: tuple[str, ...] = grading.DEFAULT_STAGE1_OBJECTS
DEFAULT_UTENSIL_OBJECTS: tuple[str, ...] = grading.DEFAULT_UTENSIL_OBJECTS
TASK3_SINK_REGION = grading.TASK3_SINK_REGION
TASK3_BEAN_RECOVERY_REGION = grading.TASK3_BEAN_RECOVERY_REGION
TASK3_BEAN_SPAWN_POSITION = grading.TASK3_BEAN_SPAWN_POSITION


def to_point3d(pos: Sequence[float]) -> Point3D:
    x, y, z = pos
    return Point3D(float(x), float(y), float(z))


def to_bounds2d(
    min_xy: tuple[float, float], max_xy: tuple[float, float]
) -> Bounds2D:
    x_min, y_min = min_xy
    x_max, y_max = max_xy
    return Bounds2D(
        x_min=float(x_min),
        y_min=float(y_min),
        x_max=float(x_max),
        y_max=float(y_max),
    )


def classify_table_area(xy: tuple[float, float]) -> str:
    return grading.classify_table_area(tuple(xy))


def score_stage1(
    object_positions: Mapping[str, Sequence[float]],
    object_names: Sequence[str] | None = None,
) -> StageScore:
    """object_positions: {name: (x, y, z)}. Missing names simply fail --
    matches the vendored function's own ``in object_positions`` guard."""
    points = {name: to_point3d(pos) for name, pos in object_positions.items()}
    kwargs = (
        {} if object_names is None else {"object_names": tuple(object_names)}
    )
    return grading.score_stage1_table_setup(points, **kwargs)


def score_stage4(
    object_bounds_min_max: Mapping[
        str, tuple[tuple[float, float], tuple[float, float]]
    ],
    object_z_values: Mapping[str, float],
    object_names: Sequence[str] | None = None,
) -> StageScore:
    """object_bounds_min_max: {name: ((x_min, y_min), (x_max, y_max))} --
    the object's 2-D world-aligned AABB, NOT a single point. Use
    ``Bounds2D.from_point`` semantics only if you truly have no footprint."""
    bounds = {
        name: to_bounds2d(min_xy, max_xy)
        for name, (min_xy, max_xy) in object_bounds_min_max.items()
    }
    kwargs = (
        {} if object_names is None else {"object_names": tuple(object_names)}
    )
    return grading.score_stage4_cleanup(bounds, object_z_values, **kwargs)


def score_stage2(*, beans_left: int, hold_seconds: float, smooth: bool) -> int:
    return grading.feed_score(
        beans_left=beans_left, hold_seconds=hold_seconds, smooth=smooth
    )


def score_stage3(
    bean_positions: Sequence[Sequence[float]], total_beans: int
) -> int:
    points = [to_point3d(p) for p in bean_positions]
    inside = grading.count_points_in_sphere(points)
    return grading.bean_recovery_score(inside, total_beans)


def movement_is_smooth(
    positions: Sequence[Sequence[float]], *, max_step: float
) -> bool:
    return grading.movement_is_smooth(
        [to_point3d(p) for p in positions], max_step=max_step
    )


def bean_on_spoon(
    spoon_pos: Sequence[float], bean_pos: Sequence[float]
) -> bool:
    """Mirrors the organizers' `integration_test.py` bean-on-spoon predicate:
    radial <= 0.060 m in xy, -0.020 <= dz <= 0.120."""
    dx = bean_pos[0] - spoon_pos[0]
    dy = bean_pos[1] - spoon_pos[1]
    dz = bean_pos[2] - spoon_pos[2]
    return dx * dx + dy * dy <= 0.060 * 0.060 and -0.020 <= dz <= 0.120


def feed_pose(head_pos: Sequence[float]) -> Point3D:
    """head prim world position + 0.17 m in z, per integration_test.py."""
    x, y, z = head_pos
    return Point3D(float(x), float(y), float(z) + 0.17)
