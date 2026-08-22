# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""The world boundary.

``WorldAdapter`` is the ONLY surface the pipeline touches. It is a Protocol:
each method performs one physical action (or read) and returns a metrics dict
for the auto-verifier. There are two implementations:

* ``MockWorld`` (this file) -- pure Python, parameter-sensitive, deterministic
  per seed. Lets the entire self-correction loop run and be unit-tested on CPU
  with no Isaac. It encodes the *known truths* from Task 3's real runs (e.g.
  the east stance is unreachable for the right-arm cup grasp; a base-carry
  satisfies the Stage-4 scorer without a cage) so a mock episode reproduces the
  real failure/success structure.
* ``IsaacWorld`` (see world_isaac.py stub) -- wires each method to the existing
  ``task3_autonomy`` primitives (DualArmController, TmrBaseAdapter, NavigateTo)
  and reads poses from PhysX. That is the only file that imports Isaac.

Keeping this boundary is what lets you develop and test the whole brain of the
system on a laptop and spend GPU time only on execution.
"""

from __future__ import annotations

import math
import random
from typing import Protocol, runtime_checkable

from task3_autonomy.navigation import stance_for
from task3_pipeline import config


@runtime_checkable
class WorldAdapter(Protocol):
    head_placement: str

    def reset(self, *, seed: int, head_placement: str) -> None: ...

    # actions -> metrics dict for the verifier
    def navigate_to(
        self, x: float, y: float, yaw: float | None = None, **p
    ) -> dict: ...
    def reach(self, side: str, object_name: str, **p) -> dict: ...
    def grasp(self, side: str, object_name: str, **p) -> dict: ...
    def lift(self, side: str, dz: float, **p) -> dict: ...
    def hold(self, seconds: float, **p) -> dict: ...
    def place(self, side: str, world_pose, **p) -> dict: ...
    def carry_object_to(
        self, object_name: str, x: float, y: float, **p
    ) -> dict: ...
    def _stance_for(
        self, object_xy: tuple[float, float], approach: str
    ) -> tuple[tuple[float, float], float]: ...
    def scoop(self, side: str, **p) -> dict: ...
    def pour(self, side: str, x: float, y: float, **p) -> dict: ...

    # reads
    def object_xy(self, name: str) -> tuple[float, float]: ...
    def object_z(self, name: str) -> float: ...
    def score_stage(self, stage: int) -> tuple[int, int, dict]: ...


# --------------------------------------------------------------------------- #
# Mock world -- reproduces Task 3's real success/failure structure so the loop
# can be exercised end-to-end on CPU.
# --------------------------------------------------------------------------- #


def _in_area(x: float, y: float, area: dict) -> bool:
    return (
        abs(x - area["center_x"]) <= area["scale_x"] / 2.0
        and abs(y - area["center_y"]) <= area["scale_y"] / 2.0
    )


class MockWorld:
    """Parameter-sensitive stand-in for Isaac. Not a physics engine -- a
    faithful model of *which parameter choices work*, learned from the repo's
    real runs, so the self-correction loop has something real to discover."""

    def __init__(self, seed: int = 0, head_placement: str = "a") -> None:
        self.reset(seed=seed, head_placement=head_placement)

    def reset(self, *, seed: int, head_placement: str) -> None:
        self._rng = random.Random(seed)
        self.seed = seed
        self.head_placement = head_placement
        # Object positions (x, y, z). Start poses from the repo. All 4 real
        # Stage-1/Stage-4 objects (plate2, cup, bowl2, spoon2) are stacked on
        # a plate in the kitchen per the organizer rules; "simple_tray" is
        # kept here only as inert legacy data (no plan references it -- see
        # config.STAGE1_OBJECTS).
        self.objects: dict[str, list[float]] = {
            "simple_tray": [-4.28, -1.62, 0.75],
            "bowl2": [-3.94, -1.92, 0.80],
            "spoon2": [-4.10, -1.70, 0.76],
            "plate2": [-4.20, -1.55, 0.75],
            "cup": [-4.185, -1.753, 0.747],
        }
        self.beans_total = 300
        self.beans_on_spoon = 0
        self.beans_in_bowl = 300
        self._held: str | None = None
        self._feed_hold_s = 0.0

    # -- actions ---------------------------------------------------------- #
    def navigate_to(self, x, y, yaw=None, **p) -> dict:
        # Navigation is proven-solid (2.9 cm). Model a small terminal error.
        err = abs(self._rng.gauss(0.028, 0.008))
        result = {"terminal_error_m": round(err, 4)}
        reach_object = p.get("reach_check_object")
        if reach_object is not None and reach_object in self.objects:
            ox, oy, _ = self.objects[reach_object]
            result["object_dist_m"] = round(math.hypot(x - ox, y - oy), 4)
        return result

    def navigate_to_avoiding_island(self, x, y, yaw=None, **p) -> dict:
        """Mirror of `IsaacWorld.navigate_to_avoiding_island` (added
        2026-08-09) so `MockWorld` still satisfies the interface
        `plan_stage1`/`plan_stage4` actually call.

        Without it, every MockWorld-backed stage test raised
        `AttributeError` at the navigate step -- including
        `test_plan_stage{1,4}_navigate_target_never_inside_island_
        footprint`, the two guards whose whole job is to catch a navigate
        target sitting inside the kitchen island. They had been failing
        for that reason since 2026-08-09, which is exactly the window in
        which `stance_for` started silently returning stances inside the
        island (see `navigation.WEST_WALL_BBOX`). A broken test double
        hid the regression its own assertion was written to catch.

        Delegates straight to `navigate_to`: MockWorld carries no pose, so
        it cannot compute a real detour, and inventing one would feed the
        stage tests waypoints no real run would ever produce.
        `route_avoiding_island` has its own unit tests in
        `task3_autonomy/tests/test_navigation.py`.
        """
        return self.navigate_to(x, y, yaw, **p)

    def _stance_for(self, object_xy, approach: str):
        return stance_for(object_xy, approach)

    def reach(self, side, object_name, **p) -> dict:
        stance = p.get("approach_stance", "east")
        # KNOWN TRUTH: from the east stance the right arm cannot reach the cup
        # grasp Y-offset (ends ~6 cm south / 5 cm high). North stance reaches.
        if side == "right" and stance == "east":
            return {
                "position_error_m": 0.079,
                "strict_reach": False,
                "ee_dy_m": 0.061,
            }
        err = abs(self._rng.gauss(0.02, 0.01))
        return {
            "position_error_m": round(err, 4),
            "strict_reach": err <= 0.05,
            "ee_dy_m": round(self._rng.gauss(0, 0.01), 4),
        }

    def grasp(self, side, object_name, **p) -> dict:
        stance = p.get("approach_stance", "east")
        height = p.get("grasp_height_above_origin_m", 0.068)
        # Firm cage only when reachable (north) AND correct rim height.
        good = stance == "north" and abs(height - 0.068) < 1e-6
        gripper = 0.076 if good else self._rng.uniform(0.58, 0.78)
        if good:
            self._held = object_name
        # Honest hold evidence: only a truly good grasp reports the object
        # following the end-effector at close range. A grasp that merely
        # LOOKS closed (wrong stance/height, caught body not rim) must NOT
        # claim a hold -- this is what exercises outcomes.classify_grasp's
        # "closed on empty air" check.
        follows_ee = bool(good)
        dist = 0.01 if good else round(self._rng.uniform(0.15, 0.35), 3)
        return {
            "gripper_rad": round(gripper, 3),
            # REV12 follow-up: mirror world_isaac.py's real grasp() --
            # `contact` reflects whether the grasp actually succeeded
            # (`good`), not a hardcoded True, so classify_grasp's MISS
            # path is reachable through the mock too.
            "contact": bool(good),
            "object_follows_ee": follows_ee,
            "object_ee_dist_m": dist,
        }

    def lift(self, side, dz, **p) -> dict:
        # An unreached/weak grasp flings on lift (the real Stage-4 bug).
        if self._held is None:
            return {"object_rise_m": 0.16, "ik_ok": False}  # fling artifact
        if self._held:
            self.objects[self._held][2] += dz
        return {"object_rise_m": round(dz, 3), "ik_ok": True}

    def hold(self, seconds, **p) -> dict:
        held = self._held is not None
        return {
            "z_drop_m": 0.0 if held else 0.2,
            "held_seconds": seconds if held else 0.0,
            "required_seconds": seconds,
        }

    def place(self, side, world_pose, **p) -> dict:
        if self._held:
            self.objects[self._held][0] = world_pose[0]
            self.objects[self._held][1] = world_pose[1]
            self._held = None
        return {"scored": True}

    def carry_object_to(self, object_name, x, y, z=None, **p) -> dict:
        # Base-carry / controlled-slide relocation: no rim cage required by
        # this primitive itself (the calling stage plan is responsible for
        # having done an honest grasp first, per outcomes.classify_grasp).
        method = p.get("method", "base_carry")
        if method == "grasp_place" and p.get("approach_stance") == "east":
            return {
                "scored": False,
                "reason": "grasp_place from east flings object",
            }
        self.objects[object_name][0] = x
        self.objects[object_name][1] = y
        target_z = z if z is not None else config.SINK_TABLETOP_Z
        self.objects[object_name][2] = max(
            self.objects[object_name][2], target_z
        )
        if self._held == object_name:
            self._held = None
        return {"scored": True}

    def scoop(self, side, **p) -> dict:
        pitch = p.get("entry_pitch_deg", 30.0)
        depth = p.get("drag_depth_m", 0.03)
        good = 30.0 <= pitch <= 45.0 and depth >= 0.05
        self.beans_on_spoon = (
            self._rng.randint(6, 12) if good else self._rng.randint(0, 2)
        )
        return {
            "beans_on_spoon": self.beans_on_spoon,
            "scored": self.beans_on_spoon >= 4,
        }

    def feed_hold(self, seconds, **p) -> dict:
        # Stage 2 hold: smooth + >=3 s with beans in the feed zone.
        smooth = p.get("scoop_speed", "medium") != "fast"
        if self.beans_on_spoon > 0 and smooth:
            self._feed_hold_s = seconds
        return {
            "held_seconds": self._feed_hold_s,
            "required_seconds": config.FEED_HOLD_SECONDS,
            "z_drop_m": 0.0,
            "beans_left": self.beans_on_spoon,
            "smooth": smooth,
        }

    def pour(self, side, x, y, **p) -> dict:
        # Return beans to the recovery region (== bowl).
        # Low, slow -> high ratio.
        height = p.get("pour_height_m", 0.08)
        rate = p.get("tilt_rate", "medium")
        base = 0.95 if (height <= 0.05 and rate == "slow") else 0.7
        ratio = min(1.0, base + self._rng.uniform(-0.03, 0.03))
        self.beans_in_bowl = int(self.beans_total * ratio)
        return {
            "beans_delivered": self.beans_in_bowl,
            "ratio": round(ratio, 3),
            "scored": ratio >= config.STAGE3_RATIO_FOR_2PTS,
        }

    # -- reads ------------------------------------------------------------ #
    def object_xy(self, name):
        return (self.objects[name][0], self.objects[name][1])

    def object_z(self, name):
        return self.objects[name][2]

    def score_stage(self, stage: int) -> tuple[int, int, dict]:
        if stage == 1:
            # DEV SMOKE-TEST ONLY (see config.py docstring): coarse "roughly
            # in the dining room" check. The real Stage-1 scorer (a LATER
            # task) will check each object against its actual assigned seat
            # from seats.py instead of this rectangle.
            passed = [
                n
                for n in config.STAGE1_OBJECTS
                if _in_area(*self.object_xy(n), config.DINING_AREA)
            ]
            return min(4, len(passed)), 4, {"passed": passed}
        if stage == 2:
            # smooth + >=3 s hold -> points = beans_left (cap 4)
            s = (
                min(4, self.beans_on_spoon)
                if self._feed_hold_s >= config.FEED_HOLD_SECONDS
                else 0
            )
            return (
                s,
                4,
                {"hold_s": self._feed_hold_s, "beans": self.beans_on_spoon},
            )
        if stage == 3:
            ratio = self.beans_in_bowl / self.beans_total
            score = (
                4
                if ratio >= 1.0
                else 3
                if ratio >= 0.9
                else 2
                if ratio >= 0.8
                else 0
            )
            return score, 4, {"ratio": round(ratio, 3)}
        if stage == 4:
            # All 4 real utensils (config.STAGE1_OBJECTS -- no tray) must be
            # returned to the marked sink region.
            b = config.SINK_BOUNDS
            passed = []
            for name in config.STAGE1_OBJECTS:
                x, y = self.object_xy(name)
                if (
                    b["x_min"] <= x <= b["x_max"]
                    and b["y_min"] <= y <= b["y_max"]
                    and self.object_z(name) >= config.SINK_TABLETOP_Z
                ):
                    passed.append(name)
            return min(4, len(passed)), 4, {"passed": passed}
        return 0, 4, {}
