# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""IsaacWorld -- the real WorldAdapter.

This is the ONE file that imports Isaac and the existing task3_autonomy
primitives. Every method has the same signature as MockWorld and must return
the same metrics keys, so the orchestrator/verifier/memory/policy code above
is reused unchanged.

Wiring: this file reuses the PROVEN grasp geometry and hold-verification
primitive from ``scripts/task3/verify_grasp_lift.py`` (10/10 cup grasp) --
the pure functions/constants there (``cup_grasp_target``,
``object_grasp_target``, ``object_follows_end_effector``, ``STANCE``,
``PREGRASP_Z``, ``GRASP_Z``, ``TRAVEL_SPINE_M``, ...) are imported directly,
not reimplemented. The Stage-1/4 reach-wall fix (docs/
TASK3_MASTER_EXECUTION_PLAN_2026-07-24.md section 4) lives in ``reach()``:
every manipulation drives the base to a stance computed from the object's
LIVE PhysX pose each call (never a hardcoded per-episode world coordinate),
using the same base-relative offset that made the proven cup grasp work.

Construct this AFTER ``isaaclab.app.AppLauncher`` has created the Isaac Sim
app (mirrors ``scripts/task3/verify_grasp_lift.py`` / ``run_episode.py``).
``reset()`` does the actual Isaac scene composition (room + robot + physics
reset); ``__init__`` only stores configuration so the module stays
CPU-importable (no Isaac import at module scope).
"""

from __future__ import annotations

import contextlib
import math
import queue
import sys
import threading
import time
from pathlib import Path
from typing import Any

from task3_autonomy.arms import DEFAULT_CONTACT_FREEZE_MAX_TARGET_RAD
from task3_autonomy.arms import GRIPPER_OPEN_RAD
from task3_autonomy.grasp_contract import (
    GraspContractError,
    GraspMemoryEntry,
    append_grasp_memory,
    load_candidates,
    load_ranked,
)
from task3_autonomy.er_grasp_orientation import (
    offset_along_approach as _offset_along_approach,
)
from task3_autonomy.grasp_reanchor import ReanchorAction, reanchor_candidate
import inspect as _inspect

from task3_autonomy.navigation import pose_reached as _nav_pose_reached
from task3_autonomy.navigation import (
    STANCE_REACH_RADIUS_M,
    ProgressWatchdog,
    point_clears_island,
)
from task3_autonomy.navigation import (
    MEASURED_REACH_LIMIT_M as _MEASURED_REACH_LIMIT_M,
)
from task3_autonomy.navigation import stance_for as _stance_for_impl
from task3_pipeline import config

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENES_DIR = REPO_ROOT / "scripts" / "scenes"
COMMON_DIR = REPO_ROOT / "scripts" / "common"
EVALUATION_DIR = REPO_ROOT / "scripts" / "evaluation" / "task3"
TASK3_SCRIPTS_DIR = REPO_ROOT / "scripts" / "task3"


def _load_gripper_profiles():
    """Import the vendored, import-safe gripper_profiles.py by path (REV20
    §3, SONNET_REV20_UNBLOCK_AND_CONTINUE.md). Same pattern as
    official_scoring.py's grading.py loader: register in sys.modules before
    exec since the module uses ``from __future__ import annotations``.

    The vendored file computes its own ``REPO_ROOT`` as
    ``Path(__file__).resolve().parents[2]`` -- correct at its ORIGINAL
    location (repo_root/task3_isaacsim/scripts/gripper_profiles.py) but
    wrong once vendored one level deeper at
    third_party/ebim_grading/task3_isaacsim/scripts/gripper_profiles.py
    (resolves to third_party/ebim_grading itself, not the real repo root --
    confirmed by a direct import check, not assumed). Rebuild each
    profile's ``robot_usd`` against the correct root rather than edit the
    vendored file (PROVENANCE.md: unmodified).
    """
    import importlib.util
    import sys as _sys

    path = (
        REPO_ROOT
        / "third_party"
        / "ebim_grading"
        / "task3_isaacsim"
        / "scripts"
        / "gripper_profiles.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_ebim_gripper_profiles", path
    )
    module = importlib.util.module_from_spec(spec)
    _sys.modules["_ebim_gripper_profiles"] = module
    spec.loader.exec_module(module)

    wrong_root = module.REPO_ROOT
    module.GRIPPER_PROFILES = {
        key: profile._replace(
            robot_usd=REPO_ROOT / profile.robot_usd.relative_to(wrong_root)
        )
        for key, profile in module.GRIPPER_PROFILES.items()
    }
    return module


gripper_profiles = _load_gripper_profiles()

# Camera used only when record_video=True (same framing as verify_grasp_lift).
CAMERA_POSITION = (-1.6, -3.4, 2.2)
CAMERA_LOOK_AT = (-4.1, -1.7, 0.8)
VIDEO_FPS = 2


def render_tick_stride(
    dt: float, video_fps: float, record_video: bool, *, idle_hz: float = 10.0
) -> int:
    """How many physics ticks may pass between Kit render passes.

    `_tick()` used to call `sim.step(render=True)` on EVERY main-thread
    tick. A Kit render pass is the single most expensive thing in the tick
    and almost none of them are read:

    - `VIDEO_FPS` is 2 and `dt` is 0.005, so the video path saves one frame
      per 100 ticks. **99 of every 100 renders were discarded even while
      recording video.**
    - Every camera/annotator read in this codebase pumps `app.update()` a
      few times itself immediately before reading (see
      `_capture_live_er_frame`, `_capture_body_camera_for_graspgenx`), so
      none of them depend on the per-tick render either.

    Measured cost of the un-decimated loop: 86-107 ms of WALL time per tick
    (`WORLD_ISAAC_TICK`'s own `s_per_tick`, runs 1 and 3, 2026-08-21) --
    9-12 ticks/s, i.e. the sim advancing at ~0.05x real time. One
    `reach()` in the cup path budgets 24.5 simulated seconds = 4,900 ticks,
    so it costs 7-9 real minutes; the cup ladder's 9 reach() calls cost
    63-79 real minutes per candidate, which is exactly the 80-100 min/run
    the owner reports.

    Returns the stride so the render cadence is DERIVED from what actually
    consumes frames, never guessed. While recording, it is exactly the
    video capture cadence, so every render produced is a frame saved and
    the video is unchanged. While not recording, nothing consumes frames
    per-tick at all, so this falls back to a slow `idle_hz` keep-alive
    rather than 0 -- Kit subsystems expect to be pumped periodically, and a
    modest floor costs almost nothing while removing ~95% of the passes.
    """
    if not (dt > 0.0 and video_fps > 0.0 and idle_hz > 0.0):
        raise ValueError("dt, video_fps and idle_hz must be positive")
    hz = video_fps if record_video else idle_hz
    return max(1, round(1.0 / (dt * hz)))

# Robot spawn when NOT skipping navigation (matches run_episode.py).
FULL_ROBOT_SPAWN_POSITION = (-4.6, 2.7, 0.0)
FULL_ROBOT_SPAWN_YAW = -90.0

# Reach-fix (plan section 4): the proven cup grasp used STANCE=(-3.32,-1.72)
# against a cup at CUP_GRASP_XY=(-4.145,-1.75) approx (-4.185,-1.753) spawn --
# a base-relative offset of dx=+0.865, dy=+0.033 from the object, facing
# west. That offset (not the absolute coordinate) is what is reused here,
# recomputed from the object's LIVE pose every call. The offset/radius
# constants and the island-clearance clamp live in
# task3_autonomy.navigation (CPU-testable, shared with MockWorld) --
# see navigation.stance_for().

# Step 3 (plans/handoff.md sec 5): the final approach standoff/duration for
# the rate-limited gentle descend -- servoing the full descend at max rate
# let the fingers shove the object before ever settling near it.
# Derived from `navigation.pose_reached`'s own arrival tolerance rather
# than restated, so "already parked" can never disagree with what
# navigate_to itself would have accepted as arrived.
NAVIGATE_ARRIVAL_TOLERANCE_M = float(
    _inspect.signature(_nav_pose_reached)
    .parameters["position_tolerance_m"]
    .default
)

GENTLE_DESCEND_M = 0.08
GENTLE_DESCEND_SECONDS = 3.0

# How far a live ER-2 grasp point may sit from the object it claims to be on
# before the answer is refused (`_live_er_grasp_pose`). DERIVED, not chosen:
# GOTCHAS.md's ER-2 section measures wide-shot pointing at ~5.7 cm of error
# and the perception camera here IS a wide shot, so anything inside that band
# is the model working as documented. Doubling it leaves room for the object
# being off-centre in its own bbox while still rejecting the failure that
# actually costs episodes -- a point on the wrong object, or on the counter
# beside it, which drives the jaws into the scene and flings things across
# the room. It is a VETO threshold, never a source of grasp geometry.
LIVE_ER_GRASP_MAX_MISS_M = 0.114

# How far behind the D405 mount the wrist camera sits, along its own view
# ray, so the near plane clears the gripper housing. Small because the
# camera's near plane is 0.01 m; the point is only to get outside the shell.
WRIST_CAMERA_BACKOFF_M = 0.06

# How far BEYOND the pad midpoint the wrist camera aims, along the same ray.
# Aiming at the pads points the camera at its own fingers; the object sits
# past them. 0.20 m puts the grasp region comfortably inside the frame at
# the ~0.25 m the pads sit from the mount.
WRIST_CAMERA_LOOK_BEYOND_M = 0.20

# Render grasp frames from the robot's own wrist mount instead of the fixed
# scene camera. DEFAULT OFF, on evidence, not on preference.
#
# The wrist camera is the architecturally correct choice: it moves with the
# arm, it is what the organisers record (observation.images.wrist_left), and
# GOTCHAS measures the fixed wide shot at ~5.7 cm of pointing error, which
# is baked into every grasp point this project has produced.
#
# But as built it is not yet usable. Un-calibrated, the camera sits inside
# the gripper housing and renders its shell -- ER-2 answered "grasp the
# narrow handle of the spoon from the side" from an image containing no
# spoon, caught only by the grasp-point veto at miss_m 0.2472. Calibrated
# against the pad midpoint it does clear the housing and see the counter,
# the plate and the spoon, but the gripper body still fills most of the
# frame, and the printed eye/pad positions disagree by 3.5 m in y, which
# means the mount and pad poses are being read at different times or the
# local transform is wrong.
#
# STATUS after four calibration attempts, all measured by looking at the
# rendered frame rather than at the numbers:
#
#   1. identity transform      -> renders the inside of the gripper housing
#   2. aim at pad midpoint     -> renders the gripper filling the frame,
#                                 counter and plate barely visible at the edge
#   3. aim past pads on the
#      mount->pad ray          -> aim point z=1.088, ABOVE both pads (0.991)
#                                 and eye (0.919); looks up and away
#   4. aim along tool +Z,
#      eye behind the pads     -> uniform grey; camera is inside a solid
#
# The eye/pad frame bug IS fixed (they agreed to 0.15 m in attempt 4, versus
# 3.5 m before, once the mount pose came from the articulation tensor API
# instead of the frozen XformCache). What remains is placing the optical
# frame somewhere that actually sees the grasp region, and guessing at aim
# vectors is not converging.
#
# The right way to finish it is to stop guessing: render from a batch of
# candidate poses in one probe, look at the images, and pick. That is a
# contained piece of work and it is what the next session should do.
#
# The scene camera is what produced this project's only firm hold
# (gripper_position_rad 0.4077), so the default stays there.
USE_WRIST_CAMERA = False

# Body/head camera: a top-level articulation body at z=1.29 that clears the
# grippers, unlike the two wrist mounts buried inside the housings. Pose
# picked by rendering a grid of candidates and LOOKING at them
# (scripts/task3/probe_head_camera_poses.py): of twelve, the pair that shows
# the counter, tray and cup is the mount raised 0.05 m and aimed at the
# object.
HEAD_CAMERA_BODY = "head_camera_mounting_point"
HEAD_CAMERA_RISE_M = 0.05
USE_HEAD_CAMERA = True

# Position gain applied to the driven gripper joint for the close. The asset
# authors 3.0 and nothing ever changed it, which caps closing torque at
# ~3 N*m against a 50 N*m effort limit -- see
# DualArmController.set_gripper_stiffness for the measurement. 60.0 keeps the
# closing torque an order of magnitude below that authored limit while giving
# the jaws enough authority to close against contact, so the effort limit
# stays the thing that bounds force, as designed.
# The roll that points the gripper's FINGERS DOWN at the table: math.pi.
#
# REVERTED 2026-08-21 after a wrong change. It was briefly set to 0.0 on the
# strength of one number -- `pad_midpoint_z - wrist_z`, which measures +0.021
# at roll=pi and -0.089 at roll=0 -- read as "pi puts the fingers up".
#
# That reading was WRONG, and the giveaway is in `tool_offset`'s own
# docstring: `ee_world_poses()` does not return the wrist. It returns
# `<side>_fr3v2_hand_tcp`, a FRANKA hand tcp at xyz="0 0 0.1034" -- "a frame
# no physical link tracks" on this Robotiq robot. It is a PHANTOM point
# 103.4 mm along the hand's z axis.
#
# So at roll=pi that axis points DOWN and the phantom frame is projected
# 103 mm BELOW the pads, which is why the pads measure "above" it. Measured
# directly: pad_minus_wrist_z = +0.1033 at roll=pi and -0.1033 at roll=0 --
# both exactly the 0.1034 tcp offset, sign-flipped. The number was never
# about the gripper's orientation at all; it was the tcp offset being
# rotated.
#
# Settled by photograph, not arithmetic
# (scripts/task3/probe_roll_visual_check.py, outputs/task3_roll_check/):
# at roll=pi the Robotiq fingers hang down over the table; at roll=0 the arm
# flings itself up and away and the gripper ends up high against the wall.
#
# Do not "fix" this again from a z-offset. If it is ever in doubt, run the
# visual check -- it takes one run and produces two photographs.
TOP_DOWN_ROLL_RAD = math.pi

CLOSE_GRIPPER_STIFFNESS = 60.0

# Half-width, in pixels, of the stage-two crop sent back to ER-2.
# DERIVED from the frame, not chosen: an eighth of the capture's short side
# at the 640x480 the head camera renders, which is a box comfortably larger
# than any single object on the counter (so the whole object plus context
# stays in frame) while giving the model ~5x the pixels on target that the
# wide shot did.
ER_GRASP_CROP_HALF_PX = 60

# How far the object may be displaced during the pre-close re-center before
# that re-center is abandoned. DERIVED: it is the gripper's own jaw span
# (`perception_grasp.GRIPPER_MAX_OPENING_M`, the 2F-85's 0.085 m opening).
# Once the object has moved further than the jaws can open, the grasp the
# re-center is refining cannot succeed however long it servos, so continuing
# only pushes the object further -- measured on run 5 as a 14.5 cm slide
# followed by the plate landing on the floor.
from task3_autonomy.perception_grasp import (  # noqa: E402
    GRIPPER_MAX_OPENING_M as _GRIPPER_MAX_OPENING_M,
)

RECENTER_MAX_OBJECT_PUSH_M = _GRIPPER_MAX_OPENING_M

# Collision approximation forced onto the scoring objects at scene build --
# see IsaacWorld._correct_scoring_object_collision for the measurement and
# for why the organisers permit it. `convexDecomposition` rather than `sdf`:
# both preserve the concave underside a jaw has to hook, and decomposition
# is the standard PhysX choice for graspable rigid props, while SDF
# collision carries a per-shape cost this scene does not need.
SCORING_OBJECT_COLLISION_APPROXIMATION = "convexDecomposition"

# ONLY these are replaced. `sdf` and `convexDecomposition` already preserve
# a concave shape -- `sdf` more accurately than anything else PhysX offers --
# so rewriting them would be a downgrade, not a correction. The live scene
# really does mix the two: the first run of this code reported
# `bowl2:Cylinder_003(sdf)` and `spoon2:Tea_Spoon(sdf)`, which must be left
# exactly as the organisers authored them. `None` covers a mesh with
# CollisionAPI and no approximation authored at all, which is the case this
# exists for -- PhysX then defaults to convexHull.
SCORING_OBJECT_COLLISION_REPLACEABLE = (
    None,
    "none",
    "convexHull",
    "boundingCube",
    "boundingSphere",
)


# Half-turn roll symmetry selection: OFF, on measurement.
#
# A parallel jaw is symmetric about its approach axis, so `roll` and
# `roll + 180` are the same grasp, and choosing the one whose IK solution
# puts joint 7 nearest its current angle looked like free accuracy.
#
# Run 16, the first run in which the selection actually executed, priced it:
# `GRASP_ROLL_CHOICE object='cup' side='left' requested_roll=0.0
# chosen_roll=-180.0 joint7_travel_rad=0.9551`. That 0.955 rad is the CHEAPER
# of the two options, against a joint that manages ~0.1 rad in the 4 s
# `arms.reach` allows (12 Nm / damping 500 = 0.024 rad/s). Both candidates
# are ~10x over budget, so choosing between them cannot help -- and the
# 180-degree option made it worse, turning a ramp that had been completing
# into `descend_gentle_ramp ok=False`, because the wrist now has to spin
# through half a turn during the approach itself.
#
# The real conclusion is bigger than the roll: for these grasp poses IK wants
# ~1 rad of wrist rotation, and joint 7 cannot deliver it in the time the
# reach allows. The fix is more wrist travel per second (the damping change
# in 8939f3a, at an intermediate value) or more time before the timed legs
# begin -- not a smarter choice of equivalent target.
GRASP_ROLL_SYMMETRY_ENABLED = False


class _RollSelectionDisabled(Exception):
    """Internal control flow for the gate above."""


class _ObjectPushedAway(Exception):
    """Raised inside the re-center's tick hook to unwind out of
    `arms.reach`'s servo loop. Private, and caught by the only caller."""

    def __init__(self, moved_m: float) -> None:
        super().__init__(f"object pushed {moved_m:.4f} m during re-center")
        self.moved_m = moved_m


def _resolve_camera_prim_path(camera: Any) -> str:
    """USD prim path for whatever `rep.create.camera()` handed back.

    It returns a `ReplicatorItem`, which is NOT a string and does NOT have
    `GetPath()` -- `sim_camera_perception` assumes one of those two and
    raises `AttributeError: 'ReplicatorItem' object has no attribute
    'GetPath'` on the live path (observed 2026-08-14,
    `outputs/keep_live_er_run1.log`). Replicator's own return type is not
    stable across Kit versions, so this tries the cheap accessors in order
    and only then falls back to asking the stage, rather than pinning one
    accessor that a Kit upgrade can quietly change again.
    """
    if isinstance(camera, str):
        return camera
    for accessor in ("GetPath", "get_output_prims"):
        try:
            value = getattr(camera, accessor)()
        except Exception:  # noqa: BLE001, S112
            continue
        path = getattr(value, "pathString", value)
        if isinstance(path, str) and path.startswith("/"):
            return path
    text = str(camera)
    if text.startswith("/"):
        return text
    # Last resort: the stage knows. Replicator authors its cameras under
    # /Replicator, and this method runs immediately after creating exactly
    # one, so the most recently authored camera prim is ours.
    from omni.usd import get_context
    from pxr import UsdGeom

    stage = get_context().get_stage()
    camera_paths = [
        prim.GetPath().pathString
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Camera)
    ]
    if not camera_paths:
        raise RuntimeError(
            "could not resolve the perception camera's prim path: "
            f"rep.create.camera() returned {camera!r} and the stage has no "
            "UsdGeom.Camera prims"
        )
    return camera_paths[-1]

# Q2 (2026-08-03, SYNC 22/23): real GPU evidence (task3-submission @
# 659d792, tick 67120-68320) traced cup's counter-edge knockoff to a
# push_approach target 0.037m PAST this same ~0.855m FR3 reach ceiling
# (task3_autonomy/perception_targets.py REACH_LIMIT_M) -- the arm strained
# at/beyond its kinematic limit, directly above the object, for the full
# 8s reach() budget (1600/1600 ticks "IK-solved" but never within
# position tolerance -- error oscillating 0.08-0.66m, never converging)
# before the object left the counter. VM B's earlier N1 hypothesis (the
# retract-tuck's joint-space LERP sweeping into the object) is REFUTED for
# this run: all 5 tuck_z_delta['cup'] samples across the episode are ~0.0,
# including the one immediately after the fall. Reuse the same measured
# ceiling as a pre-flight gate instead of letting push_approach commit to
# a doomed multi-second reach() next to the object.
PUSH_APPROACH_REACH_LIMIT_M = 0.855

# R7 T2 (plans/SYNC.md 2026-08-04 ~19:48 UTC): R7 T1's offline IK
# feasibility sweep found the default STANCE_REACH_RADIUS_M (~0.780m, the
# radius `reach()`'s own grasp calibration depends on and must not change)
# sits exactly in a 0%-feasible zone for push_approach's target (0/60,750
# sampled combinations at this distance or farther from the object).
# Moving the stance ~0.30m closer measured 6.8% feasible (1,698/25,650 at
# that offset and its neighbors), the largest lever found across stance,
# target-Z, orientation, and warm-start. Bounded below by
# `navigation.BASE_HALF_WIDTH` (0.40m): the sweep's most aggressive tested
# offset (~0.38m radius) would put the robot base's own footprint edge
# past the object's center -- a real collision risk, not just "tight" --
# so this stops at 0.30m off, not the sweep's full extent.
PUSH_STANCE_RADIUS_M = STANCE_REACH_RADIUS_M - 0.30

# How far the PUSH stance's radius may grow when no angle clears at
# PUSH_STANCE_RADIUS_M -- which is every kitchen object, since that whole
# annulus sits inside the counter's inflated footprint. The measured arm
# reach ceiling is the only defensible bound: past it the stance exists but
# the arm cannot get to the object from it. Imported rather than retyped so
# it cannot drift from the value perception_targets enforces.
#
# This used to be `_rotate_to_clear_island`'s silent default for EVERY
# caller. The grasp path shares that function and must not grow -- see the
# note at `_stance_for`.
PUSH_STANCE_GROWTH_CEILING_M = _MEASURED_REACH_LIMIT_M

# Q3 (2026-08-03, SYNC 22-24): VM B's N2 (task3_autonomy/perception_targets.py)
# beat the hardcoded push-contact height for exactly two objects offline --
# bowl2 and spoon2 -- and was explicitly refuted for cup (N3) and did not
# win for plate2. Scoped here to the same two objects on purpose: a
# perception-everywhere change is exactly the kind of stacked speculative
# fix that has kept this project at 0 for ten sessions.
PUSH_PERCEPTION_OBJECTS = frozenset({"bowl2", "spoon2"})


def _lazy_isaac_imports() -> dict[str, Any]:
    """Import everything Isaac-specific. Only ever called after AppLauncher."""
    for path in (
        SCENES_DIR,
        COMMON_DIR,
        EVALUATION_DIR,
        TASK3_SCRIPTS_DIR,
        REPO_ROOT,
    ):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))

    # These names are never referenced directly in this function -- they are
    # captured by `dict(locals())` below and consumed via the `m["name"]`
    # dict returned to callers. Ruff's F401 cannot see that use and its
    # --fix (enabled in .pre-commit-config.yaml) WILL silently delete this
    # whole block if the noqa markers are removed -- verified: `ruff check`
    # flags all of them on the unmodified committed file. Keep the markers.
    import grading  # noqa: F401

    # P0.8 (handoff sec 17.2 items 1-3): scoop()/feed_hold()/pour() are thin
    # adapters over this script's own Stage-2 geometry constants and
    # grading.py's real Stage-3 sphere scorer -- imported, not re-derived.
    # ``stage2`` is the whole module (not select names) because several of
    # its constants (DINING_TARGET, HEAD_Z_OFFSET_M, ...) are read directly
    # as ``stage2.NAME`` below, mirroring how ``vgl`` is used elsewhere in
    # this file.
    import run_stage2_feeding as stage2  # noqa: F401
    import verify_grasp_lift as vgl  # noqa: F401
    from integration_test import resolve_prim_path  # noqa: F401
    from run_episode import (  # noqa: F401
        _fix_single_articulation_root,
        _save_rgb_frame,
        make_headless_robot_usd,
        prepare_rigid_body_view_path,
    )
    from scene_robot_room_keyboard import (  # noqa: F401
        configure_keyboard_control_stage,
        configure_robot_room_stage,
        disable_robot_external_wrenches,
        make_control_scene_cfg,
        reset_robot_to_default_state,
        yaw_to_quat,
    )
    from teleop_targets import _quaternion_from_rpy  # noqa: F401

    from isaacsim.core.prims import RigidPrim  # noqa: F401

    import isaaclab.sim as sim_utils  # noqa: F401
    from isaaclab.scene import InteractiveScene  # noqa: F401
    from isaaclab.sim import SimulationContext  # noqa: F401

    from task3_autonomy.arms import (  # noqa: F401
        DualArmController,
        gripper_holds_object,
    )
    from task3_autonomy.navigation import (  # noqa: F401
        TASK3_DOOR_X,
        TASK3_KITCHEN_LANE_Y,
        route_avoiding_island,
        route_via_door,
    )
    from task3_autonomy.skills import (  # noqa: F401
        TRANSIT_ARM_POSE,
        NavigateTo,
        RotateTo,
        TmrBaseAdapter,
        ramp_arm_pose,
    )

    return dict(locals())


class IsaacWorld:
    """Wraps DualArmController + TmrBaseAdapter + NavigateTo + PhysX reads."""

    def __init__(
        self,
        *,
        simulation_app: Any = None,
        record_video: bool = False,
        out_dir: str = "outputs/task3_pipeline",
        object_names: tuple[str, ...] = config.STAGE1_OBJECTS,
        skip_navigation: bool = False,
        skip_grasp: bool = False,
        use_curobo_stance: bool = True,
        stage4_objects: tuple[str, ...] | None = None,
        stage1_objects: tuple[str, ...] | None = None,
        close_hold_on_contact: bool = False,
        select_nearer_arm_side: bool = False,
        push_perception_targets: bool = False,
        curobo_rate_both_arms: bool = False,
        reach_gate_enabled: bool = True,
        push_stance_navigate_budget_s: float = 25.0,
        use_ranked_grasp: bool = True,
        perception_grasp: bool = False,
        live_er_grasp: bool = False,
        er_grasp_crop_refine: bool = True,
        curobo_grasp: bool = False,
        gripper: str | None = None,
    ) -> None:
        # simulation_app is the isaaclab.app.AppLauncher().app object the
        # caller must construct BEFORE this class (mirrors
        # verify_grasp_lift.py / run_episode.py) -- scene composition calls
        # app.update() during stage setup, so this cannot be None once
        # reset() actually builds the scene.
        self.simulation_app = simulation_app
        # REV20 Sec 3 (SONNET_REV20_UNBLOCK_AND_CONTINUE.md): default is
        # `gripper=None` -> the exact file this repo has ALWAYS actually
        # loaded (assets/mobile_fr3_duo_v0_2.usd, 68 MB). This is NOT the
        # same file as gripper_profiles.py's own "panda" entry (the
        # organizers' 1.5 KB thin reference layer at
        # third_party/franka_description/urdfs/
        # mobile_fr3_duo_v0_2_franka_hand.usd)
        # -- "we have never been running either official profile" (handoff
        # doc). Routing the CURRENT default through get_gripper_profile
        # would silently swap in an untested file, which is exactly the
        # regression this migration must not cause. Every grasp constant in
        # this repo was calibrated against the current file's fingers, so
        # it stays the default until the robotiq path has measured parity.
        # `gripper="robotiq"` opts into the real competition asset;
        # `gripper="panda"` opts into the organizers' official thin-layer
        # panda profile (untested here, not the current default).
        if gripper is None:
            self.gripper_profile = None
            self._robot_usd_path = (
                REPO_ROOT / "assets" / "mobile_fr3_duo_v0_2.usd"
            )
        else:
            self.gripper_profile = gripper_profiles.get_gripper_profile(
                gripper
            )
            self._robot_usd_path = self.gripper_profile.robot_usd
        self.record_video = record_video
        # Resolved on first use -- `self.sim` does not exist yet here, and
        # the stride is derived from `sim.cfg.dt`. See render_tick_stride.
        self.__render_tick_stride: int | None = None
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.frames_dir = self.out_dir / "frames"
        self.object_names = tuple(object_names)
        self.skip_navigation = skip_navigation
        self.skip_grasp = skip_grasp
        # REV12 T6: plan_stage1 (and any future caller) checks this before
        # using reach_and_grasp_ranked -- default ON, DEFAULT ON per the
        # ladder; --no-ranked-grasp (run_task3.py) flips it for A/B (T8).
        self.use_ranked_grasp = use_ranked_grasp
        # REVIEW #9 (handoff sec 76): which objects plan_stage4 ATTEMPTS.
        # None = all of them (unchanged default). score_stage(4) always
        # scores the full set, so max_score stays 4 either way -- this only
        # bounds the work so the stage can finish inside its 3600 s ceiling.
        self.stage4_objects = tuple(stage4_objects) if stage4_objects else None
        # 2026-08-14: the stage-1 twin of the above, added for the same
        # reason -- see plan_stage1. None = all objects (unchanged default).
        self.stage1_objects = tuple(stage1_objects) if stage1_objects else None
        # 2026-08-14: default for grasp()'s `close_hold_on_contact`, which
        # REV13 T4-followup left as an explicit opt-in "pending live GPU
        # verification". Exposing it as a world-level default (still False)
        # is what makes that verification runnable from the CLI; a per-call
        # `p["close_hold_on_contact"]` still wins over this.
        self.close_hold_on_contact = bool(close_hold_on_contact)
        # GATE B0 (handoff sec 68): cuRobo batch-IK stance search, proven
        # (sec 64-67) to beat navigation.stance_for()'s fixed reach radius.
        # Built lazily in _stance_for() (needs a live robot post-reset()),
        # never rebuilt per call. stance_for() stays the fallback -- a
        # search miss must never abort a stage (see _stance_for below).
        self.use_curobo_stance = use_curobo_stance
        self._curobo_stance_search = None
        # P5 (plans/LOOP_PROMPT_VM_A.md rev 2): every reach()/push() call in
        # this codebase is hardcoded to "right" (world_isaac.py:534 says so
        # outright). Both arm bases already exist and _arm_base_relative
        # already takes a side -- this flag, default OFF (unchanged
        # behavior, trivially revertible), lets carry_object_to() pick
        # whichever side's live target_norm_from_arm_base_m is smaller
        # instead of assuming "right". VM B proved side selection can flip
        # IK success 0%->100% for a single object (SYNC 10); this
        # productionizes that into the Stage 4 push/stance path itself.
        self.select_nearer_arm_side = select_nearer_arm_side
        # Q3: default-off, scoped to PUSH_PERCEPTION_OBJECTS only (see that
        # constant's own comment). Cache is populated ONCE per episode (one
        # batched ER call, not one per retry attempt -- the same "cost
        # discipline" N2/N4's probes already established) by
        # _ensure_perception_push_targets(), called lazily from
        # _push_object_to() the first time an eligible object is pushed.
        self.push_perception_targets = push_perception_targets
        self._perception_push_target_cache: dict[
            str, tuple[float, float, float]
        ] = {}
        self._perception_push_attempted = False
        # Correction to the T5a REFUTED verdict (plans/SYNC.md 2026-08-04
        # ~14:45 UTC): navigate_to has never arrived at a curobo_stance_for
        # candidate in any run recorded since T4 -- T4's fix picks
        # candidates ~2-3m away, and this budget was hardcoded at 25.0s.
        # CLI/kwarg-overridable so it can be A/B'd without a code edit;
        # default stays 25.0 (unchanged behavior) until GPU-verified.
        self.push_stance_navigate_budget_s = push_stance_navigate_budget_s
        # Q5 (SYNC 21/25, curobo_stance.py originally :313): the batch-IK
        # stance search used to rate every candidate against the right
        # arm's base only. Default off -- unchanged behaviour.
        self.curobo_rate_both_arms = curobo_rate_both_arms
        # T5 (LOOP_PROMPT_VM_A_REV4.md): default True = unchanged behavior.
        # Lets the reach-limit pre-flight gate (_reach_limit_exceeded,
        # Q2/Q4) be A/B'd against reliability -- it is correct engineering
        # that may also be refusing the exact attempts that produced the
        # project's only point (handoff sec 105).
        self.reach_gate_enabled = reach_gate_enabled
        # REV16 Phase C: default-off. When True, reach()'s grasp-target
        # computation tries a perception-derived point (mask centroid +
        # principal-axis grasp, task3_autonomy/perception_grasp.py)
        # BEFORE the hand-fitted cup/object constant path -- on any
        # failure (annotator not ready, empty mask, no IK-feasible side)
        # it falls back to the existing constant path unchanged, same
        # contract as _stance_for's curobo/fallback split above. Nothing
        # in the live grasp path calls this unless the flag is set.
        self.perception_grasp = perception_grasp
        self._perception_render_product = None
        self._perception_seg_annotator = None
        self._perception_depth_annotator = None
        self._perception_cam_annotator = None
        self._perception_rgb_annotator = None
        self._perception_camera_prim_path = None
        self._perception_resolution = (960, 540)
        # 2026-08-14 (owner directive): ask Gemini ER-2 for the grasp POSE --
        # position and orientation -- live, once per grasp attempt, instead
        # of reading a frozen candidate file or commanding a fixed top-down
        # wrist. Off by default so every existing run is unchanged; enabled
        # by run_task3.py's --live-er-grasp.
        self.live_er_grasp = live_er_grasp
        # Stage two of the two-stage perception GOTCHAS prescribes. On by
        # default because the wide shot alone measurably misses (allobj_7
        # spoon2: miss_m 0.0495), and every failure path inside it falls
        # back to the wide-shot answer, so the worst case is today's
        # behaviour.
        self.er_grasp_crop_refine = er_grasp_crop_refine
        # cuMotion grasp planning: plan approach+grasp as joint trajectories
        # instead of servoing at a Cartesian target. Off by default; enabled
        # by run_task3.py's --curobo-grasp. One planner per side, built on
        # first use because construction loads the robot description.
        self.curobo_grasp = curobo_grasp
        self._curobo_planners: dict[str, Any] = {}

        self.head_placement = "a"
        self.seed = 0

        # Populated by reset().
        self._m: dict[str, Any] = {}
        self.sim = None
        self.scene = None
        self.robot = None
        self.adapter = None
        self.arms = None
        self.object_views: dict[str, Any] = {}
        self._tick_count = 0
        # sec 19b W1.1: heartbeat instrumentation, tracked since process
        # start (not since reset()) so the wall_s field answers "how long
        # has this process been alive", the same question §19's stalled
        # runs left unanswered for 8-17 minutes at a time.
        self._proc_start_time = time.time()
        self._last_heartbeat_tick = 0
        self._last_heartbeat_time = self._proc_start_time
        self._frames_written = 0
        self._rgb_annotator = None
        self._render_product = None
        self._base_hold_anchor: tuple[float, float] | None = None
        # Caller-tunable base-hold gains (see run_world_isaac_grasp.py's
        # --grasp-base-hold-kp / --base-hold-max-mps). Set here (not in
        # reset()) so a caller assigning these before reset() isn't
        # silently clobbered -- reset() previously reset _base_hold_kp to
        # 4.0 unconditionally after every override, which meant every
        # historical "Lever 2" kp8/kp12 run (handoff.md 4.5/4.6) actually
        # executed at kp=4.0 regardless of the CLI flag.
        self._base_hold_kp = 4.0
        self._base_hold_max_mps = 0.25
        self._held: str | None = None
        # WHICH ARM is holding `_held`. Without this, carry_object_to falls
        # back to a default side and can drive an empty arm -- see its own
        # side resolution for the measurement.
        self._held_side: str | None = None
        # The gripper opening the object stopped the jaws at, so the carry
        # can keep commanding it -- see carry_object_to's loop.
        self._held_gripper_rad: float | None = None
        # Set by grasp_bimanual() on a verified two-arm hold; carry_object_to
        # branches on this to re-command BOTH arms each tick instead of one.
        self._held_sides: tuple[str, str] | None = None
        self._held_gripper_rad_bimanual: dict[str, float] | None = None
        self._active_object: str | None = None
        self.phases: list[dict[str, Any]] = []
        # P0.7: score_stage(2)/(3) need feed_hold()/pour()'s own measured
        # outcome, which the WorldAdapter action/score split (score_stage
        # takes no arguments) does not otherwise carry across the call
        # boundary -- set by feed_hold()/pour(), read by score_stage().
        self._last_feed_result: dict[str, Any] | None = None
        self._last_pour_result: dict[str, Any] | None = None

    # ------------------------------------------------------------------ #
    # Scene lifecycle
    # ------------------------------------------------------------------ #

    def reset(self, *, seed: int, head_placement: str) -> None:
        m = _lazy_isaac_imports()
        self._m = m
        self.seed = seed
        self.head_placement = head_placement

        if self.skip_navigation:
            # Same clear rotation-safe spot verify_grasp_lift uses for fast
            # arm iteration -- >= 1.0 m radial clearance, close to the
            # kitchen stance so a short final leg still exercises real
            # navigate_to()/reach() code (not a teleport).
            spawn_position = (-3.0, -3.1, FULL_ROBOT_SPAWN_POSITION[2])
            spawn_yaw_deg = 180.0
        else:
            spawn_position = FULL_ROBOT_SPAWN_POSITION
            spawn_yaw_deg = FULL_ROBOT_SPAWN_YAW

        sim = m["SimulationContext"](
            m["sim_utils"].SimulationCfg(
                dt=0.005, device="cuda:0", gravity=(0.0, 0.0, -9.81)
            )
        )
        if self.simulation_app is None:
            raise RuntimeError(
                "IsaacWorld requires simulation_app (the AppLauncher().app "
                "object) -- construct AppLauncher before IsaacWorld."
            )
        m["configure_keyboard_control_stage"](
            m["configure_robot_room_stage"],
            self.simulation_app,
            sim.stage,
            room_path=REPO_ROOT / "assets" / "robot_room.usd",
            task="task3",
            head_placement=head_placement,
            robot_position=spawn_position,
            robot_yaw=spawn_yaw_deg,
            dynamic_beans=False,
        )

        object_paths = {
            name: m["prepare_rigid_body_view_path"](
                sim.stage, m["resolve_prim_path"](sim.stage, name)
            )
            for name in self.object_names
        }

        scene = m["InteractiveScene"](
            m["make_control_scene_cfg"](
                num_envs=1,
                robot_path=m["make_headless_robot_usd"](self._robot_usd_path),
                robot_position=spawn_position,
                robot_rotation=m["yaw_to_quat"](spawn_yaw_deg),
                gripper=(
                    self.gripper_profile.name
                    if self.gripper_profile is not None
                    else None
                ),
            )
        )
        m["_fix_single_articulation_root"](
            sim.stage, "/World/envs/env_0/Robot"
        )
        sim.reset()
        scene.reset()
        robot = scene["robot"]
        m["reset_robot_to_default_state"](robot, scene.env_origins)
        scene.write_data_to_sim()

        self._correct_scoring_object_collision(object_paths)

        object_views = {}
        for name, path in object_paths.items():
            view = m["RigidPrim"](prim_paths_expr=path, name=f"task3_{name}")
            initialize = getattr(view, "initialize", None)
            if callable(initialize):
                initialize()
            object_views[name] = view

        self.sim = sim
        self.scene = scene
        self.robot = robot
        self.object_views = object_views
        # Each object's height as the episode spawned it, and the floor's,
        # so `_object_has_fallen` can tell "still on its surface" from "on
        # the floor" without a magic threshold. Recorded here because this
        # is the only moment the scene is known-undisturbed.
        self._spawn_object_z: dict[str, float] = {}
        for name in object_views:
            try:
                self._spawn_object_z[name] = float(
                    self.object_position(name)[2]
                )
            except Exception:  # noqa: BLE001, S112
                continue
        self._tick_count = 0
        self._last_heartbeat_tick = 0
        self._last_heartbeat_time = time.time()
        self._frames_written = 0
        self._base_hold_anchor = None
        self._held = None
        self._active_object = None
        self._last_grasp_offset: dict[str, tuple[float, float, float]] = {}
        # 2026-08-14: the wrist yaw `descend` actually commanded for this
        # object, so grasp()'s pre-close re-center can reproduce the SAME
        # orientation instead of snapping back to yaw=0 and undoing the
        # ranked candidate's chosen grasp direction.
        self._last_grasp_yaw: dict[str, float] = {}
        # The FULL wrist quaternion `descend` commanded for this object, not
        # just its yaw. `_last_grasp_yaw` alone was enough only while every
        # approach was straight down; with a live ER-2 approach the roll is
        # one of three angles and rebuilding the orientation from it loses
        # the tilt entirely -- see grasp()'s pre-close re-center.
        self._last_grasp_quat: dict[
            str, tuple[float, float, float, float]
        ] = {}
        # Measured live by tool_offset()/_pad_body_ids(); never a constant.
        self._tool_offset_cache: dict[str, tuple[float, float, float] | None]
        self._tool_offset_cache = {}
        self._pad_body_ids_cache: dict[str, list[tuple[str, int]]] = {}
        # P5.1 (2026-08-11): pre-lift ee/object pose, keyed by object_name,
        # captured at the START of lift() -- hold()'s three_predicate_hold
        # evidence needs a "did the object actually rise since it was
        # grasped" baseline, not "did it rise during hold()'s own static
        # window" (hold() commands a STATIONARY target throughout, so a
        # rise measured only within hold() would be ~0 even for a real
        # hold). See hold()'s own comment for the full reasoning.
        self._pre_lift_baseline: dict[
            str, tuple[tuple[float, float, float], tuple[float, float, float]]
        ] = {}
        # REV16 Phase C follow-up (owner correction, 2026-08-09): populated
        # by _precompute_perception_grasp_targets() at the END of reset(),
        # on the MAIN thread -- see that method's docstring for why. Empty
        # (not None) when --perception-grasp is off or precompute found
        # nothing; _perception_grasp_target()'s cache-miss path already
        # treats that identically to "no candidate", falling back to the
        # constant path either way.
        self._perception_grasp_cache: dict[
            str, "perception_grasp.GraspCandidate | None"
        ] = {}
        # 2026-08-14: the live-capture handoff between the stage worker
        # thread and the main thread. See `request_live_capture` /
        # `service_main_thread_requests` for the whole mechanism and why it
        # has to exist at all. A plain unbounded Queue: at most one request
        # is ever outstanding, because the only caller blocks on its own
        # reply before issuing another.
        self._live_capture_requests: "queue.Queue[dict[str, Any]]" = (
            queue.Queue()
        )
        self._live_capture_seq = 0
        # NOTE: `_spawn_object_z` is deliberately NOT reset here. It is
        # populated earlier in this same method, right after `object_views`
        # is built, and an assignment at this point ran AFTER that and wiped
        # it every episode -- which is why `_object_has_fallen` never fired
        # once in a live run despite being correct and tested. `getattr` in
        # the predicate meant the bug degraded silently into "never skip"
        # instead of raising.
        # Set while a video-frame render is outstanding, so the worker
        # drops frames instead of queueing them. Doubles as that
        # request's own `done` Event -- the servicer clears it.
        self._video_request_pending = threading.Event()
        # reset() rebuilds the stage/object_views each episode, so a prior
        # episode's semantics-applied flag must not suppress re-applying
        # them here -- see _ensure_object_semantics's docstring.
        self._object_semantics_applied = False
        self.phases = []
        self._last_feed_result = None
        self._last_pour_result = None

        # P0.9 (handoff sec 16.10(D)/17.6): head/face contact-force sensor,
        # same ContactSensor pattern already used for the base/gripper
        # (scripts/task3/run_stage1_setup.py). No Stage-2 point may be
        # banked without this -- safety is a HARD FAIL (sec 1). Guarded like
        # its siblings: any failure leaves the sensor None rather than
        # crashing reset(), and _head_contact_force_n() reports that as
        # "unavailable", never a false-safe zero.
        self._head_contact_sensor = None
        try:
            from isaaclab.sensors import ContactSensor, ContactSensorCfg
            from isaaclab.sim.schemas import activate_contact_sensors

            head_prim_path = m["resolve_prim_path"](sim.stage, "head")
            activate_contact_sensors(
                head_prim_path, threshold=0.0, stage=sim.stage
            )
            self._head_contact_sensor = ContactSensor(
                ContactSensorCfg(
                    prim_path=head_prim_path,
                    update_period=0.0,
                    history_length=1,
                )
            )
            sim.reset()
            scene.reset()
        except Exception:  # pragma: no cover - GPU/API dependent
            self._head_contact_sensor = None

        # 2026-08-17 (owner directive: try close-on-contact for the
        # bimanual cup grasp): per-side gripper ContactSensor, mirroring
        # this same block's own _head_contact_sensor pattern above and
        # scripts/task3/run_stage1_setup.py's M1-V1 gripper sensor
        # (plans/handoff.md sec 15.6/15.2) -- but at THIS project's real
        # gripper='robotiq' prim scope, not that script's AG2F120S one.
        #
        # FIRST attempt (GPU-measured, outputs/task3_probe_gripper_contact/
        # log.txt) pointed the sensor at the {side}_Robotiq_2F_85 SCOPE
        # prim and failed outright: "could not find any bodies with
        # contact reporter API" -- that scope is a container Xform, not a
        # rigid body, so activate_contact_sensors had nothing to attach to.
        # The real rigid bodies are the finger pad links two levels down
        # (confirmed by dumping the live prim tree):
        # /World/envs/env_0/Robot/right_Robotiq_2F_85/right_Robotiq_2F_85/
        # {left,right}_inner_finger -- and the asset has a genuine,
        # separately-documented left-side naming typo (double underscore,
        # "left__Robotiq_2F_85", matching this asset's other known
        # left/right-name typos, e.g. "right_iight_2_link" at ~L1421) that
        # makes a hand-typed left-side path wrong. Reusing
        # scene_robot_room_keyboard.gripper_pad_prim_paths() -- the same
        # already-proven helper _pad_body_ids() uses -- sidesteps the typo
        # entirely by reading the real prim paths instead of guessing them,
        # then pointing the sensor at a regex covering both inner_finger
        # pads under that side's real parent scope (ContactSensorCfg's
        # prim_path accepts a regex expression matching multiple prims,
        # confirmed in contact_sensor_cfg.py's own docstring). Guarded
        # identically to _head_contact_sensor: any failure (wrong gripper
        # profile, prim missing, sensor API error) leaves the sensor None,
        # never a false-safe zero.
        self._gripper_contact_sensors: dict[str, Any] = {}
        for side in ("left", "right"):
            try:
                from isaaclab.sensors import ContactSensor, ContactSensorCfg
                from isaaclab.sim.schemas import activate_contact_sensors
                from scene_robot_room_keyboard import gripper_pad_prim_paths

                pad_paths = gripper_pad_prim_paths(
                    sim.stage, "/World/envs/env_0/Robot", side
                )
                if not pad_paths:
                    self._gripper_contact_sensors[side] = None
                    continue
                pad_parent = pad_paths[0].rsplit("/", 1)[0]
                gripper_prim_path = f"{pad_parent}/.*_inner_finger"
                activate_contact_sensors(
                    pad_parent, threshold=0.0, stage=sim.stage
                )
                self._gripper_contact_sensors[side] = ContactSensor(
                    ContactSensorCfg(
                        prim_path=gripper_prim_path,
                        update_period=0.0,
                        history_length=1,
                    )
                )
                sim.reset()
                scene.reset()
            except Exception:  # pragma: no cover - GPU/API dependent
                self._gripper_contact_sensors[side] = None

        if self.record_video:
            import omni.replicator.core as rep

            self.frames_dir.mkdir(parents=True, exist_ok=True)
            for stale in self.frames_dir.glob("rgb_*.png"):
                stale.unlink()
            camera = rep.create.camera(
                position=CAMERA_POSITION, look_at=CAMERA_LOOK_AT
            )
            self._render_product = rep.create.render_product(
                camera, (640, 360)
            )
            self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
            self._rgb_annotator.attach([self._render_product])

        self.adapter = m["TmrBaseAdapter"](robot, num_envs=1, device="cuda:0")
        self.arms = m["DualArmController"](
            robot,
            self.simulation_app,
            gripper=(
                self.gripper_profile.name
                if self.gripper_profile is not None
                else None
            ),
        )

        # Phase 0 (verify_grasp_lift-proven): raise the spine for transit
        # clearance, then tuck the arms into the transit pose BEFORE any
        # base motion -- this is what keeps the tucked arms from sweeping
        # the island counter during navigation.
        spine_ok = self.arms.move_spine(
            m["vgl"].TRAVEL_SPINE_M,
            step=self._tick,
            dt=sim.cfg.dt,
            timeout_s=6.0,
            tolerance_m=0.02,
        )
        m["ramp_arm_pose"](robot, m["TRANSIT_ARM_POSE"], step=self._tick)
        self.arms.sync_targets_from_measured()
        self._log_phase("reset_tuck", spine_ok, spine_ok=bool(spine_ok))

        # REV16 Phase C follow-up (owner correction, 2026-08-09): the
        # segmentation/depth/camera_params annotators may only be touched
        # from the MAIN thread (world_isaac.py's own sec 20b: the render
        # pipeline ticked cleanly ~1460 times during reset(), which runs on
        # the main thread, and froze on its first call from the stage
        # worker thread -- Kit's app-update pump is not documented
        # thread-safe off the AppLauncher/CUDA-context thread). reset()
        # itself runs on the main thread, so this is the last safe place to
        # read a real frame before orchestrator.py's _run_stage_isolated
        # hands stage execution to a worker thread. Objects have not moved
        # since spawn at this point (nothing has touched them yet), so a
        # grasp point computed now is exactly as valid as one computed at
        # first-reach time.
        self._precompute_perception_grasp_targets()

        # Warm the live ER-2 render pipeline here too, for the same reason
        # _precompute_perception_grasp_targets() does: annotator attach +
        # first-render is only safe on the MAIN thread, and reset() is the
        # last place that's true before stage execution hands off to the
        # worker. Without this, the first `request_live_capture()` during a
        # stage pays that cold-start cost (attach + Replicator's "populates
        # on the NEXT render pass" warm-up) against its own timeout_s -- and
        # if it doesn't finish in time, the worker abandons it and resumes
        # ticking physics while it is still in flight. Measured 2026-08-15,
        # keep_run_task3_liveer_cup.log: a live capture timed out after 20s,
        # the abandoned request was serviced 10 minutes later while the
        # worker was already deep in a grasp attempt, and the resulting
        # concurrent PhysX access crashed the process (double free, after a
        # burst of "PxDirectGPUAPI: not allowed while simulation is
        # running"). request_live_capture()'s own abandonment guard makes
        # that safe now regardless, but paying the cold-start cost here
        # means the first real attempt has a real chance of getting a live
        # frame instead of always timing out into the fallback.
        if self.live_er_grasp:
            self._ensure_perception_annotators()
            for _ in range(5):
                self.simulation_app.update()

    @property
    def _render_tick_stride(self) -> int:
        if self.__render_tick_stride is None:
            self.__render_tick_stride = render_tick_stride(
                self.sim.cfg.dt, VIDEO_FPS, self.record_video
            )
            print(
                "RENDER_TICK_STRIDE "
                f"stride={self.__render_tick_stride} dt={self.sim.cfg.dt} "
                f"video_fps={VIDEO_FPS} record_video={self.record_video} "
                f"(render passes cut to 1 in {self.__render_tick_stride})",
                flush=True,
            )
        return self.__render_tick_stride

    def _tick(self) -> None:
        m = self._m
        m["disable_robot_external_wrenches"](self.robot)
        if self._base_hold_anchor is not None:
            from task3_autonomy.navigation import base_twist_toward

            # min_creep_mps: this call never had it, and it is the fourth
            # place in this codebase caught by the same defect (see
            # carry_object_to's identical comment/fix, ~L5636) --
            # position_kp * distance decays toward zero as the base nears
            # its anchor, but the wheel drives have a ~2s velocity-tracking
            # lag (DRIVE_DAMPING=500), so a commanded speed that keeps
            # shrinking never lets real correcting motion start. Measured
            # here (scripts/task3/probe_base_hold_drift.py,
            # outputs/task3_base_hold_probe/): with this hold's own
            # position_kp=4.0, the dead zone is any error below ~2cm
            # (4.0*0.02=0.08), and a real per-tick trace through
            # reach_bimanual's pregrasp/standoff/gentle-ramp sequence showed
            # base_anchor_err_m oscillating between ~0.007m and ~0.05m
            # instead of converging and staying there -- a real, if mild,
            # underdamped hold, not the wrap-boundary yaw display artifact
            # this probe was originally launched to check (yaw itself was
            # confirmed smooth/monotonic through the same trace, not
            # oscillating). 0.08 m/s is the same proven value already used
            # at the other three call sites, not a fresh guess.
            hold_vx, hold_vy = base_twist_toward(
                self.adapter.pose(),
                self._base_hold_anchor,
                max_linear_mps=self._base_hold_max_mps,
                position_kp=self._base_hold_kp,
                min_creep_mps=0.08,
            )
            self.adapter.apply_twist(hold_vx, hold_vy, hold_heading=True)
        self.scene.write_data_to_sim()
        # sec 20b [OBSERVED live on a Lightning L4, 2026-07-26]: a faulthandler
        # dump on a genuinely stalled `--order 4 --skip-navigation` run showed
        # the orchestrator's stage worker thread (orchestrator.py's
        # `_run_stage_isolated`) frozen inside THIS call -- `sim.step()`'s
        # default `render=True` reaches
        # isaacsim.core.api.../simulation_context.py's `self._app.update()`
        # (confirmed by reading that file at the exact line the dump named).
        # The identical call ticked cleanly ~1460 times during reset() (which
        # runs on the MAIN thread) and froze on its very first call from the
        # stage worker thread. Kit's app-update pump is not documented as
        # thread-safe off the thread that owns the AppLauncher/CUDA context,
        # so only the main thread may call step(render=True); every other
        # thread gets the physics-only `_physics_context._step()` path
        # instead, skipping `_app.update()` entirely.
        # CAVEAT: this makes stage-plan ticks (all of which run on the worker
        # thread) physics-only -- `record_video`'s per-tick RGB capture below
        # needs the render pipeline to have actually run, so video capture
        # during stage execution (not just reset()) is NOT fixed by this
        # change and needs its own follow-up (handoff.md sec 20b).
        on_main_thread = threading.current_thread() is threading.main_thread()
        # Render on the cadence something actually READS a frame, not on
        # every physics tick -- see `render_tick_stride` for the measured
        # cost this removes. Worker-thread ticks are unchanged (they must
        # never render at all; see the note above).
        render_stride = self._render_tick_stride
        self.sim.step(
            render=on_main_thread and self._tick_count % render_stride == 0
        )
        self.scene.update(self.sim.cfg.dt)
        if self.record_video and self._rgb_annotator is not None:
            capture_every = max(1, round(1.0 / (self.sim.cfg.dt * VIDEO_FPS)))
            if self._tick_count % capture_every == 0:
                if on_main_thread:
                    # reset()'s own ticks: render already ran above, so the
                    # annotator has fresh data and this is the original path,
                    # unchanged.
                    if self._m["_save_rgb_frame"](
                        self._rgb_annotator,
                        self.frames_dir,
                        self._frames_written,
                    ):
                        self._frames_written += 1
                else:
                    # Stage ticks: nothing has rendered, so saving here would
                    # re-save the last rendered frame forever -- which is
                    # exactly what it used to do. Ask the main thread for a
                    # real render instead. Best-effort: if no supervisor is
                    # servicing requests the call times out and the episode
                    # continues without video rather than stalling on it.
                    self.request_video_frame()
        self._tick_count += 1
        if self._tick_count % 250 == 0:
            now = time.time()
            interval_ticks = self._tick_count - self._last_heartbeat_tick
            interval_s = now - self._last_heartbeat_time
            s_per_tick = interval_s / interval_ticks if interval_ticks else 0.0
            pose = self.adapter.pose()
            print(
                "WORLD_ISAAC_TICK "
                + str(
                    {
                        "tick": self._tick_count,
                        "wall_s": round(now - self._proc_start_time, 2),
                        "s_per_tick": round(s_per_tick, 5),
                        "x": round(pose.x, 4),
                        "y": round(pose.y, 4),
                        "yaw": round(pose.yaw, 4),
                    }
                ),
                flush=True,
            )
            self._last_heartbeat_tick = self._tick_count
            self._last_heartbeat_time = now

    def _arm_base_relative(
        self, side: str, target_world_xyz
    ) -> tuple[list[float], float] | None:
        """C2.5 Part A (REVIEW #3 sec R3.3 / plan sec 5 C2.5): the
        reachability audit needs every reach/push target's distance from
        the ARM's own mount frame (`{side}_base`), not the mobile base
        root or the world frame -- FR3's ~0.855m max reach is measured
        from there. Returns (relative_xyz, norm) or None if unavailable
        (e.g. mock world, no robot yet). Plain-Python quaternion rotation
        (this file has no top-level `import torch`, and this is only a
        3-vector rotate -- not worth adding one for).
        """
        if self.robot is None:
            return None
        if len(target_world_xyz) < 3:
            # navigate_to()'s phase logs a 2D [x, y] BASE target (not an
            # arm-reach target) through this same `target=` kwarg -- it is
            # not a reachability-relevant point and must not be treated as
            # one (confirmed live: this raised IndexError on
            # target_world_xyz[2] the first time navigate_to ran with this
            # instrumentation in place).
            return None
        try:
            body_names = list(self.robot.body_names)
            idx = body_names.index(f"{side}_base")
        except (ValueError, AttributeError):
            return None
        base_pos = self.robot.data.body_pos_w[0, idx].tolist()
        base_quat = self.robot.data.body_quat_w[0, idx].tolist()  # w,x,y,z
        dx = target_world_xyz[0] - base_pos[0]
        dy = target_world_xyz[1] - base_pos[1]
        dz = target_world_xyz[2] - base_pos[2]
        w, x, y, z = base_quat
        # Rotate (dx,dy,dz) by the INVERSE of the base's orientation
        # (conjugate quaternion, since it's a unit quaternion):
        # v' = v + 2w*(qv x v) + 2*(qv x (qv x v)),
        # with qv = (-x,-y,-z) for the inverse.
        qvx, qvy, qvz = -x, -y, -z
        # t = 2 * (qv x v)
        tx = 2.0 * (qvy * dz - qvz * dy)
        ty = 2.0 * (qvz * dx - qvx * dz)
        tz = 2.0 * (qvx * dy - qvy * dx)
        # v' = v + w*t + qv x t
        rx = dx + w * tx + (qvy * tz - qvz * ty)
        ry = dy + w * ty + (qvz * tx - qvx * tz)
        rz = dz + w * tz + (qvx * ty - qvy * tx)
        norm = math.sqrt(rx * rx + ry * ry + rz * rz)
        return ([round(rx, 4), round(ry, 4), round(rz, 4)], round(norm, 4))

    def _log_phase(self, name: str, ok: bool, **detail: Any) -> None:
        base = self.adapter.pose()
        entry = {
            "phase": name,
            "ok": bool(ok),
            "tick": self._tick_count,
            "base": [round(base.x, 3), round(base.y, 3), round(base.yaw, 3)],
        }
        if self._base_hold_anchor is not None:
            entry["base_anchor_err_m"] = round(
                math.hypot(
                    base.x - self._base_hold_anchor[0],
                    base.y - self._base_hold_anchor[1],
                ),
                4,
            )
        else:
            entry["base_anchor_err_m"] = None
        if self.arms is not None:
            entry["spine"] = round(self.arms.measured_spine_position(), 4)
            entry["target_spine"] = round(self.arms.spine, 4)
            left_ee, right_ee = self.arms.ee_world_poses()
            entry["left_ee"] = [round(v, 4) for v in left_ee[0]]
            entry["right_ee"] = [round(v, 4) for v in right_ee[0]]
        if (
            self._active_object is not None
            and self._active_object in self.object_views
        ):
            entry["obj"] = [
                round(v, 4) for v in self.object_position(self._active_object)
            ]
        entry.update(detail)
        # C2.5 Part A: every phase that logs a `target` (pregrasp,
        # descend_standoff, descend, push_approach, push_contact, ...)
        # gets the same target's distance from the arm's own mount frame
        # for free, no per-call-site edits needed. `side` defaults to
        # "right" -- every reach()/push call in this codebase today uses
        # the right arm only; pass side=<...> explicitly in **detail at
        # the call site if that ever changes.
        target = detail.get("target")
        if target is not None:
            side = detail.get("side", "right")
            rel = self._arm_base_relative(side, target)
            if rel is not None:
                entry["target_rel_arm_base_m"] = rel[0]
                entry["target_norm_from_arm_base_m"] = rel[1]
        self.phases.append(entry)
        print("WORLD_ISAAC_DBG " + str(entry), flush=True)

    # ------------------------------------------------------------------ #
    # Navigation
    # ------------------------------------------------------------------ #

    def navigate_to(self, x, y, yaw=None, **p) -> dict:
        # sec 21 Bug A (owner, 2026-07-26): _tick() unconditionally
        # re-issues a hold-toward-anchor twist whenever
        # self._base_hold_anchor is set, and this method sets that anchor
        # at the END of every call (so a later manipulation phase can hold
        # position) -- clear it here so a stale anchor from a PREVIOUS call
        # doesn't clobber this call's own twist every tick (carry_object_to
        # already does this before its own loop, below).
        self._base_hold_anchor = None

        # M1 (owner-assigned, plans/LOOP_PROMPT_VM_B.md rev 2): VM A traced a
        # scoring object (plate2) falling off the table DURING this call, not
        # during a push. reset() tucks the arm into TRANSIT_ARM_POSE before
        # any base motion specifically "to keep the tucked arms from sweeping
        # the island counter" (:368-370), and _push_object_to() (:1280-1288)
        # re-applies that SAME tuck before ITS OWN internal navigate_to call,
        # for exactly this reason -- but only for navigate_to calls made
        # WHILE working one object. This method is also called directly by
        # plan_stage1/plan_stage4's per-object "navigate" step, transitioning
        # to a NEW object right after the previous object's push_retract
        # left the arm extended up over the table (world_isaac.py
        # push_retract, :1581-1598, does not re-tuck). That un-tucked
        # transition is the one navigate_to call site the existing tuck
        # never covered. Apply it here, once, so every navigate_to call
        # gets the same guarantee reset()/_push_object_to already rely on --
        # idempotent (a no-op ramp) when the arm is already tucked.
        self._m["ramp_arm_pose"](
            self.robot, self._m["TRANSIT_ARM_POSE"], step=self._tick
        )
        self.arms.sync_targets_from_measured()

        objs_before = {
            name: [round(v, 4) for v in self.object_position(name)]
            for name in self.object_views
        }

        max_linear = p.get("max_linear_mps", 0.5)
        budget_s = p.get("budget_s", 45.0)
        trace_every_ticks = p.get("trace_every_ticks", 0)
        min_creep = p.get("min_creep_mps", 0.0)
        skill = self._m["NavigateTo"](
            (x, y),
            yaw,
            max_linear_mps=max_linear,
            min_creep_mps=min_creep,
        )
        watchdog = ProgressWatchdog()
        done = False
        stalled = False
        for tick_local in range(int(budget_s / self.sim.cfg.dt)):
            pose = self.adapter.pose()
            if trace_every_ticks and tick_local % trace_every_ticks == 0:
                objs_now = {
                    name: [round(v, 4) for v in self.object_position(name)]
                    for name in self.object_views
                }
                print(
                    f"NAVOBJTRACE tick={self._tick_count} "
                    f"pose=({pose.x:.4f},{pose.y:.4f},{pose.yaw:.4f}) "
                    f"objs={objs_now}",
                    flush=True,
                )
            if watchdog.sample(self._tick_count, pose.x, pose.y):
                stalled = True
                self.adapter.apply_twist(0.0, 0.0)
                self._tick()
                break
            vx, vy, done = skill.compute(pose)
            if done:
                self.adapter.apply_twist(0.0, 0.0)
                self._tick()
                break
            self.adapter.apply_twist(vx, vy)
            self._tick()
        else:
            self.adapter.apply_twist(0.0, 0.0)
            self._tick()
        final_pose = self.adapter.pose()
        err = math.hypot(x - final_pose.x, y - final_pose.y)
        self._base_hold_anchor = (final_pose.x, final_pose.y)

        objs_after = {
            name: [round(v, 4) for v in self.object_position(name)]
            for name in self.object_views
        }
        obj_z_delta = {
            name: round(objs_after[name][2] - objs_before[name][2], 4)
            for name in objs_before
        }
        self._log_phase(
            "navigate_to",
            done and not stalled,
            target=[round(x, 3), round(y, 3)],
            terminal_error_m=round(err, 4),
            stalled=stalled,
            all_objects_z_before={k: v[2] for k, v in objs_before.items()},
            all_objects_z_after={k: v[2] for k, v in objs_after.items()},
            obj_z_delta=obj_z_delta,
        )
        result = {"terminal_error_m": round(err, 4)}
        reach_object = p.get("reach_check_object")
        if reach_object is not None and reach_object in self.object_views:
            ox, oy, _ = self.object_position(reach_object)
            result["object_dist_m"] = round(
                math.hypot(final_pose.x - ox, final_pose.y - oy), 4
            )
        if stalled:
            result["stalled"] = True
            result["pose_trace"] = watchdog.pose_trace
        return result

    def navigate_to_avoiding_island(self, x, y, yaw=None, **p) -> dict:
        """Like `navigate_to`, but routes around the kitchen island first
        if the straight line from here to `(x, y)` would cross it.

        2026-08-09 (O1 investigation): originally inlined inside
        `_push_object_to` only (GPU-confirmed 3/3 there: a direct
        `navigate_to(*stance_xy)` drove straight through the island's real
        PhysX collider whenever the base and the target stance sat on
        opposite sides of it, `ProgressWatchdog` correctly firing on the
        real, near-zero displacement). Extracted here after the SAME
        failure recurred in `stages.py`'s own general "navigate" step
        (used by every Stage-4 object except `cup`, which has its own
        hand-fit safe stance -- `spoon2_run1.log`: `navigate_to` stalled
        at `terminal_error_m: 1.5` on the very first call, before
        `_push_object_to` ever ran) -- the island-crossing bug was never
        scoped to the push path specifically, it is a property of
        `navigate_to` itself whenever a straight line crosses the island.
        No-op (falls straight through to a single `navigate_to` call, byte
        -identical to before) unless `route_avoiding_island`'s exact
        straddle geometry is detected.
        """
        m = self._m
        current_xy = (self.adapter.pose().x, self.adapter.pose().y)
        route = m["route_avoiding_island"](current_xy, (x, y))
        result: dict = {}
        for waypoint in route[1:-1]:
            self.navigate_to(*waypoint, **p)
        result = self.navigate_to(x, y, yaw, **p)
        return result

    def _rotate_to(self, target_yaw: float, budget_s: float = 15.0) -> bool:
        # sec 21 Bug A's sibling (owner, 2026-07-26, confirmed live right
        # after Bug A's fix landed): same missing-anchor-clear defect as
        # navigate_to() -- TmrBaseAdapter.apply_twist's hold_heading=True
        # path (called by _tick()'s hold-anchor block, wz_cmd defaulting to
        # 0.0) damps/cancels this method's own commanded wz every tick
        # whenever a stale anchor is set (e.g. right after a preceding
        # navigate_to() call, which always sets one on success).
        self._base_hold_anchor = None
        # min_creep_radps: the rotational twin of the min_creep_mps already
        # wired into this class's navigate_to(). Without it a rotation that
        # starts just outside tolerance commands a rate the drives never
        # start from rest, and the watchdog reports a stall the base never
        # had -- see RotateTo.compute's docstring for the measured case
        # (4.13 degrees from a 4.0-degree tolerance, 0.0094 rad of motion in
        # 1000 ticks). 0.08 matches the linear creep already in use at
        # reach()'s stance approach.
        skill = self._m["RotateTo"](
            target_yaw,
            yaw_tolerance_rad=math.radians(4.0),
            min_creep_radps=0.08,
        )
        # Watchdog samples yaw (radians) through the same displacement-based
        # sample()/lookback logic navigate_to() and carry_object_to() use --
        # min_move_m compares against yaw distance here, not meters.
        watchdog = ProgressWatchdog(min_move_m=math.radians(1.0))
        for _ in range(int(budget_s / self.sim.cfg.dt)):
            pose = self.adapter.pose()
            if watchdog.sample(self._tick_count, pose.yaw, 0.0):
                self.adapter.apply_twist(0.0, 0.0, 0.0)
                self._tick()
                self._log_phase(
                    "_rotate_to",
                    False,
                    target_yaw=round(target_yaw, 3),
                    stalled=True,
                    pose_trace=watchdog.pose_trace,
                )
                return False
            wz, done = skill.compute(pose)
            if done:
                self.adapter.apply_twist(0.0, 0.0, 0.0)
                self._tick()
                return True
            self.adapter.apply_twist(0.0, 0.0, wz)
            self._tick()
        self.adapter.apply_twist(0.0, 0.0, 0.0)
        self._tick()
        return False

    # ------------------------------------------------------------------ #
    # Manipulation
    # ------------------------------------------------------------------ #

    def object_position(self, name: str) -> tuple[float, float, float]:
        positions, _ = self.object_views[name].get_world_poses()
        row = positions.tolist()[0]
        return (float(row[0]), float(row[1]), float(row[2]))

    def _pad_body_ids(self, side: str) -> list[tuple[str, int]]:
        """Articulation body indices of `side`'s two Robotiq `*_inner_finger`
        pads.

        Two documented traps are avoided here, both found 2026-08-13 and
        both recorded in `scene_robot_room_keyboard.py`:

        1. `robot.body_names.index(name)` is AMBIGUOUS. Both
           `left_Robotiq_2F_85` and `right_Robotiq_2F_85` contain a
           `left_inner_finger` AND a `right_inner_finger` (Robotiq's own
           left/right, no arm prefix), so `.index()` silently returns one
           arm's copy for both. That bug made a right-arm servo measure the
           LEFT arm's stationary finger through an entire sweep.
        2. `XformCache` is FROZEN for robot bodies. IsaacLab's Articulation
           owns the GPU tensor view and PhysX never writes those transforms
           back to USD, so an XformCache read stays at the rest pose while
           the arm really moves.

        The resolution is the same one that file settled on: take the
        correctly PATH-scoped prims, and disambiguate the name-collided body
        indices ONCE by comparing each candidate's tensor position against
        the prim's XformCache position at this instant -- accurate right
        now, before staleness can matter. Thereafter only the tensor state
        is read. No hardcoded index or ordering assumption.
        """
        # Lazily created: these methods are reachable on an IsaacWorld that
        # was constructed but never reset() (CPU tests do exactly that).
        if getattr(self, "_pad_body_ids_cache", None) is None:
            self._pad_body_ids_cache = {}
        cached = self._pad_body_ids_cache.get(side)
        if cached is not None:
            return cached
        # Measuring the pads is an accuracy improvement, never a
        # precondition -- a world without a real articulation (MockWorld,
        # every CPU test double) must behave exactly as it did before.
        try:
            _ = self.robot.body_names
        except Exception:  # noqa: BLE001 - any test double shape
            self._pad_body_ids_cache[side] = []
            return []
        # The default asset (assets/mobile_fr3_duo_v0_2.usd, gripper=None)
        # is NEITHER a Franka hand NOR the Robotiq-named asset. Measured
        # from the live articulation 2026-08-14, its per-arm gripper bodies
        # are `<side>_gripper_base` plus a 4-bar linkage named
        # `<side>_{left,right}_{1,2,support}_link` -- there is no
        # `*_leftfinger`/`*_rightfinger` and no `*_inner_finger` anywhere in
        # `robot.body_names`.
        #
        # 2026-08-16: the `_2_link` pair this method used to return is WRONG
        # -- `scripts/task3/find_moving_pads.py` swept the gripper fully
        # open to fully closed and measured `left_left_2_link`/
        # `left_right_2_link` moving 0.0032 m, essentially rigid, while
        # `left_left_support_link`/`left_right_support_link` moved 0.047 m
        # and 0.050 m over the same sweep -- a real, physically plausible
        # aperture. This independently confirms (and finally root-causes)
        # plans/SESSION_START_2026-08-16.md's own `PADS_DO_NOT_TRACK_JOINT`
        # finding: `pad_separation_m` had been reading exactly 0.034 m
        # across every commanded gripper angle in this project's entire
        # history, because the bodies read were never the ones the joint
        # actually moves. This is also
        # `scripts/task3/curobo/fr3_duo_left_arm.yml`'s `link_names` bug --
        # cuMotion's goalset was asking to place two RIGID, 0.034 m-apart
        # points into an 0.085 m-apart target, which is geometrically
        # impossible for any joint configuration and fully explains
        # "Goalset planning returned None" independent of stance or frame
        # (the frame conversion itself was separately verified correct,
        # scripts/task3/curobo/probe_frame_agreement.py, disagreement_m
        # 0.031). Every `object_follows_ee`/`pad_midpoint_to_object_m`
        # check that used this method inherited the same error.
        #
        # NOTE the asset also ships a typo: the right arm's `_2_link` body
        # is `right_iight_2_link`, not `right_right_2_link` -- assumed (not
        # yet independently confirmed) to affect `_support_link` the same
        # way, since both pairs come from the same asset-wide naming bug.
        # Matched explicitly rather than silently falling back to the other
        # arm's link, which is the exact class of bug this method exists to
        # avoid.
        names = list(self.robot.body_names)
        for pair in (
            (f"{side}_left_support_link", f"{side}_right_support_link"),
            (f"{side}_left_support_link", f"{side}_iight_support_link"),
        ):
            ids = [
                (n, names.index(n)) for n in pair if names.count(n) == 1
            ]
            if len(ids) == 2:
                self._pad_body_ids_cache[side] = ids
                return ids

        # 2026-08-16, gripper="robotiq" (task1_isaacsim/assets/Robotiq_2f_85
        # _with_d405_mobile_fr3_duo_v0_2.usd): the fallback below (XformCache
        # distance to a live tensor position) returned the SAME two indices
        # for BOTH side='left' and side='right' -- measured,
        # scripts/task3/probe_robotiq_ik_frame.py -- because that cache read
        # is frozen at the asset's authored rest pose (same defect already
        # documented above for the default asset) and both sides' stale
        # references happened to resolve to the same live candidate. This
        # asset's pads are named for FINGER side, not arm side
        # ("left_inner_finger"/"right_inner_finger" exist once per gripper,
        # i.e. per arm), so USD's own duplicate-name suffixing is the
        # unambiguous signal instead: `left_Robotiq_2F_85` is authored before
        # `right_Robotiq_2F_85` everywhere in this asset (confirmed by body
        # order: left_base=27 before right_base=28, left_fr3v2_link0=30
        # before right_fr3v2_link0=31, ...), so the LEFT arm's gripper gets
        # the first (unsuffixed) copy of these names and the RIGHT arm's
        # gripper gets Isaac Sim's auto-suffixed "_0" copy. Confirmed by
        # scripts/task3/dump_robotiq_body_names.py: right_inner_finger=88,
        # left_inner_finger=89 (left arm); right_inner_finger_0=91,
        # left_inner_finger_0=92 (right arm).
        for pair in (
            (
                ("left_inner_finger", "right_inner_finger")
                if side == "left"
                else ("left_inner_finger_0", "right_inner_finger_0")
            ),
        ):
            ids = [
                (n, names.index(n)) for n in pair if names.count(n) == 1
            ]
            if len(ids) == 2:
                self._pad_body_ids_cache[side] = ids
                return ids

        from pxr import UsdGeom
        from scene_robot_room_keyboard import gripper_pad_prim_paths

        paths = gripper_pad_prim_paths(
            self.sim.stage, "/World/envs/env_0/Robot", side
        )
        resolved: list[tuple[str, int]] = []
        if paths:
            cache = UsdGeom.XformCache()
            for path in paths:
                prim = self.sim.stage.GetPrimAtPath(path)
                name = prim.GetName()
                ref = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
                candidates = [
                    i for i, n in enumerate(self.robot.body_names) if n == name
                ]
                if not candidates:
                    continue
                best_i, best_d = candidates[0], float("inf")
                for i in candidates:
                    bp = self.robot.data.body_pos_w[0][i]
                    d = (
                        (float(bp[0]) - ref[0]) ** 2
                        + (float(bp[1]) - ref[1]) ** 2
                        + (float(bp[2]) - ref[2]) ** 2
                    )
                    if d < best_d:
                        best_i, best_d = i, d
                resolved.append((name, best_i))
        self._pad_body_ids_cache[side] = resolved
        return resolved

    def tool_offset(
        self, side: str, refresh: bool = False
    ) -> tuple[float, float, float] | None:
        """The REAL gripper pad midpoint, expressed in the frame the IK
        actually solves for. `None` if the pads cannot be read.

        `scripts/common/dual_arm_lula.py` solves for
        `LEFT_END_EFFECTOR = "left_fr3v2_hand_tcp"` out of
        `mobile_fr3_duo_v0_2_franka_hand.urdf`, whose `hand_tcp` joint is a
        FRANKA hand's `xyz="0 0 0.1034"`. The simulated robot is a Robotiq
        2F-85 whose fingers are `*_inner_finger` links, so `hand_tcp` is a
        frame no physical link tracks -- every reach residual this project
        has ever measured was the distance from the object to a point in
        empty space (see `scripts/task3/calibrate_tool_frame.py`, which
        measures the same quantity through ROS).

        Measured, never hardcoded: both frames are rigidly fixed to the
        same wrist, so this is a constant of the ASSET, and reading it live
        keeps it correct if the gripper, mount or URDF ever changes. That
        matters because every stage's perception feeds these same targets.
        """
        if getattr(self, "_tool_offset_cache", None) is None:
            self._tool_offset_cache = {}
        if not refresh and side in self._tool_offset_cache:
            return self._tool_offset_cache[side]
        pads = self._pad_body_ids(side)
        if len(pads) != 2:
            self._tool_offset_cache[side] = None
            return None
        mid = [0.0, 0.0, 0.0]
        for _, idx in pads:
            bp = self.robot.data.body_pos_w[0][idx]
            for a in range(3):
                mid[a] += float(bp[a]) / 2.0
        tcp_pos, tcp_quat = self.arms.ee_world_poses()[
            0 if side == "left" else 1
        ]
        from teleop_targets import Pose, pose_world_to_base

        rel = pose_world_to_base(
            Pose(tuple(mid), (1.0, 0.0, 0.0, 0.0)), tcp_pos, tcp_quat
        )
        offset = tuple(float(v) for v in rel.position)
        self._tool_offset_cache[side] = offset
        return offset

    def tcp_target_for_pads(
        self, side: str, pad_xyz, quat_wxyz
    ) -> tuple[float, float, float]:
        """Where to command `hand_tcp` so the real finger PADS land on
        `pad_xyz` at orientation `quat_wxyz`.

        `arms.reach`/`set_arm_target` command the IK frame
        (`<side>_fr3v2_hand_tcp`), but a grasp point is a statement about
        the fingers. Measured 2026-08-14 on the default asset: the pad
        midpoint is [0, 0, +0.0186] m in the tcp frame, rigid to 0.0 m
        stdev across four arm poses. Nothing in this pipeline accounted for
        it, so every commanded grasp point was ~18.6 mm short along the
        tool axis -- with the +0.075 m standoff that placed the pads about
        3.3 cm above `plate2`'s 0.770 m top surface, which is what closing
        to 0.06 rad on empty air looks like.

        Derived every run by `tool_offset()`, never stored: if the gripper,
        mount or URDF changes, this follows. Falls back to an unchanged
        `pad_xyz` when the pads cannot be measured, so no path can regress
        to worse-than-before behaviour.
        """
        offset = self.tool_offset(side)
        if offset is None:
            return tuple(float(v) for v in pad_xyz)
        from teleop_targets import Pose, pose_base_to_world

        # pads_world = tcp + R(quat) @ offset, so tcp = pad_xyz - R @ offset.
        # Rotating the offset about the origin gives exactly R @ offset.
        rotated = pose_base_to_world(
            Pose(tuple(offset), (1.0, 0.0, 0.0, 0.0)),
            (0.0, 0.0, 0.0),
            tuple(float(v) for v in quat_wxyz),
        ).position
        return (
            float(pad_xyz[0]) - float(rotated[0]),
            float(pad_xyz[1]) - float(rotated[1]),
            float(pad_xyz[2]) - float(rotated[2]),
        )

    def _stance_for(
        self,
        object_xy: tuple[float, float],
        approach: str,
        contact_z: float | None = None,
        stance_radius_m: float | None = None,
        stance_max_radius_m: float | None = None,
    ):
        """Compute a base stance for `object_xy`.

        GATE B0 (handoff sec 68): tries the cuRobo batch-IK stance search
        first (sec 64 proved navigation.stance_for()'s FIXED reach radius
        is the root cause of most Stage 1/4 reach failures; sec 65-67
        proved this search beats it). Only falls back to the old
        navigation.stance_for() -- reused the base-relative offset that
        made verify_grasp_lift.py's cup grasp work 10/10, clamped clear of
        the kitchen island footprint -- when the search is disabled, fails
        to build (e.g. cuRobo/warp import issue), or finds no practically
        reachable candidate. A search miss must never abort a stage.

        `contact_z` (N2, handoff sec 93/94): the push path's contact
        height, when given, is validated ALONGSIDE the pregrasp height so
        a stance is only accepted if the motion actually needed at the
        end of `_push_object_to` is reachable, not just the pregrasp
        approach. `reach()`'s own grasp call site never passes this --
        unaffected.

        `stance_radius_m` (R7 T2): forwarded only to the fallback path
        (`navigation.stance_for`) -- CHOSE (the curobo branch) has never
        won a single push candidate this project (R7 finding 5, `SYNC.md`
        2026-08-04 ~19:48 UTC), so its own search is left untouched rather
        than guessing how a radius override should interact with it.
        `None` (default) is the original, unaffected radius.
        """
        if self.use_curobo_stance:
            try:
                if self._curobo_stance_search is None:
                    from task3_autonomy.curobo_stance import (
                        CuroboStanceSearch,
                    )

                    self._curobo_stance_search = CuroboStanceSearch(
                        self, rate_both_arms=self.curobo_rate_both_arms
                    )
                result = self._curobo_stance_search.stance_for(
                    object_xy, approach, contact_z=contact_z
                )
                if result is not None:
                    return result
            except Exception as exc:  # noqa: BLE001
                import traceback

                print(
                    f"WARN: curobo stance search raised ({exc!r}), "
                    "falling back to navigation.stance_for()",
                    flush=True,
                )
                traceback.print_exc()
        # `stance_max_radius_m=None` means the stance is never placed
        # farther from the object than `stance_radius_m` allows. Only the
        # push path passes a ceiling (its 0.48 m annulus is entirely inside
        # the counter, so it cannot get a stance any other way); the grasp
        # path must not, because `reach()` is calibrated at exactly
        # STANCE_REACH_RADIUS_M. See navigation._rotate_to_clear_island.
        return _stance_for_impl(
            object_xy,
            approach,
            radius_m=stance_radius_m,
            max_radius_m=stance_max_radius_m,
        )

    _WRIST_CAMERA_PATHS = {
        "left": (
            "/World/envs/env_0/Robot/left_d405_camera_with_mount"
            "/d405_camera_link/left_Camera"
        ),
        "right": (
            "/World/envs/env_0/Robot/right_d405_camera_with_mount"
            "/d405_camera_link/right_Camera"
        ),
    }

    def verify_grasp_by_wrist_camera(
        self, side: str, object_name: str, *, min_pixel_frac: float = 0.02
    ) -> dict:
        """Owner directive (2026-08-19): grip must be confirmed by a SECOND
        camera -- the gripping arm's own wrist cam -- not by contact-force/
        object_follows_ee telemetry alone. Uses the semantic segmentation
        already applied per-object by `_ensure_object_semantics()` (each
        object's prim is labeled `class=object_name`): attaches a
        segmentation annotator to `side`'s wrist camera, and checks whether
        `object_name`'s label occupies a real fraction of the central
        (between-the-pads) region of that camera's view. Returns
        `{"verified": bool, "pixel_frac": float, "reason": str}` -- never
        raises; a missing/failed segmentation read is `verified=False`, not
        an exception, so a caller's retry loop degrades safely.
        """
        self._ensure_object_semantics()
        try:
            import omni.replicator.core as rep
        except Exception as exc:  # noqa: BLE001
            return {
                "verified": False,
                "pixel_frac": 0.0,
                "reason": f"replicator_unavailable:{exc!r}",
            }
        cam_path = self._WRIST_CAMERA_PATHS.get(side)
        if cam_path is None:
            return {
                "verified": False,
                "pixel_frac": 0.0,
                "reason": f"unknown_side:{side!r}",
            }
        attr = f"_grip_verify_annotator_{side}"
        annotator = getattr(self, attr, None)
        if annotator is None:
            try:
                render_product = rep.create.render_product(
                    cam_path, (320, 180)
                )
                annotator = rep.AnnotatorRegistry.get_annotator(
                    "semantic_segmentation"
                )
                annotator.attach([render_product])
                setattr(self, attr, annotator)
            except Exception as exc:  # noqa: BLE001
                return {
                    "verified": False,
                    "pixel_frac": 0.0,
                    "reason": f"annotator_attach_failed:{exc!r}",
                }
        for _ in range(5):
            self.simulation_app.update()
        try:
            data = annotator.get_data()
            seg = data["data"]
            id_to_labels = data["info"]["idToLabels"]
        except Exception as exc:  # noqa: BLE001
            return {
                "verified": False,
                "pixel_frac": 0.0,
                "reason": f"segmentation_read_failed:{exc!r}",
            }
        target_id = None
        for id_str, label_info in id_to_labels.items():
            if label_info.get("class") == object_name:
                target_id = int(id_str)
                break
        if target_id is None or getattr(seg, "size", 0) == 0:
            return {
                "verified": False,
                "pixel_frac": 0.0,
                "reason": "object_not_in_segmentation",
            }
        h, w = seg.shape[0], seg.shape[1]
        cy0, cy1 = int(h * 0.25), int(h * 0.75)
        cx0, cx1 = int(w * 0.25), int(w * 0.75)
        center = seg[cy0:cy1, cx0:cx1]
        pixel_frac = float((center == target_id).mean()) if center.size else 0.0
        verified = pixel_frac >= min_pixel_frac
        return {
            "verified": verified,
            "pixel_frac": round(pixel_frac, 5),
            "reason": "ok" if verified else "object_not_visible_in_grip_zone",
        }

    def _ensure_object_semantics(self) -> None:
        """REV16 Phase C / O3 follow-up (2026-08-09): apply a semantic
        label to each of the 4 target objects' prims, once per episode.

        GPU-confirmed (proofs/2026-08-09_o3_perception_pump/): the
        segmentation frame renders real pixel data (the render-pump fix
        above), but `idToLabels` only ever contains `BACKGROUND`/
        `UNLABELLED` -- confirmed via a CPU-only USD scan that NONE of
        `plate2`/`cup`/`bowl2`/`spoon2` (or any descendant) carries a
        semantic schema/attribute anywhere in `assets/robot_room.usd`, and
        no code in this repo ever applied one at runtime either.
        `instance_segmentation_fast` reports by semantic label, so without
        this, objects are structurally invisible to it regardless of
        render timing. The exact runtime helper module differs across
        Isaac Sim versions (`isaacsim.core.utils.semantics` vs the older
        `omni.isaac.core.utils.semantics`), and this image's `extscache`
        listing didn't resolve it by inspection alone -- try both, then
        fall back to the raw `pxr.Semantics` API directly (available once
        the full Kit app has booted, unlike the standalone CPU-only USD
        scan used to diagnose this, which lacked that extension). Never
        raises -- if every path fails, objects stay unlabeled exactly as
        today; `_precompute_perception_grasp_targets`'s existing
        cache-miss fallback already covers that case.
        """
        if getattr(self, "_object_semantics_applied", False):
            return
        self._object_semantics_applied = True
        add_update_semantics = None
        for module_name in (
            "isaacsim.core.utils.semantics",
            "omni.isaac.core.utils.semantics",
        ):
            try:
                import importlib

                module = importlib.import_module(module_name)
                add_update_semantics = module.add_update_semantics
                break
            except Exception:  # noqa: BLE001
                continue
        applied = []
        failed = []
        for object_name, view in self.object_views.items():
            prim_paths = getattr(view, "prim_paths", None)
            if not prim_paths:
                failed.append(object_name)
                continue
            try:
                prim = self.sim.stage.GetPrimAtPath(prim_paths[0])
                if add_update_semantics is not None:
                    add_update_semantics(
                        prim, semantic_label=object_name, type_label="class"
                    )
                else:
                    from pxr import Semantics

                    sem_api = Semantics.SemanticsAPI.Apply(prim, "Semantics")
                    sem_api.CreateSemanticTypeAttr().Set("class")
                    sem_api.CreateSemanticDataAttr().Set(object_name)
                applied.append(object_name)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARN: could not apply semantics to {object_name!r} "
                    f"({exc!r}), segmentation will not label it this "
                    "episode",
                    flush=True,
                )
                failed.append(object_name)
        print(
            f"DIAG object semantics: helper={add_update_semantics!r} "
            f"applied={applied} failed={failed}",
            flush=True,
        )

    # Mount path of the D405 on the left gripper base, as found in the
    # composed stage. Kept as a suffix match rather than an absolute path so
    # it survives the env_0 prefix and any re-parenting of the gripper.
    WRIST_CAMERA_MOUNT_SUFFIX = "left_gripper_base/_20s_camera_stand/d405"

    def _create_head_camera(self, aim_world) -> str | None:
        """Author a camera at the robot's HEAD mount, aimed at `aim_world`.

        WHY THE HEAD AND NOT THE WRIST. The robot has three camera mount
        points, matching the three cameras
        `scene_robot_room_keyboard.ROBOTIQ_CAMERA_RELATIVE_PATHS` documents:
        a head/ZED mount and two D405 wrist mounts. The wrist mounts sit
        INSIDE the gripper housings -- four separate attempts to place a
        camera there rendered the housing interior, the gripper filling the
        frame, a view aimed above everything, and the inside of a solid.
        `head_camera_mounting_point` is a top-level articulation body at
        z=1.29 m with its own fixed joint: it clears the grippers, and being
        a real body its pose comes from the tensor API rather than the
        XformCache that is frozen for robot bodies.

        THE POSE IS CHOSEN BY LOOKING, not derived and not guessed.
        `scripts/task3/probe_head_camera_poses.py` renders a grid of
        candidates in one run; of twelve, the two that actually show the
        counter, the tray and the cup are `elev=0.15, back=0.10`. That is
        the mount position raised 0.05 m, aimed at the object. The first
        sweep aimed at `base + forward*0.7` and rendered floor in all twelve
        -- the robot's yaw was -1.570 while the objects sit at +X, so
        "forward" pointed at empty tile. Aim at the OBJECT.
        """
        try:
            from omni.usd import get_context
            from pxr import Gf, UsdGeom

            names = list(self.robot.body_names)
            if HEAD_CAMERA_BODY not in names:
                return None
            idx = names.index(HEAD_CAMERA_BODY)
            head = self.robot.data.body_pos_w[0, idx].tolist()
            eye = [
                float(head[0]),
                float(head[1]),
                float(head[2]) + HEAD_CAMERA_RISE_M,
            ]

            stage = get_context().get_stage()
            cam_path = "/World/task3_head_camera"
            prim = stage.GetPrimAtPath(cam_path)
            if not prim or not prim.IsValid():
                cam = UsdGeom.Camera.Define(stage, cam_path)
                cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))
            else:
                cam = UsdGeom.Camera(prim)

            forward = [float(a) - e for a, e in zip(aim_world, eye)]
            n = math.sqrt(sum(v * v for v in forward)) or 1.0
            forward = [v / n for v in forward]
            up_hint = (0.0, 0.0, 1.0)
            right = [
                forward[1] * up_hint[2] - forward[2] * up_hint[1],
                forward[2] * up_hint[0] - forward[0] * up_hint[2],
                forward[0] * up_hint[1] - forward[1] * up_hint[0],
            ]
            rn = math.sqrt(sum(v * v for v in right))
            if rn < 1e-6:
                right, rn = [1.0, 0.0, 0.0], 1.0
            right = [v / rn for v in right]
            up = [
                right[1] * forward[2] - right[2] * forward[1],
                right[2] * forward[0] - right[0] * forward[2],
                right[0] * forward[1] - right[1] * forward[0],
            ]
            # USD cameras look down their own -Z. The camera is NOT parented
            # to the mount: it is re-placed from the live body pose on every
            # capture, which keeps it correct as the base drives without
            # depending on any USD transform staying in sync.
            xf = UsdGeom.Xformable(cam.GetPrim())
            xf.ClearXformOpOrder()
            xf.AddTransformOp().Set(
                Gf.Matrix4d(
                    right[0], right[1], right[2], 0.0,
                    up[0], up[1], up[2], 0.0,
                    -forward[0], -forward[1], -forward[2], 0.0,
                    eye[0], eye[1], eye[2], 1.0,
                )
            )
            return cam_path
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARN: head camera placement failed ({exc!r}); "
                "falling back to the scene camera",
                flush=True,
            )
            return None

    def _ensure_perception_annotators(self) -> None:
        """Lazily create a camera + render product carrying the three
        annotators `task3_autonomy/perception_grasp.py` needs (REV16
        C.0): ground-truth instance segmentation, z-depth, and the raw
        view/projection matrices. Independent of `record_video`'s own
        camera/render product -- this one is framed on the counter
        (`sim_camera_perception.PERCEPTION_CAMERA_POSITION/LOOK_AT`, the
        same placement the M1 OWL-ViT backend already uses), not the
        whole-room framing `record_video` wants. Built once; safe to call
        every `reach()` attempt.
        """
        if self._perception_render_product is not None:
            return
        # REV16 Phase C.4: omni.replicator.core is a Kit extension, not a
        # plain package -- it is only importable once ENABLED (being
        # *registered*, which happens during app boot regardless of any
        # flag, is not the same thing). GPU-verified 2026-08-08: a bare
        # import here raised ModuleNotFoundError every time (safely caught
        # by _perception_grasp_target's own try/except, so the fallback
        # always worked, but the perception path itself never ran,
        # proofs/2026-08-08_rev16/phase_c_validation/). The fix is NOT an
        # enable_extension() call here -- this method runs on the stage
        # worker thread (orchestrator.py's _run_stage_isolated), and
        # enabling a Kit extension mid-episode from a non-main thread hung
        # for 10+ minutes on omni.kit.material.library's startup (a
        # coroutine scheduled for the main thread's event loop that a
        # worker thread can never pump -- GPU-verified, same session).
        # The real fix is `_app_launcher_config`'s `enable_cameras` in
        # run_task3.py, which now also follows `--perception-grasp`: that
        # runs the extension enable at AppLauncher time, on the main
        # thread, before any stage worker thread exists -- exactly how
        # `--record-video` already does it safely. By the time this method
        # runs, the extension is already enabled and this import is a
        # normal, fast, safe import.
        import omni.replicator.core as rep

        from task3_pipeline.sim_camera_perception import (
            PERCEPTION_CAMERA_LOOK_AT,
            PERCEPTION_CAMERA_POSITION,
        )

        # WRIST CAMERA FIRST, scene camera only as a fallback.
        #
        # The robot carries a real D405 mount on its gripper base
        # (`.../left_gripper_base/_20s_camera_stand/d405`, found by
        # `scripts/task3/probe_robot_cameras.py`) but NO camera sensor prim:
        # every hit there is Xform/Mesh geometry, and the only
        # `UsdGeom.Camera` prims in the whole stage are Kit's four default
        # viewport cameras. So the hardware is modelled and nothing renders
        # from it -- which is why grasping has been driven by a free-floating
        # scene camera pinned at PERCEPTION_CAMERA_POSITION that does not
        # move with the robot.
        #
        # That matters for accuracy, not tidiness. GOTCHAS' ER-2 section:
        # "Wide-shot pointing lands ~5.7 cm off. Wide shot for semantics,
        # cropped or wrist-camera view (~15 cm) for geometry." Every grasp
        # point so far has been a wide shot's grasp point. It is also what
        # the organisers themselves record -- their datasets ship
        # `observation.images.wrist_left` and `wrist_right` beside the head
        # view.
        #
        # An earlier enumeration reported "zero cameras on the robot" and it
        # was wrong for a subtle reason worth keeping: it did not traverse
        # INSTANCE PROXIES, and the robot is instanced.
        # WRIST CAMERA: implemented, calibrated from the robot's own pad
        # geometry, and DEFAULT OFF -- see USE_WRIST_CAMERA.
        camera = None
        if USE_HEAD_CAMERA:
            try:
                aim = self.object_position(
                    self._active_object or config.STAGE1_OBJECTS[0]
                )
            except Exception:  # noqa: BLE001
                aim = PERCEPTION_CAMERA_LOOK_AT
            camera = self._create_head_camera(aim)
            if camera is not None:
                print(
                    f"PERCEPTION_CAMERA head mount, aimed at "
                    f"{[round(float(v), 3) for v in aim]}",
                    flush=True,
                )
        if camera is None:
            print(
                "PERCEPTION_CAMERA falling back to the fixed scene camera "
                "(no wrist mount found)",
                flush=True,
            )
            camera = rep.create.camera(
                position=PERCEPTION_CAMERA_POSITION,
                look_at=PERCEPTION_CAMERA_LOOK_AT,
            )
        # Keep the resolved prim path: `_perception_camera_pose()` reads the
        # camera's authored world transform off it every capture, which is
        # what lets the live ER-2 back-projection stay correct even if the
        # camera is later parented to a moving link (a robot-mounted head
        # camera, matching the organisers' own `observation.images.head`).
        self._perception_camera_prim_path = _resolve_camera_prim_path(camera)
        self._perception_render_product = rep.create.render_product(
            camera, self._perception_resolution
        )
        # RGB, for the live ER-2 frame. The three annotators below feed
        # `perception_grasp`'s mask/geometry path and cannot serve ER-2: it
        # needs a picture, not a segmentation buffer.
        self._perception_rgb_annotator = rep.AnnotatorRegistry.get_annotator(
            "rgb"
        )
        self._perception_rgb_annotator.attach(
            [self._perception_render_product]
        )
        self._perception_seg_annotator = rep.AnnotatorRegistry.get_annotator(
            "instance_segmentation_fast"
        )
        self._perception_seg_annotator.attach(
            [self._perception_render_product]
        )
        self._perception_depth_annotator = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        self._perception_depth_annotator.attach(
            [self._perception_render_product]
        )
        self._perception_cam_annotator = rep.AnnotatorRegistry.get_annotator(
            "camera_params"
        )
        self._perception_cam_annotator.attach(
            [self._perception_render_product]
        )

    # ------------------------------------------------------------------
    # Live capture: the worker thread asks, the main thread renders.
    #
    # Why this exists at all. `reach()` runs on the stage worker thread
    # (orchestrator._run_stage_isolated). Rendering and every Replicator
    # annotator read may only happen on the MAIN thread -- GPU-verified, and
    # the reason `_precompute_perception_grasp_targets` exists: a worker-
    # thread call FROZE, because Kit's app-update pump is a coroutine only
    # the main thread's event loop can drain. Precomputing at reset() was the
    # workaround, and it is exactly what the owner ruled out on 2026-08-14 --
    # a grasp pose computed once, before the robot has even navigated, is a
    # frozen artifact no matter how it was produced.
    #
    # So the render is moved to the thread that is allowed to do it, instead
    # of the decision being moved off the critical path. The worker parks on
    # an Event and touches NOTHING while the main thread renders, so the
    # "never two threads inside self.world" rule in _run_stage_isolated's
    # docstring still holds -- the two threads are strictly interleaved, not
    # concurrent.
    # ------------------------------------------------------------------

    def request_live_capture(
        self, object_name: str, timeout_s: float = 20.0
    ) -> dict[str, Any] | None:
        """WORKER-THREAD side. Ask the main thread for a frame; block for it.

        Returns the capture payload, or ``None`` on timeout / if no main-
        thread servicer is running. Never raises and never aborts a grasp:
        every caller treats ``None`` as "no live answer this attempt" and
        falls back, the same contract `_perception_grasp_target` already has.
        """
        self._live_capture_seq += 1
        request: dict[str, Any] = {
            "kind": "er_frame",
            "seq": self._live_capture_seq,
            "object_name": object_name,
            "done": threading.Event(),
            "payload": None,
            "error": None,
        }
        self._live_capture_requests.put(request)
        if not request["done"].wait(timeout_s):
            # CRASH ROOT CAUSE, 2026-08-15 keep_run_task3_liveer_cup.log:
            # giving up here without marking the request used to leave it
            # sitting in the queue. This worker thread then resumes ticking
            # (stepping physics) while the request is still live; if the
            # main thread's supervision loop is slow to reach it, it later
            # dequeues this same request and calls simulation_app.update()
            # from _capture_live_frame while the worker is concurrently
            # inside sim.step() -- confirmed by a faulthandler dump showing
            # both threads inside PhysX-touching calls at once, followed by
            # a burst of "PxDirectGPUAPI: not allowed while simulation is
            # running" and a double-free abort. Marking it abandoned lets
            # service_main_thread_requests skip the render instead of
            # racing it against this thread's now-resumed physics stepping.
            request["abandoned"] = True
            print(
                "LIVE_CAPTURE_TIMEOUT "
                f"object={object_name!r} waited_s={timeout_s:.1f} "
                "note='no main-thread servicer, or the render stalled; "
                "falling back to the non-live grasp path'",
                flush=True,
            )
            return None
        if request["error"] is not None:
            print(
                f"LIVE_CAPTURE_FAILED object={object_name!r} "
                f"error={request['error']!r}",
                flush=True,
            )
            return None
        return request["payload"]

    def service_main_thread_requests(self, budget_s: float = 2.0) -> int:
        """MAIN-THREAD side. Drain any pending capture requests.

        Called from the orchestrator's stage-supervision loop between short
        joins. Returns how many requests were serviced so the caller can
        distinguish "did work" from "idle" without inspecting the queue.

        Never raises: a capture failure is recorded on the request and handed
        back to the worker, which falls back. A raise here would kill the
        supervision loop and take the whole episode with it.
        """
        serviced = 0
        deadline = time.monotonic() + budget_s
        while time.monotonic() < deadline:
            try:
                request = self._live_capture_requests.get_nowait()
            except queue.Empty:
                break
            try:
                if request.get("abandoned"):
                    # The worker already timed out waiting and has resumed
                    # ticking (stepping physics) on its own thread. Calling
                    # simulation_app.update() here now would render
                    # concurrently with that -- the exact race that produced
                    # the PxDirectGPUAPI "not allowed while simulation is
                    # running" burst and the double-free abort in
                    # keep_run_task3_liveer_cup.log. Nobody is waiting on
                    # this request's payload; skip the render entirely.
                    request["error"] = "abandoned: worker already timed out"
                elif request["kind"] == "video_frame":
                    request["payload"] = self._capture_video_frame()
                else:
                    request["payload"] = self._capture_live_frame(
                        request["object_name"]
                    )
            except BaseException as exc:  # noqa: BLE001
                # BaseException, not Exception: SimulationApp tears down on
                # SystemExit and a worker blocked on this Event would then
                # wait out its full timeout with no diagnostic.
                import traceback

                request["error"] = repr(exc)
                traceback.print_exc()
            finally:
                if request.get("clear_on_done"):
                    # Video frames: the flag IS the pending marker, so
                    # clearing it is what re-arms the next frame.
                    request["done"].clear()
                else:
                    request["done"].set()
                serviced += 1
        return serviced

    def request_video_frame(self, timeout_s: float = 10.0) -> bool:
        """WORKER-THREAD side. Ask the main thread to render one video frame.

        This is the follow-up `_tick()`'s own CAVEAT asks for (handoff sec
        20b). Stage ticks run on the worker and are physics-only, so
        `record_video`'s per-tick capture had nothing rendered to save: video
        advanced during `reset()` and then FROZE for the entire episode.
        Measured on `outputs/live_er_run_1` before this fix -- frames 0-14
        differ, and every frame from 15 on is byte-identical to frame 14.

        Rendering has to happen on the main thread, and the main thread must
        not render while this thread is stepping physics, so the worker parks
        on the reply Event exactly as the ER-2 capture does. That is the
        whole reason this is a request rather than a direct call.
        """
        # BLOCKING, deliberately, and this is a safety property not a
        # performance one.
        #
        # A fire-and-forget version was tried (run 18) to stop the servo
        # stall documented below, and it SEGFAULTED at tick ~7000: without
        # the handshake the main thread runs `simulation_app.update()` while
        # the worker is inside `sim.step()`, which is the concurrent-access
        # hazard the whole park-and-service design exists to avoid. Speed is
        # not worth a core dump.
        #
        # The stall is real and is the cost of that safety. Measured, run 17
        # vs the runs around it, identical config except --record-video:
        #
        #   with video     recenter_pos_err_m 0.0318, close 'contact_lost',
        #                  gripper_position_rad 0.0 (jaws shut on air)
        #   without video  recenter_pos_err_m 0.0120, close
        #                  'contact_sustained', gripper_position_rad 0.0383
        #
        # 0.012 is inside the recenter's own 0.015 tolerance. So:
        # **--record-video measurably degrades manipulation, and a run whose
        # purpose is a SCORE should not use it.** Use it for evidence runs,
        # and read any grasp metric from a run without it.
        request: dict[str, Any] = {
            "kind": "video_frame",
            "seq": 0,
            "object_name": None,
            "done": threading.Event(),
            "payload": None,
            "error": None,
        }
        self._live_capture_requests.put(request)
        if not request["done"].wait(timeout_s):
            # Same abandonment guard as request_live_capture: without this,
            # a stale queued request can still be serviced (and render)
            # after this thread gives up and resumes stepping physics.
            request["abandoned"] = True
            return False
        return request["error"] is None

    def _capture_video_frame(self) -> bool:
        """MAIN-THREAD ONLY. Render and save one `record_video` frame."""
        if not self.record_video or self._rgb_annotator is None:
            return False
        self.simulation_app.update()
        if self._m["_save_rgb_frame"](
            self._rgb_annotator, self.frames_dir, self._frames_written
        ):
            self._frames_written += 1
            return True
        return False

    def _capture_live_frame(self, object_name: str) -> dict[str, Any]:
        """MAIN-THREAD ONLY. Render one frame and return what ER-2 needs.

        Pumps real app updates before reading: Replicator annotators populate
        on the NEXT render pass after attach, not synchronously -- the same
        empty-frame trap `_precompute_perception_grasp_targets` hit on its
        first GPU run (idToLabels == {}, data.shape == (0,)).
        """
        from PIL import Image

        self._ensure_perception_annotators()

        # RE-AIM THE HEAD CAMERA EVERY CAPTURE.
        #
        # `_ensure_perception_annotators` builds the render product once and
        # returns early forever after, so a camera placed there is frozen at
        # whatever pose the robot and object had on the first capture. The
        # base drives and the object moves, so by the second attempt the aim
        # is stale -- measured, head_cup_1: the first look scored miss_m
        # 0.0208 and the second, after the cup had shifted to z=0.809,
        # scored 0.2394 and was vetoed.
        #
        # Re-placing is cheap (it writes one transform) and it is what makes
        # this a robot-mounted camera rather than a fixed one that happens to
        # start on the robot.
        if USE_HEAD_CAMERA and self._perception_camera_prim_path:
            try:
                aim = self.object_position(object_name)
                repositioned = self._create_head_camera(aim)
                if repositioned:
                    self._perception_camera_prim_path = repositioned
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARN: head camera re-aim failed ({exc!r}); using the "
                    "previous pose",
                    flush=True,
                )

        for _ in range(5):
            self.simulation_app.update()

        rgb = self._perception_rgb_annotator.get_data()
        depth = self._perception_depth_annotator.get_data()
        cam_params = self._perception_cam_annotator.get_data()
        width_px, height_px = self._perception_resolution

        frame_dir = Path(self.out_dir) / "live_er_frames" if self.out_dir else None
        if frame_dir is None:
            frame_dir = Path("outputs") / "live_er_frames"
        frame_dir.mkdir(parents=True, exist_ok=True)
        image_path = frame_dir / (
            f"{object_name}_{self._live_capture_seq:03d}"
            f"_t{self._tick_count}.png"
        )
        Image.fromarray(rgb[:, :, :3]).save(image_path)

        return {
            "image_path": image_path,
            "depth": depth,
            "width": width_px,
            "height": height_px,
            # The camera's own view/projection matrices -- the unprojection
            # route that is measured correct on this camera. See
            # live_er_grasp.grasp_pose_from_answer for why not the authored
            # transform.
            "view_matrix": cam_params["cameraViewTransform"],
            "proj_matrix": cam_params["cameraProjection"],
        }

    def _perception_camera_pose(
        self,
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """The perception camera's world pose, read from the camera's OWN
        authored transform rather than re-derived from position/look_at --
        `sim_camera_perception._look_at_quaternion_wxyz`'s docstring explains
        why re-deriving it is the wrong move.
        """
        from omni.usd import get_context
        from pxr import UsdGeom

        stage = get_context().get_stage()
        prim = stage.GetPrimAtPath(self._perception_camera_prim_path)
        world_transform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            0
        )
        translation = world_transform.ExtractTranslation()
        rotation = world_transform.ExtractRotationQuat()
        imaginary = rotation.GetImaginary()
        return (
            (translation[0], translation[1], translation[2]),
            (
                rotation.GetReal(),
                imaginary[0],
                imaginary[1],
                imaginary[2],
            ),
        )

    def _correct_scoring_object_collision(
        self, object_paths: dict[str, str]
    ) -> None:
        """Upgrade any scoring object left on a hull-like collision
        approximation, and log what every one of them actually uses.

        **On this scene it is a NO-OP, and that is a measured result, not an
        assumption.** The live inventory (2026-08-14, run 8) is:

            plate2:Cylinder_004 = convexDecomposition
            cup:Cylinder_002    = convexDecomposition
            bowl2:Cylinder_003  = sdf
            spoon2:Tea_Spoon    = sdf

        This method was written on the hypothesis that they were all falling
        back to PhysX's default `convexHull` -- scanning `robot_room.usd`'s
        crate bytes finds no occurrence of any approximation token, which
        looked like "none authored". That hypothesis was WRONG: the
        approximations are authored, just not as literal strings in that
        file, and a convex-hulled plate is therefore NOT why the jaws push
        it. Whatever is wrong with the plate grasp, it is not this.

        Kept because it is cheap, because the inventory line is the thing
        that settled the question and will settle it again for any scene
        variant, and because a future scene really can ship a hull-like
        approximation. The organisers state the tray's collision thickness
        was adjusted, that the plate may penetrate the table, and that
        adjusting the plate's bottom collision geometry is explicitly
        allowed in official evaluation provided it only corrects collision
        behaviour -- so the upgrade path stays available.

        Only hull-like approximations are replaced. `sdf` is the most
        accurate concave collision PhysX offers and `convexDecomposition`
        already preserves concavity; rewriting either is a downgrade. An
        earlier revision of this method did exactly that to `bowl2` and
        `spoon2` before the inventory made it obvious.

        Never raises: a scene whose objects are primitives rather than
        meshes, or a USD schema without the mesh-collision API, leaves the
        objects exactly as authored. A missing collision correction costs
        grasp quality; an exception here would cost the episode.
        """
        try:
            from pxr import Usd, UsdPhysics
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARN: collision correction skipped, no pxr ({exc!r})",
                flush=True,
            )
            return

        from omni.usd import get_context

        stage = get_context().get_stage()
        corrected: list[str] = []
        # Every collision prim found and what it was, not just the ones
        # changed: an empty `corrected` list is otherwise ambiguous between
        # "already correct" and "found nothing to correct", and those call
        # for opposite next actions.
        inventory: list[str] = []
        for name, path in object_paths.items():
            root = stage.GetPrimAtPath(path)
            if not root or not root.IsValid():
                inventory.append(f"{name}:MISSING_PRIM({path})")
                continue
            found = False
            for prim in Usd.PrimRange(root):
                if not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                found = True
                try:
                    mesh_api = (
                        UsdPhysics.MeshCollisionAPI(prim)
                        if prim.HasAPI(UsdPhysics.MeshCollisionAPI)
                        else UsdPhysics.MeshCollisionAPI.Apply(prim)
                    )
                    attr = mesh_api.GetApproximationAttr()
                    if not attr:
                        attr = mesh_api.CreateApproximationAttr()
                    current = attr.Get()
                    inventory.append(f"{name}:{prim.GetName()}={current}")
                    if current not in SCORING_OBJECT_COLLISION_REPLACEABLE:
                        continue
                    attr.Set(SCORING_OBJECT_COLLISION_APPROXIMATION)
                    corrected.append(f"{name}:{prim.GetName()}({current})")
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"WARN: collision correction failed for {name!r} "
                        f"at {prim.GetPath()}: {exc!r}",
                        flush=True,
                    )
            if not found:
                inventory.append(f"{name}:NO_COLLISION_PRIM")
        print(
            "SCORING_OBJECT_COLLISION "
            f"approximation={SCORING_OBJECT_COLLISION_APPROXIMATION!r} "
            f"corrected={corrected} inventory={inventory}",
            flush=True,
        )

    def _object_has_fallen(self, object_name: str) -> bool:
        """Has this object left the surface it spawned on?

        Parameter-free by construction: an object counts as fallen when it
        is nearer the floor than the height it spawned at. No threshold to
        tune, and it holds for a counter, a table or a tray equally.

        This matters because a fallen object costs the WHOLE stage. Measured
        repeatedly this session: once plate2 (run 5) or the cup (run 19) is
        knocked to the floor, every remaining attempt still targets it --
        `ik_fail_ticks 800/800`, `recenter_pos_err_m 0.9667`, minutes per
        retry -- and the objects that are still sitting on the counter never
        get an attempt at all. Skipping a fallen object is strictly better
        than spending its retries proving the floor is out of reach.
        """
        # getattr, not attribute access: a world that has not reset yet
        # (test doubles, pre-episode) has no spawn record, and "no record"
        # must mean "not fallen" rather than an AttributeError from a
        # guard whose whole contract is to never abort anything.
        spawn_heights = getattr(self, "_spawn_object_z", None)
        if not spawn_heights:
            return False
        spawn_z = spawn_heights.get(object_name)
        if spawn_z is None:
            return False
        try:
            current_z = float(self.object_position(object_name)[2])
        except Exception:  # noqa: BLE001
            return False
        floor_z = 0.0
        return abs(current_z - floor_z) < abs(current_z - spawn_z)

    def _curobo_plan_and_execute_grasp(
        self, side: str, grasp_xyz, grasp_quat_wxyz, half_open_m=None
    ) -> bool:
        """Plan the approach+grasp with cuMotion and follow it. True if flown.

        Returns False for every failure so `reach()` continues into its
        existing servo -- the never-abort contract this file already uses for
        the live ER-2 pose and the perception cache.

        Why this exists: the servo is the measured blocker. ER-2 puts the
        grasp point 0.048 m from the object and the pads finish 0.132-0.165 m
        away, because `recenter` ends 0.05-0.07 m from its own target with IK
        solving 800/800 ticks. A planned trajectory ends AT the solution.
        """
        if not self.curobo_grasp:
            return False
        try:
            import torch

            from task3_autonomy.curobo_grasp_planner import (
                CuroboGraspPlanner,
                world_pose_to_frame,
            )

            planner = self._curobo_planners.get(side)
            if planner is None:
                yml = (
                    REPO_ROOT
                    / "scripts"
                    / "task3"
                    / "curobo"
                    / f"fr3_duo_{side}_arm.yml"
                )
                planner = CuroboGraspPlanner(str(yml), side=side)
                self._curobo_planners[side] = planner

            # Goals are expressed in the robot YAML's base_link -- `left_base`
            # / `right_base` -- NOT in world.
            body_names = list(self.robot.body_names)
            frame_link = f"{side}_base"
            if frame_link not in body_names:
                return False
            idx = body_names.index(frame_link)
            frame_pos = self.robot.data.body_pos_w[0, idx]
            frame_quat = self.robot.data.body_quat_w[0, idx]

            device = frame_pos.device
            gp = torch.tensor(
                [float(v) for v in grasp_xyz],
                device=device,
                dtype=frame_pos.dtype,
            )
            gq = torch.tensor(
                [float(v) for v in grasp_quat_wxyz],
                device=device,
                dtype=frame_pos.dtype,
            )
            local_pos, local_quat = world_pose_to_frame(
                gp, gq, frame_pos, frame_quat
            )

            current = torch.tensor(
                self.arms.measured_arm_joints(side),
                device=device,
                dtype=frame_pos.dtype,
            )
            # Log the goal AS THE PLANNER SEES IT. "Goalset planning
            # returned None" is the same status for an impossible pad
            # straddle and for an unreachable pose, so without this there is
            # no way to tell which is happening -- and guessing between them
            # has already cost two runs.
            self._log_phase(
                "curobo_grasp_goal",
                True,
                side=side,
                world_xyz=[round(float(v), 4) for v in grasp_xyz],
                frame_link=frame_link,
                local_xyz=[round(float(v), 4) for v in local_pos],
                local_quat=[round(float(v), 4) for v in local_quat],
                local_norm_m=round(float(local_pos.norm()), 4),
                current_joints=[round(v, 3) for v in
                                self.arms.measured_arm_joints(side)],
            )
            # `half_open_m` is the PAD SEPARATION the planner solves for --
            # it places the two tool frames at grasp_position +/- this along
            # the measured opening axis. The default (0.0425, i.e. 85 mm
            # apart) is a BODY grasp, and around an 80.1 mm cup that leaves
            # 2.5 mm a side. A rim straddle wants ~0.015 (30 mm), enough for
            # one pad inside the cup and one outside its 1.89 mm wall.
            #
            # Exposed because "Goalset planning returned None" is the same
            # status for an impossible straddle as for an unreachable pose
            # (see the goal log above), so being able to vary this is how the
            # two are told apart.
            result = (
                planner.plan(local_pos, local_quat, current)
                if half_open_m is None
                else planner.plan(
                    local_pos, local_quat, current,
                    half_open_m=float(half_open_m),
                )
            )
            if result is None:
                return False

            flown = []
            for leg in ("approach", "grasp"):
                if not planner.leg_succeeded(result, leg):
                    self._log_phase(
                        "curobo_grasp_plan",
                        False,
                        side=side,
                        failed_leg=leg,
                        status=str(getattr(result, "status", "")),
                    )
                    return False
                waypoints = planner.waypoints(result, leg)
                if waypoints is None:
                    return False
                flown.append((leg, waypoints))

            for leg, waypoints in flown:
                # The interpolated trajectories are ~1000 waypoints each;
                # flying every one at one tick apiece is both slower than the
                # planner intended and pointlessly fine at dt=0.005. Stride
                # to roughly the planner's own dt.
                stride = max(1, waypoints.shape[0] // 200)
                self.arms.follow_arm_joint_trajectory(
                    side,
                    waypoints[::stride],
                    step=self._tick,
                    settle_ticks=40 if leg == "grasp" else 0,
                )
                self._log_phase(
                    "curobo_grasp_leg",
                    True,
                    side=side,
                    leg=leg,
                    waypoints=int(waypoints.shape[0]),
                    stride=stride,
                )
            return True
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(
                f"WARN: cuMotion grasp plan raised ({exc!r}); "
                "falling back to the servo path",
                flush=True,
            )
            traceback.print_exc()
            return False

    def _write_crop(
        self, image_path: Any, box: tuple[int, int, int, int]
    ) -> Any | None:
        """Write ``box`` of ``image_path`` beside it and return the path.

        Returns None rather than raising: the crop refinement is an
        accuracy improvement, and losing it must cost precision, never the
        grasp -- the same never-abort contract the rest of this seam keeps.
        """
        try:
            from pathlib import Path

            from PIL import Image

            src = Path(image_path)
            with Image.open(src) as img:
                crop = img.crop(box)
                out = src.with_name(f"{src.stem}_crop{src.suffix}")
                crop.save(out)
            return out
        except Exception as exc:  # noqa: BLE001
            print(f"ER_GRASP_CROP_WRITE_FAILED {exc!r}", flush=True)
            return None

    def _answer_miss_m(
        self, answer: Any, capture: dict, object_name: str
    ) -> float | None:
        """How far this ER answer's world point lands from the object.

        Used ONLY to choose between two candidate answers (wide shot vs
        crop), never to produce a pose -- the same ground-truth-as-veto
        contract `_live_er_grasp_pose`'s miss gate follows. Returns None if
        the answer cannot be projected at all, which the caller treats as
        "cannot compare, so keep what we had".
        """
        try:
            from task3_autonomy import live_er_grasp as ler

            u, v = answer.grasp_uv
            col = min(max(int(round(u)), 0), capture["width"] - 1)
            row = min(max(int(round(v)), 0), capture["height"] - 1)
            depth_m = float(capture["depth"][row, col])
            if not math.isfinite(depth_m) or depth_m <= 0.0:
                return None
            pose = ler.grasp_pose_from_answer(
                answer,
                depth_m,
                capture["view_matrix"],
                capture["proj_matrix"],
                capture["width"],
                capture["height"],
            )
            return math.dist(pose.xyz, self.object_position(object_name))
        except Exception:  # noqa: BLE001
            return None

    def _live_er_grasp_pose(
        self, object_name: str, side: str = "left"
    ) -> Any | None:
        """WORKER-THREAD side. One live ER-2 grasp pose for THIS attempt.

        Returns a `live_er_grasp.LiveGraspPose` (world position + a full
        approach quaternion) or ``None`` for any failure at all -- no
        capture, a malformed model answer, an invalid depth reading, no API
        key, no network. ``None`` means "use the existing path", never "abort
        the grasp": the same never-abort contract `_perception_grasp_target`
        established, for the same reason -- a perception outage must cost one
        object's grasp quality, not the episode.

        The capture is main-thread (see `request_live_capture`); the HTTP
        call and all the arithmetic happen right here on the worker, which is
        safe because neither touches Kit.
        """
        if not self.live_er_grasp:
            return None
        if self._object_has_fallen(object_name):
            self._log_phase(
                "live_er_grasp",
                False,
                object=object_name,
                reason="object_has_fallen_off_its_surface",
                spawn_z=round(
                    getattr(self, "_spawn_object_z", {}).get(object_name, 0.0), 4
                ),
                current_z=round(float(self.object_position(object_name)[2]), 4),
            )
            return None
        try:
            from task3_autonomy import live_er_grasp as ler

            capture = self.request_live_capture(object_name)
            if capture is None:
                return None

            answer = ler.request_grasp_answer(
                capture["image_path"],
                object_name,
                capture["width"],
                capture["height"],
            )

            # STAGE TWO: RE-ASK ABOUT A CROP OF THE SAME FRAME.
            #
            # GOTCHAS: "Wide-shot pointing lands ~5.7 cm off. Wide shot for
            # semantics, cropped or wrist-camera view for geometry.
            # Two-stage." Quoted in three comments in this repo and
            # implemented in none of them until now.
            #
            # Cropping rather than re-capturing: `grasp()`'s "second look at
            # contact range" takes a fresh frame and at contact the head
            # camera is looking at the robot's own arm -- ER-2 answered
            # "a close-up of a robotic arm joint or tool changer interface"
            # and the miss check rejected it at 0.3555 m.
            #
            # ACCEPTED ONLY IF IT IS NOT WORSE. Measured, allobj_8, and this
            # is why the gate below exists rather than a bare try/except:
            # one crop landed off the object and ER-2 answered "The image is
            # a solid pink color with no spoon visible" -- then that answer
            # REPLACED a good wide-shot pose (shift_from_first_m 0.114), the
            # close failed, the spoon was flung 1.22 m from the gripper and
            # ended at z -8590. A confidently wrong answer is the normal
            # failure of a crop that misses, so guarding only against
            # exceptions is not additive at all; the miss gate did not catch
            # it either, because 0.0838 sits inside the 0.114 limit.
            #
            # The veto uses ground truth ONLY to compare two candidate
            # answers, never to supply a pose -- the same contract the miss
            # gate below already follows.
            refine_note: dict[str, Any] = {}
            if self.er_grasp_crop_refine:
                try:
                    box = ler.crop_box_around(
                        answer.grasp_uv[0],
                        answer.grasp_uv[1],
                        capture["width"],
                        capture["height"],
                        ER_GRASP_CROP_HALF_PX,
                    )
                    crop_path = self._write_crop(capture["image_path"], box)
                    if crop_path is not None:
                        refined = ler.answer_in_full_frame(
                            ler.request_grasp_answer(
                                crop_path,
                                object_name,
                                box[2] - box[0],
                                box[3] - box[1],
                            ),
                            box,
                        )
                        wide_miss = self._answer_miss_m(
                            answer, capture, object_name
                        )
                        refined_miss = self._answer_miss_m(
                            refined, capture, object_name
                        )
                        accept = (
                            wide_miss is not None
                            and refined_miss is not None
                            and refined_miss <= wide_miss
                        )
                        refine_note = {
                            "crop_box": list(box),
                            "wide_miss_m": (
                                round(wide_miss, 4)
                                if wide_miss is not None
                                else None
                            ),
                            "refined_miss_m": (
                                round(refined_miss, 4)
                                if refined_miss is not None
                                else None
                            ),
                            "accepted": bool(accept),
                        }
                        if accept:
                            answer = refined
                        self._log_phase(
                            "live_er_grasp_crop_refine",
                            bool(accept),
                            object=object_name,
                            **refine_note,
                        )
                except Exception as exc:  # noqa: BLE001
                    self._log_phase(
                        "live_er_grasp_crop_refine",
                        False,
                        object=object_name,
                        error=repr(exc),
                    )

            # Depth AT the grasp pixel. A zero/inf reading means ER-2 pointed
            # at empty space or through the object -- the back-projection
            # would produce a confident position on the far wall, so refuse
            # it rather than grasp there.
            depth = capture["depth"]
            u, v = answer.grasp_uv
            col = min(max(int(round(u)), 0), capture["width"] - 1)
            row = min(max(int(round(v)), 0), capture["height"] - 1)
            depth_m = float(depth[row, col])
            if not math.isfinite(depth_m) or depth_m <= 0.0:
                self._log_phase(
                    "live_er_grasp",
                    False,
                    object=object_name,
                    reason="invalid_depth_at_grasp_pixel",
                    depth_m=depth_m,
                    pixel=[col, row],
                )
                return None

            pose = ler.grasp_pose_from_answer(
                answer,
                depth_m,
                capture["view_matrix"],
                capture["proj_matrix"],
                capture["width"],
                capture["height"],
            )

            # Sanity-gate against the SCENE, not against a remembered number:
            # a grasp point that is not near the object we are reaching for
            # is a mis-point, and acting on it throws the object across the
            # room (a failure this repo has seen repeatedly). Ground truth is
            # used ONLY as a veto here -- it never supplies the pose.
            live_obj = self.object_position(object_name)
            miss_m = math.dist(pose.xyz, live_obj)
            if miss_m > LIVE_ER_GRASP_MAX_MISS_M:
                self._log_phase(
                    "live_er_grasp",
                    False,
                    object=object_name,
                    reason="grasp_point_too_far_from_object",
                    miss_m=round(miss_m, 4),
                    limit_m=LIVE_ER_GRASP_MAX_MISS_M,
                    **pose.as_log(),
                )
                return None

            # A parallel jaw is symmetric under a half turn about its
            # approach axis, so `roll` and `roll + 180` are the SAME grasp.
            # Command whichever one the wrist can actually get to. Joint 7 is
            # capped at effort/damping = 12/500 = 0.024 rad/s and `arms.reach`
            # gets 4 s, so it can travel ~0.1 rad before time runs out --
            # measured on run 11, a 0.42 rad commanded wrist move left every
            # reach 2-3 cm short with IK solving on 800/800 ticks. Halving
            # that travel is free accuracy and, unlike a gains change, cannot
            # destabilise anything.
            #
            # Compared by the IK SOLUTION each roll produces, not by how far
            # apart the two quaternions are. Run 14 established that the
            # quaternion comparison optimises the wrong thing: it changed
            # `recenter_pos_err_m` by 0.001 because the requested roll was
            # never what set joint 7's travel -- Lula picks a solution out of
            # this redundant arm's null space, and that choice is what moves
            # the wrist.
            # DISABLED on measurement -- see GRASP_ROLL_SYMMETRY_DISABLED
            # below. Kept because the machinery (preview_arm_joints) is
            # what produced the number that settled the question.
            try:
                chosen_roll = pose.roll_deg
                measured = (
                    self.arms.measured_arm_joints(side)
                    if GRASP_ROLL_SYMMETRY_ENABLED
                    else None
                )
                if measured is None:
                    raise _RollSelectionDisabled
                best_travel = None
                for candidate_roll in (pose.roll_deg, pose.roll_deg + 180.0):
                    candidate_quat = ler.quaternion_from_approach(
                        pose.tilt_deg, pose.azimuth_deg, candidate_roll
                    )
                    solution = self.arms.preview_arm_joints(
                        side, pose.xyz, candidate_quat
                    )
                    if solution is None:
                        continue
                    # Joint 7 is the one that cannot get there in time: it is
                    # capped at effort/damping = 12/500 = 0.024 rad/s and
                    # `arms.reach` gets 4 s, so ~0.1 rad is its whole budget.
                    travel = abs(solution[-1] - measured[-1])
                    if best_travel is None or travel < best_travel:
                        best_travel, chosen_roll = travel, candidate_roll
                chosen_roll = (chosen_roll + 180.0) % 360.0 - 180.0
                if best_travel is not None:
                    print(
                        "GRASP_ROLL_CHOICE "
                        f"object={object_name!r} side={side!r} "
                        f"requested_roll={pose.roll_deg:.1f} "
                        f"chosen_roll={chosen_roll:.1f} "
                        f"joint7_travel_rad={best_travel:.4f}",
                        flush=True,
                    )
                if chosen_roll != pose.roll_deg:
                    pose = ler.LiveGraspPose(
                        xyz=pose.xyz,
                        quaternion_wxyz=ler.quaternion_from_approach(
                            pose.tilt_deg, pose.azimuth_deg, chosen_roll
                        ),
                        tilt_deg=pose.tilt_deg,
                        azimuth_deg=pose.azimuth_deg,
                        roll_deg=chosen_roll,
                        reason=pose.reason,
                    )
            except _RollSelectionDisabled:
                pass
            except Exception as exc:  # noqa: BLE001
                # Purely an optimisation -- the un-flipped roll is still a
                # correct grasp, so a failure here must not lose the pose.
                print(
                    f"WARN: roll symmetry selection skipped ({exc!r})",
                    flush=True,
                )

            self._log_phase(
                "live_er_grasp",
                True,
                object=object_name,
                miss_m=round(miss_m, 4),
                depth_m=round(depth_m, 4),
                **pose.as_log(),
            )
            return pose
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(
                f"WARN: live ER-2 grasp pose raised ({exc!r}) for "
                f"{object_name!r}, falling back to the existing grasp path",
                flush=True,
            )
            traceback.print_exc()
            return None

    def _precompute_perception_grasp_targets(self) -> None:
        """REV16 Phase C follow-up (owner correction, 2026-08-09): compute
        every object's perception grasp candidate ONCE, here, at the end of
        reset() on the MAIN thread (see the call site's own comment for
        why annotators may only be touched from that thread). Populates
        `self._perception_grasp_cache`; never raises -- any failure leaves
        the cache empty/partial, and `_perception_grasp_target`'s
        cache-miss path already falls back to the constant path for
        whatever wasn't computed, the same never-abort contract the old
        per-call version had. Deliberately does NOT run IK screening here
        -- that depends on the robot's CURRENT base pose, which is only
        correct at the moment each object is actually approached (done
        fresh, per call, in `_perception_grasp_target` below, safe on the
        worker thread since it is pure IK math with no annotator calls).
        Screening against reset()'s pose would silently answer the wrong
        question for every object this episode hasn't navigated to yet.
        """
        if not self.perception_grasp:
            return
        try:
            from task3_autonomy import perception_grasp as pg

            self._ensure_object_semantics()
            self._ensure_perception_annotators()
            # 2026-08-09 (O3 follow-up): the first GPU test of the
            # reset()-time capture (main-thread precompute, no hang) found
            # idToLabels == {} and data.shape == (0,) -- the annotator
            # frame itself was empty, not a prim-path mismatch. Replicator
            # annotators populate on the NEXT render pass after attach, not
            # synchronously at attach time; nothing pumped a render between
            # `_ensure_perception_annotators()` (which only calls
            # `.attach()`) and the first `get_data()` call below. Pump a
            # few real frames first -- `--record-video`'s camera path gets
            # this for free from the episode's own ongoing `sim.step()`
            # calls elsewhere, but this is the FIRST read this episode, at
            # the end of reset(), so nothing has rendered through this
            # specific render product yet.
            for _ in range(5):
                self.simulation_app.update()
            seg_data = self._perception_seg_annotator.get_data()
            depth = self._perception_depth_annotator.get_data()
            cam_params = self._perception_cam_annotator.get_data()
            width_px, height_px = self._perception_resolution

            for object_name, view in self.object_views.items():
                prim_paths = getattr(view, "prim_paths", None)
                candidate = None
                if prim_paths:
                    try:
                        mask = pg.segment(seg_data, prim_paths[0])
                        candidate = pg.grasp_point_from_mask(
                            object_name,
                            mask,
                            depth,
                            cam_params["cameraViewTransform"],
                            cam_params["cameraProjection"],
                            width_px,
                            height_px,
                        )
                    except Exception as exc:  # noqa: BLE001
                        print(
                            "WARN: perception grasp precompute raised "
                            f"({exc!r}) for {object_name!r}, cached as "
                            "None",
                            flush=True,
                        )
                self._perception_grasp_cache[object_name] = candidate
                if candidate is not None:
                    # REV16 Phase C.4 gate: log localization error against
                    # the scene's own ground truth before trusting any k/N
                    # on top of this. candidate.xyz is a grasp POINT on the
                    # object surface, not its center -- some offset from
                    # object_position() is expected, not itself a bug; the
                    # brief's <2cm target is judged against this number,
                    # not a perfect zero.
                    gt_xyz = self.object_position(object_name)
                    error_m = (
                        sum(
                            (a - b) ** 2 for a, b in zip(candidate.xyz, gt_xyz)
                        )
                        ** 0.5
                    )
                    self._log_phase(
                        "perception_grasp_precompute",
                        True,
                        object=object_name,
                        perception_xyz=[round(v, 4) for v in candidate.xyz],
                        ground_truth_xyz=[round(v, 4) for v in gt_xyz],
                        error_m=round(error_m, 4),
                        mask_pixel_count=candidate.mask_pixel_count,
                        width_m=candidate.width_m,
                        width_ok=candidate.width_ok,
                    )
                else:
                    self._log_phase(
                        "perception_grasp_precompute",
                        False,
                        object=object_name,
                        reason="no_candidate",
                    )
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(
                f"WARN: perception grasp precompute setup raised "
                f"({exc!r}), cache stays empty -- every object falls back "
                "to the constant path this episode",
                flush=True,
            )
            traceback.print_exc()

    def _perception_grasp_target(
        self, object_name: str
    ) -> tuple[tuple[float, float, float], float] | None:
        """Look up `object_name`'s perception-derived grasp candidate
        (REV16 Phase C, computed once at reset() time by
        `_precompute_perception_grasp_targets` -- see that method for why
        this can no longer touch the annotators itself) and IK-screen it
        against the robot's CURRENT base pose. Returns `(xyz, yaw_rad)` on
        a real, IK-feasible candidate, or `None` for ANY other outcome (no
        cache entry, no IK-feasible side, or any exception) -- `reach()`'s
        caller must treat `None` as "use the existing constant path",
        never as a reason to abort the grasp attempt.
        """
        if not self.perception_grasp:
            return None
        candidate = self._perception_grasp_cache.get(object_name)
        if candidate is None:
            self._log_phase(
                "perception_grasp_target",
                False,
                object=object_name,
                reason="no_cached_candidate",
            )
            return None
        try:
            from task3_autonomy import perception_grasp as pg

            pg.screen_grasp_candidate_ik(self, candidate)
            if not candidate.any_side_feasible:
                self._log_phase(
                    "perception_grasp_target",
                    False,
                    object=object_name,
                    reason="ik_infeasible",
                    xyz=[round(v, 4) for v in candidate.xyz],
                    width_m=candidate.width_m,
                    width_ok=candidate.width_ok,
                )
                return None
            self._log_phase(
                "perception_grasp_target",
                True,
                object=object_name,
                xyz=[round(v, 4) for v in candidate.xyz],
                yaw_rad=candidate.yaw_rad,
                width_m=candidate.width_m,
                width_ok=candidate.width_ok,
                mask_pixel_count=candidate.mask_pixel_count,
            )
            return candidate.xyz, candidate.yaw_rad
        except Exception as exc:  # noqa: BLE001
            import traceback

            print(
                f"WARN: perception grasp IK screen raised ({exc!r}) for "
                f"{object_name!r}, falling back to the constant grasp "
                "path",
                flush=True,
            )
            traceback.print_exc()
            return None

    def reach(self, side, object_name, **p) -> dict:
        self._active_object = object_name

        # Refuse an object that is on the floor, HERE, before any driving or
        # servoing happens.
        #
        # The same check in `_live_er_grasp_pose` was not enough and the
        # difference is worth stating: returning None from there means "no
        # live pose, use the fallback path", not "skip this object". Run 20
        # shows the consequence exactly -- the guard correctly logged
        # `object_has_fallen_off_its_surface` for the cup at
        # `current_z 0.0353`, and the pipeline then reached and closed on it
        # anyway through the ground-truth fallback, ending with the jaws
        # 0.918 m from an object lying on the floor.
        #
        # A fallen object cannot be picked and every attempt on it is a
        # stage-budget-sized hole (run 19 lost most of an hour that way), so
        # this returns the existing unreachable result rather than inventing
        # a new failure mode -- the caller already treats that as "this one
        # skill attempt failed", not "abort the stage".
        if self._object_has_fallen(object_name):
            self._log_phase(
                "reach_skipped_object_has_fallen",
                False,
                object=object_name,
                spawn_z=round(
                    getattr(self, "_spawn_object_z", {}).get(object_name, 0.0), 4
                ),
                current_z=round(
                    float(self.object_position(object_name)[2]), 4
                ),
            )
            return {
                "position_error_m": 999.0,
                "strict_reach": False,
                "ee_dy_m": 0.0,
                "reason": "object_has_fallen_off_its_surface",
            }

        approach = p.get("approach_stance", "east")
        m = self._m
        vgl = m["vgl"]

        live_obj = self.object_position(object_name)
        # R9 T4: an explicit stance override (from a ranked grasp
        # candidate, see reach_and_grasp_ranked below) replaces the
        # computed stance wholesale -- same additive-only contract as
        # grasp_xyz_override/grasp_yaw_override (vm-b L2, 2026-08-02).
        # Omitted, this is byte-identical to the prior behavior.
        # 2026-08-14: the override is only honoured if the base can
        # physically stand there. `assets/derived/grasp_ranked/*.json` was
        # generated 2026-08-11, three days BEFORE the island/west-wall
        # stance fix, and it froze the pre-fix stances into a file --
        # 244 of its 248 ranked candidates put the base centre inside
        # KITCHEN_ISLAND_BBOX (plate2 rank0 = (-3.829, -1.643), which is
        # 0.06m inside the counter's east edge). Because the override
        # replaced the computed stance *wholesale*, every ranked grasp
        # attempt drove at a solid counter, arrived ~0.8m short and
        # rotated wrong, and then logged the familiar `ik_ok_ticks: 0/1200`
        # -- the arm solving against an object that was behind it. The
        # stance fix was real but this path never saw it.
        # Falling back to the live computed stance is strictly better than
        # a stance that is known-unreachable, and it keeps the override
        # contract for every candidate that IS reachable (bowl2 rank0).
        stance_xy = None
        if "stance_xy_override" in p:
            candidate_xy = tuple(p["stance_xy_override"])
            if point_clears_island(candidate_xy):
                stance_xy = candidate_xy
                stance_yaw = p.get("stance_yaw_override", 0.0)
            else:
                print(
                    "STANCE_OVERRIDE_REJECTED "
                    f"object={object_name!r} "
                    f"override_xy=({candidate_xy[0]:.3f},"
                    f"{candidate_xy[1]:.3f}) "
                    "reason=inside_kitchen_island_footprint",
                    flush=True,
                )
        if stance_xy is None:
            stance_xy, stance_yaw = self._stance_for(
                (live_obj[0], live_obj[1]), approach
            )

        # Drive to the reach-safe stance, then square up.
        # min_creep_mps (navigation.py base_twist_toward, L3 2026-08-02):
        # position_kp * distance decays to zero as distance shrinks, but the
        # wheel drives have a ~2s velocity-tracking lag (DRIVE_DAMPING=500)
        # -- a commanded speed that keeps shrinking faster than the wheels
        # can track it never lets real velocity catch up, which was flagged
        # as a plausible cause of navigate_rotate_spot's ~0.087m stall but
        # never GPU-confirmed nor wired into this call. The grip-proof
        # diagnostic (2026-08-11) reproduced the identical signature here:
        # navigate_to stalled=True at terminal_error_m 0.104-0.564 across
        # five attempts, every one short of the ~0.03m tolerance. Floor
        # commanded speed at 0.08 m/s so the wheels have a sustained,
        # non-decaying target to track through the final approach; caller
        # stops the instant pose_reached() is true, so worst-case overshoot
        # is one tick's travel (<=0.4mm at sim.cfg.dt=0.005s).
        # 2026-08-14: retry in place. reach() used to re-drive the base to
        # the stance on EVERY ranked candidate, even when it was already
        # parked there -- measured on outputs/e2e_calibrated.log, close #1
        # failed at tick 11490 and close #2 did not happen until 23363,
        # ~10 min later, almost all of it re-navigating to a stance the
        # base had not left. That is why a 33-minute run managed three
        # grasp attempts instead of ~20.
        #
        # The tolerance is derived, not chosen: `pose_reached`'s own
        # arrival tolerance is what navigate_to would itself have accepted,
        # so if the base already satisfies it, the drive is by definition a
        # no-op. Anything further away still drives exactly as before.
        settled_pose = self.adapter.pose()
        stance_gap_m = math.dist(
            (settled_pose.x, settled_pose.y), tuple(stance_xy)
        )
        already_parked = stance_gap_m <= NAVIGATE_ARRIVAL_TOLERANCE_M
        if already_parked:
            print(
                "NAVIGATE_SKIPPED_ALREADY_PARKED "
                f"object={object_name!r} gap_m={stance_gap_m:.4f} "
                f"tolerance_m={NAVIGATE_ARRIVAL_TOLERANCE_M:.4f}",
                flush=True,
            )
        else:
            # 2026-08-16: verify arrival, retry if not. GPU-measured
            # (outputs/keep_capture_robot_cameras_v2.log, tick 1750-6898):
            # a routed stance (via-door / island-detour waypoints can put
            # the straight-line-equivalent distance well past what
            # budget_s=25.0 covers at max_linear_mps=0.25, e.g. ~4.9m here)
            # can run this call's tick budget out with `stalled: False` --
            # the base was still making real progress, just hadn't
            # arrived -- landing 1.03m short. The return value used to be
            # discarded entirely, so `reach()` proceeded straight into
            # `open_before_approach`'s arm-IK targeting with the base
            # nowhere near the object. That target is computed from the
            # object's real position, so it was genuinely unreachable from
            # the base's actual (short) position -- Lula correctly kept
            # reporting "no solution" every ~5s, 47 times, because nothing
            # ever re-drove the base afterward. This is why the base
            # appeared to "stall" during grasp targeting even though the
            # earlier isolated `navigate_to()`-alone stall (a DIFFERENT,
            # ProgressWatchdog-detected wheel-lag stall) never showed an
            # IK failure at all -- two distinct failure modes, same
            # frozen-base symptom. A shortfall from a budget timeout
            # (not a genuine stall) is bounded and shrinks each retry, so
            # a small bounded retry (not an unbounded loop) reliably
            # closes it: the leftover 1.03m at 0.25 m/s needs ~4s, well
            # inside a fresh 25s budget.
            for attempt in range(3):
                nav_result = self.navigate_to(
                    *stance_xy,
                    max_linear_mps=0.25,
                    budget_s=25.0,
                    min_creep_mps=0.08,
                )
                gap_m = math.dist(
                    (self.adapter.pose().x, self.adapter.pose().y),
                    tuple(stance_xy),
                )
                if gap_m <= NAVIGATE_ARRIVAL_TOLERANCE_M:
                    break
                if nav_result.get("stalled"):
                    # A genuine ProgressWatchdog stall (base stopped
                    # moving, not just out of budget) won't be fixed by
                    # repeating the identical call -- stop retrying and
                    # let the caller's own unreachable-target handling
                    # take over, same as before this fix.
                    break
                print(
                    "NAVIGATE_STANCE_RETRY "
                    f"object={object_name!r} attempt={attempt + 1} "
                    f"gap_m={gap_m:.4f} "
                    f"tolerance_m={NAVIGATE_ARRIVAL_TOLERANCE_M:.4f}",
                    flush=True,
                )
        self._rotate_to(stance_yaw)
        settled = self.adapter.pose()
        self._base_hold_anchor = (settled.x, settled.y)

        # vm-b L2 (2026-08-02, owner GPU-first directive): an explicit
        # grasp_yaw_override applies to ANY object, not just cup -- lets a
        # caller (e.g. er_grasp_pose.py) substitute an ER-2-derived wrist
        # orientation for a real live reach()/grasp attempt. Additive only:
        # omitted, this is byte-identical to the prior cup-only behavior.
        if "grasp_yaw_override" in p:
            grasp_yaw_rad = p["grasp_yaw_override"]
        else:
            grasp_yaw_rad = (
                p.get("grasp_yaw_rad", 0.0) if object_name == "cup" else 0.0
            )
        top_down = m["_quaternion_from_rpy"](TOP_DOWN_ROLL_RAD, 0.0, grasp_yaw_rad)
        # Remember it for grasp()'s pre-close re-center (see _last_grasp_yaw).
        self._last_grasp_yaw[object_name] = grasp_yaw_rad
        # The approach the standoff/pregrasp legs back off along. Straight
        # down until a live orientation says otherwise, which reproduces the
        # old world-+Z arithmetic exactly (er_grasp_orientation.
        # offset_along_approach's own test pins this).
        approach_tilt_deg = 0.0
        approach_azimuth_deg = 0.0

        # STEP 2C (docs/TASK3_MASTER_EXECUTION_PLAN_2026-07-24.md sec 4 /
        # plans/handoff.md sec 5): compute the grasp target from the LIVE
        # object pose ONCE, before any arm motion, and reuse its XY for the
        # pregrasp point too. Previously pregrasp_xy was a hardcoded
        # constant (vgl.CUP_GRASP_XY) offset ~5-6cm in -Y from the real
        # grasp target, so the descend leg was a lateral slide that dragged
        # the fingers across the cup and pushed it out of the jaws --
        # confirmed this session (outputs/t3_diag_instrumented/result.json):
        # ik_fail_ticks was 0 throughout (IK never failed), so the miss was
        # not a solver/reach failure -- it was this built-in slant. Using
        # the same XY for both legs makes the final approach a pure
        # vertical drop.
        live_obj = self.object_position(object_name)  # re-read post-settle
        # REV16 Phase C: try the perception-derived grasp point first
        # (default off; see __init__'s perception_grasp flag and
        # _perception_grasp_target's own docstring). Any non-success
        # returns None, which falls straight into the unchanged
        # cup/object constant path below -- byte-identical behavior when
        # the flag is off or the attempt fails.
        # 2026-08-14 (owner directive): a LIVE ER-2 call, this attempt, for
        # this object, returning position AND orientation. It is tried first
        # because it is the only source here that is neither a constant nor a
        # file written days ago -- the perception-grasp cache below is
        # computed at reset() before the robot has navigated, and everything
        # after it is a fixed offset. Any failure returns None and falls
        # straight through to the unchanged paths.
        live_pose = self._live_er_grasp_pose(object_name, side)
        perception_grasp_result = (
            None if live_pose is not None
            else self._perception_grasp_target(object_name)
        )
        if live_pose is not None:
            grasp_target = live_pose.xyz
            top_down = live_pose.quaternion_wxyz
            approach_tilt_deg = live_pose.tilt_deg
            approach_azimuth_deg = live_pose.azimuth_deg
            # grasp()'s pre-close re-center reproduces the wrist orientation
            # from this; handing it the live roll keeps the re-center from
            # snapping the wrist back and undoing the chosen approach.
            self._last_grasp_yaw[object_name] = math.radians(
                live_pose.roll_deg
            )
        elif perception_grasp_result is not None:
            grasp_target, perception_yaw_rad = perception_grasp_result
            top_down = m["_quaternion_from_rpy"](
                TOP_DOWN_ROLL_RAD, 0.0, perception_yaw_rad
            )
        elif object_name == "cup":
            grasp_xy_offset = p.get("grasp_y_offset", vgl.CUP_GRASP_Y_OFFSET)
            grasp_target = vgl.cup_grasp_target(
                live_obj,
                rim_x_offset=p.get("cup_rim_x_offset", vgl.CUP_RIM_X_OFFSET),
                grasp_y_offset=grasp_xy_offset,
                grasp_z_offset=0.0,
            )
        else:
            grasp_target = vgl.object_grasp_target(
                live_obj,
                x_offset=p.get("object_grasp_x_offset", 0.0),
                y_offset=p.get("object_grasp_y_offset", 0.0),
                z_offset=p.get(
                    "grasp_height_above_origin_m",
                    p.get("object_grasp_z_offset", 0.075),
                ),
            )
        # vm-b L2: an explicit grasp_xyz_override replaces the computed
        # target wholesale (any object, including cup) -- same additive-
        # only contract as grasp_yaw_override above.
        if "grasp_xyz_override" in p:
            grasp_target = tuple(p["grasp_xyz_override"])
        # 2026-08-14: everything above computes where the FINGERS should be
        # (an ER grasp point, a rim offset, a height above the object's
        # origin). Everything below commands the IK frame, `hand_tcp`. Those
        # are not the same point: the pad midpoint measures [0, 0, +0.0186] m
        # in the tcp frame (rigid, 0.0 stdev over 4 arm poses --
        # scripts/task3/measure_tool_offset_isaac.py). Converting here, once,
        # is what makes every downstream command -- pregrasp, standoff,
        # gentle ramp, final descend -- land the PADS on the grasp point.
        # `grasp_target` itself stays in pad space so the object-relative
        # bookkeeping (_last_grasp_offset, the honest-hold check) keeps
        # measuring what it always measured.
        # Record the orientation actually commanded, now that every branch
        # above has had its say, so grasp()'s pre-close re-center reproduces
        # THIS wrist pose instead of rebuilding a straight-down one.
        self._last_grasp_quat[object_name] = tuple(top_down)

        # cuMotion first: plan the approach and the grasp as joint
        # trajectories and fly them, instead of servoing at `top_down` leg by
        # leg below. If it plans and flies, everything from `pregrasp` to the
        # final converge is already done and this returns straight into the
        # reach gate. If anything at all goes wrong it returns False and the
        # unchanged servo path runs, so this is additive.
        if self._curobo_plan_and_execute_grasp(
            side, grasp_target, top_down,
            half_open_m=p.get("curobo_half_open_m"),
        ):
            position_error_m = self.arms.position_error(
                side, self.tcp_target_for_pads(side, grasp_target, top_down)
            )
            self._log_phase(
                "curobo_grasp_flown",
                True,
                object=object_name,
                side=side,
                position_error_m=round(position_error_m, 4),
            )
            return {
                "position_error_m": round(position_error_m, 4),
                "strict_reach": position_error_m <= 0.015,
                "ee_dy_m": 0.0,
                "reason": "curobo_grasp_flown",
            }

        tcp_grasp_target = self.tcp_target_for_pads(
            side, grasp_target, top_down
        )
        # Back off to the pregrasp point along the gripper's OWN approach
        # axis. For a top-down grasp this is `(x, y, PREGRASP_Z)`, exactly as
        # before -- the offset is the vertical gap the old code wrote
        # literally. For a tilted approach, moving in world +Z instead would
        # carry the wrist sideways ACROSS the object rather than away from
        # it, which is the same lateral-slide error this function already
        # documents for the old hardcoded pregrasp_xy.
        pregrasp = _offset_along_approach(
            tcp_grasp_target,
            approach_tilt_deg,
            approach_azimuth_deg,
            vgl.PREGRASP_Z - tcp_grasp_target[2],
        )

        # set_arm_target (called inside self.arms.reach()) raises ValueError
        # if the target lies outside the CartesianTargetTracker's workspace
        # limits (handoff sec 47.3 / plan sec 1.3) -- this was NOT guarded
        # in any of this function's three arm.reach() calls until this fix,
        # and did escape uncaught live once Stage 4 started calling reach()
        # for every object (handoff sec 48): a neighboring object knocked
        # out of position by an earlier object's manipulation can leave a
        # LATER object's grasp target genuinely unreachable, and that must
        # fail this one skill attempt, not abort the whole stage.
        REACH_UNREACHABLE_RESULT = {
            "position_error_m": 999.0,
            "strict_reach": False,
            "ee_dy_m": 0.0,
            "reason": "reach_unreachable",
        }
        # OPEN THE JAWS BEFORE APPROACHING. `reach()` never did this: the
        # only caller of `arms.release()` anywhere is place/carry/push/pour,
        # so the approach ran with whatever gripper state the previous action
        # happened to leave behind -- and after any failed grasp attempt that
        # state is CLOSED (`gripper_position_rad` 0.0003-0.0073 in every
        # close log). Every retry therefore drove a closed fist into the
        # object.
        #
        # This is the shape of damage seen on every object in every run:
        # plate2 sliding 14.5 cm across the counter before falling off, cup
        # pushed 0.0852 m and the jaws then closing 7.6 cm away from it
        # (`close_outcome: contact_lost`). The grasp point, the orientation
        # and the IK were all fine -- the hand was shut.
        #
        # Failure is logged, not fatal: an unopened gripper makes THIS grasp
        # worse, aborting the reach makes the whole object fail.
        # 2026-08-20 (real bug, GPU-confirmed): `release()` used to run at
        # the asset's native stiffness (3.0, see the comment below on why
        # that value is so soft) -- measured 28/28 `open_before_approach`
        # calls failing across three real runs, gripper_position_rad
        # stuck at 0.70-0.82 against a GRIPPER_OPEN_RAD=0.9 target
        # (tolerance 0.02), never actually reaching open within the 1.5s
        # budget. A gripper that never opens past ~90% has less margin to
        # accept the object cleanly, and a close was never once even
        # attempted in any of those runs. Same fix as the comment below
        # already applies AFTER opening (raise to CLOSE_GRIPPER_STIFFNESS
        # so the open position holds through approach contact) -- doing
        # it BEFORE too makes the open command itself converge for real.
        try:
            self.arms.set_gripper_stiffness(side, CLOSE_GRIPPER_STIFFNESS)
        except Exception as exc:  # noqa: BLE001
            print(f"WARN: pre-open gripper stiffness set failed ({exc!r})", flush=True)
        gripper_open_ok = self.arms.release(
            side, step=self._tick, dt=self.sim.cfg.dt, timeout_s=1.5
        )
        # 2026-08-16: the gripper ran the WHOLE approach (pregrasp,
        # descend_standoff, the 600-tick descend_gentle_ramp) on the
        # asset's native stiffness (3.0) -- `grasp()` (a separate method,
        # called only after `reach()` returns) is the only place that
        # ever raised it to CLOSE_GRIPPER_STIFFNESS, and only right
        # before the deliberate close ramp. At stiffness 3.0 the holding
        # torque is negligible (~0.2 N*m for a few hundredths rad of
        # error), so any incidental contact during the approach -- and
        # the standoff sits close enough to the object that this is
        # exactly what descend_aborted_object_pushed (moved_m 0.0851)
        # exists to catch -- trivially shoves the passive linkage open.
        # Measured this session: open_before_approach logged
        # gripper_position_rad 0.8804 (sensible), but the close ramp's
        # own first tick started at 0.9762-0.9847 -- already drifted
        # past GRIPPER_OPEN_RAD=0.9 before any deliberate close command
        # ran, which is consistent with every close this session then
        # ending at ~1.0-1.02 regardless of stiffness/stall/damping
        # tuning at the CLOSE end alone. Applying the tuned stiffness
        # here too, so the open position is actually held through
        # approach contact instead of drifting. NOT YET n>=3 verified.
        try:
            self.arms.set_gripper_stiffness(side, CLOSE_GRIPPER_STIFFNESS)
        except Exception as exc:  # noqa: BLE001
            print(
                f"WARN: approach gripper stiffness set failed ({exc!r})",
                flush=True,
            )
        self._log_phase(
            "open_before_approach",
            gripper_open_ok,
            object=object_name,
            gripper_position_rad=round(self.arms.gripper_position(side), 4),
        )

        pregrasp_ik_stats: dict[str, Any] = {}
        try:
            ok = self.arms.reach(
                side,
                pregrasp,
                top_down,
                step=self._tick,
                dt=self.sim.cfg.dt,
                timeout_s=8.0,
                ik_stats=pregrasp_ik_stats,
                # Same joint-thrash guard as push_approach/push_standoff
                # (see those call sites) -- this reach() call never had it,
                # which is how v13's IK elbow/wrist flip during the
                # cup-grasp descent went undetected: EE position error
                # stayed small so the trend-based joint_runaway guard
                # never fired, but the COMMANDED joint target itself
                # jumped ~1.2 rad in ~130 ticks. This catches that
                # regardless of what the EE error says.
                max_joint_delta_rad=p.get(
                    "reach_pregrasp_max_joint_delta_rad", 0.5
                ),
                # 2026-08-20 (EBiM Task 3): PROGRESS.md's own per-tick
                # telemetry shows pregrasp is the single biggest per-attempt
                # cost -- it converges to ~0.02-0.04m within ~200 ticks then
                # idles the remaining ~1400 of its 1600-tick budget without
                # ever crossing DEFAULT_POSITION_TOLERANCE_M (0.02m). IK is
                # succeeding every tick and no joint thrash/runaway fires,
                # so neither guard above catches it -- a real convergence
                # plateau, not danger. 400 ticks (2s) is longer than the
                # ~200-tick window this codebase's own trace sampling
                # already shows real convergence completing in, so a
                # genuinely still-converging descent is not cut short.
                plateau_bail_ticks=p.get(
                    "reach_pregrasp_plateau_bail_ticks", 400
                ),
            )
        except ValueError as exc:
            self._log_phase("pregrasp", False, reason=str(exc))
            return REACH_UNREACHABLE_RESULT
        self._log_phase(
            "pregrasp",
            ok,
            target=[round(v, 3) for v in pregrasp],
            ik=pregrasp_ik_stats,
        )

        # STEP 3 (plans/handoff.md sec 5): drive fast down to a standoff
        # just above the grasp point, then rate-limit the final approach so
        # contact with the object is gentle -- servoing at full rate all
        # the way down is what let the fingers shove the cup before ever
        # settling. `gentle_descend_m` and `gentle_descend_seconds` are
        # tunable (Step 3's own suggested sweep is the standoff/duration).
        # Defined before the ramp block so the final-converge check below
        # is valid even when the ramp is disabled (gentle_descend_m == 0).
        ramp_aborted_tick = None
        gentle_descend_m = p.get("gentle_descend_m", GENTLE_DESCEND_M)
        gentle_descend_seconds = p.get(
            "gentle_descend_seconds", GENTLE_DESCEND_SECONDS
        )
        if gentle_descend_m > 0.0:
            standoff_target = _offset_along_approach(
                tcp_grasp_target,
                approach_tilt_deg,
                approach_azimuth_deg,
                gentle_descend_m,
            )
            standoff_ik_stats: dict[str, Any] = {}
            try:
                standoff_ok = self.arms.reach(
                    side,
                    standoff_target,
                    top_down,
                    step=self._tick,
                    dt=self.sim.cfg.dt,
                    timeout_s=6.0,
                    ik_stats=standoff_ik_stats,
                    # See pregrasp call above -- v13's IK flip was measured
                    # in this exact phase (descend_standoff).
                    max_joint_delta_rad=p.get(
                        "reach_standoff_max_joint_delta_rad", 0.5
                    ),
                )
            except ValueError as exc:
                self._log_phase("descend_standoff", False, reason=str(exc))
                return REACH_UNREACHABLE_RESULT
            self._log_phase(
                "descend_standoff",
                standoff_ok,
                target=[round(v, 3) for v in standoff_target],
                ik=standoff_ik_stats,
            )
            ramp_ticks = max(
                1, round(gentle_descend_seconds / self.sim.cfg.dt)
            )
            # 2026-08-14: this loop used to command blindly for all
            # `ramp_ticks` and then log ok=True unconditionally -- it
            # reported success no matter what the arm actually did. It is
            # the only motion in reach() with no tracking check, and it is
            # where the objects were being destroyed.
            #
            # Measured (`outputs/cup_only.log`, cup): at
            # `descend_standoff` the wrist is correctly placed 0.14 m above
            # the cup (ee [-4.204,-1.747,0.891], obj [-4.185,-1.755,0.751]).
            # One phase later the wrist is at [-4.062,-0.980,1.789] -- 0.9 m
            # UP and 0.77 m sideways, against a command whose x/y never
            # change and whose z only descends -- and the cup has been
            # flung to [-2.771,-1.322,0.814], landing on the floor at
            # z=0.091 about 1.5 m away. Every later grasp then "failed"
            # against an object that was no longer on the counter, which is
            # what made this look like a gripper problem for so long.
            #
            # So: verify the wrist is actually following the ramp, and bail
            # the moment it is not, instead of thrashing for the remaining
            # ticks. The tolerance is derived, not fitted -- it is the
            # ramp's own travel (`gentle_descend_m`), i.e. "the wrist has
            # deviated by more than the whole motion we are performing",
            # which is unambiguously divergence at any scale or scene.
            ee_index = 0 if side == "left" else 1
            ramp_tolerance_m = gentle_descend_m
            ramp_deviation_m = 0.0
            ramp_aborted_tick = None
            # This loop drives set_arm_target/command() directly instead of
            # going through arms.reach(), so it never had ANY joint-space
            # guard -- only the EE-deviation check below, which an IK
            # elbow/wrist flip can pass (EE barely moves, joint config
            # jumps). Same guard and threshold as the reach() calls above
            # in this function; this is the leg the comments above already
            # identify as "where the objects were being destroyed", so it
            # is the highest-risk gap to leave unguarded.
            ramp_max_joint_delta_rad = p.get(
                "reach_ramp_max_joint_delta_rad", 0.5
            )
            ramp_arm_joint_ids = list(
                getattr(self.arms.joint_groups, f"{side}_arm")
            )
            ramp_prev_commanded_joints: list[float] | None = None
            ramp_joint_flip_tick: int | None = None
            try:
                for tick in range(1, ramp_ticks + 1):
                    # Interpolate the FULL 3-D point from the standoff to the
                    # grasp target. This used to ramp z alone and set x/y
                    # straight to the target, which was equivalent while the
                    # standoff differed from the target in z only -- i.e.
                    # while every approach was straight down.
                    #
                    # Once the standoff is offset along a tilted approach
                    # axis, holding x/y fixed makes tick 1 a STEP of up to
                    # `gentle_descend_m * sin(tilt)` (5.7 cm at 45 degrees).
                    # The wrist cannot follow a step, so it lags, and the
                    # tracking guard correctly aborts the ramp for divergence
                    # that the ramp itself commanded. Measured on
                    # outputs/keep_live_er_run3.log: aborted at tick 568 of
                    # 600 with ramp_deviation_m 0.0854 against an 0.08
                    # tolerance, after a 45-degree ER-2 approach.
                    #
                    # Straight-line interpolation is byte-identical to the
                    # old arithmetic for a top-down approach, because there
                    # standoff x/y already equal the target's.
                    frac = tick / ramp_ticks
                    commanded = tuple(
                        standoff_target[i]
                        + (tcp_grasp_target[i] - standoff_target[i]) * frac
                        for i in range(3)
                    )
                    self.arms.set_arm_target(side, commanded, top_down)
                    self.arms.command()
                    commanded_joints_now = self.arms._position_targets[
                        0, ramp_arm_joint_ids
                    ].tolist()
                    if ramp_prev_commanded_joints is not None:
                        joint_delta = max(
                            abs(c - p_)
                            for c, p_ in zip(
                                commanded_joints_now, ramp_prev_commanded_joints
                            )
                        )
                        if joint_delta > ramp_max_joint_delta_rad:
                            ramp_joint_flip_tick = tick
                            ramp_aborted_tick = tick
                            break
                    ramp_prev_commanded_joints = commanded_joints_now
                    self._tick()
                    ee_now = self.arms.ee_world_poses()[ee_index][0]
                    ramp_deviation_m = math.dist(commanded, ee_now)
                    if ramp_deviation_m > ramp_tolerance_m:
                        ramp_aborted_tick = tick
                        break
            except ValueError as exc:
                self._log_phase("descend_gentle_ramp", False, reason=str(exc))
                return REACH_UNREACHABLE_RESULT
            self._log_phase(
                "descend_gentle_ramp",
                ramp_aborted_tick is None,
                ramp_ticks=ramp_ticks,
                ramp_seconds=round(gentle_descend_seconds, 3),
                ramp_deviation_m=round(ramp_deviation_m, 4),
                ramp_tolerance_m=round(ramp_tolerance_m, 4),
                ramp_aborted_tick=ramp_aborted_tick,
                ramp_joint_flip_tick=ramp_joint_flip_tick,
                ramp_joint_flip_max_delta_rad=ramp_max_joint_delta_rad,
            )
            if ramp_joint_flip_tick is not None:
                print(
                    "GENTLE_RAMP_JOINT_FLIP "
                    f"object={object_name!r} side={side!r} "
                    f"tick={ramp_joint_flip_tick}/{ramp_ticks} "
                    f"max_delta_rad={ramp_max_joint_delta_rad:.4f}",
                    flush=True,
                )
            if ramp_aborted_tick is not None:
                print(
                    "GENTLE_RAMP_DIVERGED "
                    f"object={object_name!r} side={side!r} "
                    f"tick={ramp_aborted_tick}/{ramp_ticks} "
                    f"deviation_m={ramp_deviation_m:.4f} "
                    f"tolerance_m={ramp_tolerance_m:.4f}",
                    flush=True,
                )
                return REACH_UNREACHABLE_RESULT

        # REVERTED (plans/handoff.md sec 4.16-4.17): two experiments this
        # session tried to remove or shorten this final full-speed
        # converge call (sec 6 option (a): drop it entirely; a settle-dwell
        # compromise) on the theory that it was only "shoving the cup".
        # Both measured WORSE than this call being present: removing it
        # entirely starved base-hold of settle time (base_anchor_err_m blew
        # out to 24.8cm, cup flung ~3.26m); the settle-dwell compromise
        # fixed the fling but base drift was still worse than this
        # baseline and the net grasp result (gripper 0.111 rad, object_ee
        # _dist 0.152m) did not beat this call's own 0.223 rad / 0.107m.
        # This call is a real, keep-it fix, not a defect -- restored
        # verbatim to the config that produced the session's best result.
        # `skip_final_converge=True` reproduces the (confirmed worse)
        # experiment without another code edit, if ever needed again.
        # 2026-08-14: the two reverted experiments above both REMOVED this
        # call unconditionally, which is why they lost the base-hold settle
        # time and measured worse. This is a third, different option: keep
        # the call in every case where it does work, and skip it only when
        # the ramp has ALREADY landed the wrist inside this call's own
        # arrival tolerance -- i.e. when converging is definitionally a
        # no-op and all it can contribute is a shove.
        #
        # That case now exists because it did not before: with the ramp
        # tracking guard the ramp finishes on target instead of thrashing.
        # Measured on outputs/pad_probe.log (cup): the ramp completes with
        # the cup still on the counter at z=0.747, then this converge moves
        # the wrist just 0.017 m (0.875 -> 0.858) and flings the cup 0.6 m
        # to the floor at z=0.035. Every later phase then "failed" against
        # an object that was no longer there.
        #
        # Derived, not fitted: the threshold IS this call's own
        # position_tolerance_m, so skipping can never accept a pose the
        # call itself would have rejected.
        descend_ik_stats: dict[str, Any] = {}
        final_converge_tolerance_m = 0.015
        ee_before_converge = self.arms.ee_world_poses()[
            0 if side == "left" else 1
        ][0]
        converge_gap_m = math.dist(tcp_grasp_target, ee_before_converge)
        already_converged = (
            ramp_aborted_tick is None
            and gentle_descend_m > 0.0
            and converge_gap_m <= final_converge_tolerance_m
        )
        if already_converged:
            print(
                "FINAL_CONVERGE_SKIPPED_ALREADY_THERE "
                f"object={object_name!r} gap_m={converge_gap_m:.4f} "
                f"tolerance_m={final_converge_tolerance_m:.4f}",
                flush=True,
            )
            strict_reach = True
        elif not p.get("skip_final_converge", False):
            # Same object-displacement guard the pre-close re-center uses.
            # This converge is the leg where the damage actually happens:
            # measured on run 8, `base_anchor_err_m` jumps 0.0476 -> 0.1236
            # across it while plate2 slides across the counter. Those two
            # facts are one fact -- the arm is pressing into an object that
            # cannot yield, and Newton's third law moves the lighter thing,
            # which here is the 374 kg base on velocity-controlled wheels
            # with a ~2 s tracking lag. Stop pressing and the base stops
            # being dragged.
            converge_obj_start = self.object_position(object_name)

            def _converge_step() -> None:
                self._tick()
                moved = math.dist(
                    self.object_position(object_name), converge_obj_start
                )
                if moved > RECENTER_MAX_OBJECT_PUSH_M:
                    raise _ObjectPushedAway(moved)

            try:
                strict_reach = self.arms.reach(
                    side,
                    tcp_grasp_target,
                    top_down,
                    step=_converge_step,
                    dt=self.sim.cfg.dt,
                    timeout_s=6.0,
                    position_tolerance_m=0.015,
                    ik_stats=descend_ik_stats,
                    # See pregrasp call above -- same guard, this is the
                    # final full-speed converge onto the grasp point.
                    max_joint_delta_rad=p.get(
                        "reach_final_max_joint_delta_rad", 0.5
                    ),
                    # 2026-08-20: this phase is now the single biggest
                    # per-run cost (measured: 31% of a full run's wall
                    # time, more than pregrasp) -- same plateau signature
                    # as pregrasp's own fix (position_error stuck ~0.024-
                    # 0.035m against this call's 0.015m tolerance for the
                    # rest of the 1200-tick budget), just never guarded
                    # here. Same 400-tick window.
                    plateau_bail_ticks=p.get(
                        "reach_final_plateau_bail_ticks", 400
                    ),
                )
            except _ObjectPushedAway as pushed:
                strict_reach = False
                self._log_phase(
                    "descend_aborted_object_pushed",
                    False,
                    object=object_name,
                    moved_m=round(pushed.moved_m, 4),
                    limit_m=RECENTER_MAX_OBJECT_PUSH_M,
                )
            except ValueError as exc:
                self._log_phase("descend", False, reason=str(exc))
                return REACH_UNREACHABLE_RESULT
        else:
            strict_reach = False
        position_error_m = self.arms.position_error(side, tcp_grasp_target)
        contact_tolerance = vgl.FINAL_APPROACH_CONTACT_TOLERANCE_M
        reached = strict_reach or position_error_m <= contact_tolerance

        ee_pos = self.arms.ee_world_poses()[0 if side == "left" else 1][0]
        ee_dy = ee_pos[1] - tcp_grasp_target[1]

        self._last_grasp_target = {object_name: grasp_target}
        # M0 (ACTIVE_BRIEF sec 3/5): remember the offset from the object's
        # LIVE origin to the commanded grasp point (standoff along the tool
        # axis + any x/y rim offset). grasp()'s hold check re-derives the
        # expected EE position from the object's CURRENT pose using this
        # same offset, instead of comparing raw object-origin distance
        # against the wrist -- the old comparison always included the
        # standoff itself, making the gate arithmetically unsatisfiable.
        self._last_grasp_offset[object_name] = (
            grasp_target[0] - live_obj[0],
            grasp_target[1] - live_obj[1],
            grasp_target[2] - live_obj[2],
        )
        self._log_phase(
            "descend",
            reached,
            strict_reach=reached,
            position_error_m=round(position_error_m, 4),
            target=[round(v, 3) for v in grasp_target],
            ik=descend_ik_stats,
        )
        return {
            "position_error_m": round(position_error_m, 4),
            "strict_reach": bool(reached),
            "ee_dy_m": round(ee_dy, 4),
        }

    def grasp(self, side, object_name, **p) -> dict:
        self._active_object = object_name

        # Same fallen-object refusal as `reach()`, and it needs to be here
        # too because reach and grasp are SEPARATE skills in the stage plan.
        # Refusing the reach does not refuse the grasp: run 21 skipped the
        # reach for the fallen cup (`reach_skipped_object_has_fallen`,
        # current_z 0.0339) and then closed the gripper anyway with the
        # object **2.766 m away** -- 300 ticks of closing on empty air, and
        # a `close` record that has to be read carefully to see it meant
        # nothing.
        if self._object_has_fallen(object_name):
            self._log_phase(
                "grasp_skipped_object_has_fallen",
                False,
                object=object_name,
                spawn_z=round(
                    getattr(self, "_spawn_object_z", {}).get(object_name, 0.0), 4
                ),
                current_z=round(
                    float(self.object_position(object_name)[2]), 4
                ),
            )
            return {
                "held": False,
                "scored": False,
                "reason": "object_has_fallen_off_its_surface",
            }

        vgl = self._m["vgl"]
        m = self._m

        # LEVER 1: re-center on the LIVE cup pose right before close.
        # Between reach() and grasp(), the cup may have moved due to contact.
        # Re-read the actual PhysX pose and recompute the close target.
        #
        # T5 (ACTIVE_BRIEF.md/handoff sec 23-24): tried generalizing this to
        # every object (the base keeps drifting during descend's
        # gentle-ramp settle -- base_anchor_err_m 0.03-0.05 m observed live
        # -- and the arm holds a WORLD-frame target computed once in
        # reach(), so that drift alone pulls the end-effector 0.13-0.29 m
        # from non-cup objects by close time). **Reverted** (handoff sec
        # 24-25): for non-cup objects this raised an uncaught
        # `ValueError: world target lies outside CartesianTargetTracker
        # limits` and aborted the whole stage -- confirmed byte-for-byte
        # reproducible on both Stage 1 (with a preceding reach()/descend)
        # and Stage 4 (cold start, no preceding reach()) after a clean
        # container restart ruled out environment contamination. A hard
        # crash is strictly worse than the drift it was meant to fix.
        # Back to cup-only, plus defensive exception handling so this
        # exception class can never again abort a stage outright.
        live_obj = self.object_position(object_name)
        live_grasp_target = None
        if object_name == "cup":
            # 2026-08-20 (EBiM Task 3 forensic audit + fix): this branch
            # used to unconditionally recompute from the hand-fitted
            # CUP_RIM_X_OFFSET/CUP_GRASP_Y_OFFSET constants, discarding
            # whatever target the immediately-preceding reach() actually
            # commanded to (e.g. a grasp_xyz_override from a validated
            # descent correction or a live grasp candidate) -- every other
            # object already avoids this via _last_grasp_offset below.
            # `224d91f` worked around it per-caller by manually computing
            # and passing cup_rim_x_offset/grasp_y_offset; this makes the
            # same correctness automatic for every caller by giving cup
            # the identical _last_grasp_offset-reuse path, falling back to
            # the constants only when there is no preceding reach() in
            # this run (_last_grasp_offset unset) or a caller explicitly
            # overrides -- same additive-only contract as
            # grasp_xyz_override in reach().
            last_offset = self._last_grasp_offset.get("cup")
            explicit_override = "cup_rim_x_offset" in p or "grasp_y_offset" in p
            if last_offset is not None and not explicit_override:
                live_grasp_target = (
                    live_obj[0] + last_offset[0],
                    live_obj[1] + last_offset[1],
                    live_obj[2] + last_offset[2],
                )
            else:
                grasp_xy_offset = p.get("grasp_y_offset", vgl.CUP_GRASP_Y_OFFSET)
                live_grasp_target = vgl.cup_grasp_target(
                    live_obj,
                    rim_x_offset=p.get("cup_rim_x_offset", vgl.CUP_RIM_X_OFFSET),
                    grasp_y_offset=grasp_xy_offset,
                    grasp_z_offset=0.0,
                )
            # Recenter used a fresher live pose than descend()'s -- refresh
            # the stored offset so the honest-hold check below uses this
            # more current standoff, not the one recorded before any drift.
            self._last_grasp_offset[object_name] = (
                live_grasp_target[0] - live_obj[0],
                live_grasp_target[1] - live_obj[1],
                live_grasp_target[2] - live_obj[2],
            )
        else:
            # 2026-08-14: T5's generalization of this re-center, restored.
            # It was reverted (sec 24-25) because `arms.reach` raised an
            # uncaught `ValueError: world target lies outside
            # CartesianTargetTracker limits` and aborted the whole stage --
            # but the `except ValueError` guard below, added for the cup at
            # the same time, already makes that failure non-fatal. The
            # reason for the revert no longer applies; the drift it was
            # meant to fix does.
            #
            # Measured on 2026-08-14 (`outputs/keep_v3.log`, plate2): the
            # grasp target was well chosen at 0.045 m from the object at
            # attempt start, the DESCENT then pushed the object 0.121 m,
            # and the gripper closed against a stale target 0.107 m away --
            # `close_outcome: contact_sustained` with
            # `object_follows_ee: False`. This matches the range this
            # method's own comment records for non-cup objects (0.13-0.29 m).
            #
            # Derived, not fitted: the offset is the one the grasp planner
            # itself chose in `descend` (stored at `_last_grasp_offset`,
            # object-origin -> commanded grasp point), re-anchored to where
            # the object actually is now. No new constant, and it holds for
            # any object, seed or scene.
            offset = self._last_grasp_offset.get(object_name)
            if offset is not None:
                live_grasp_target = (
                    live_obj[0] + offset[0],
                    live_obj[1] + offset[1],
                    live_obj[2] + offset[2],
                )
        # SECOND LOOK, AT CONTACT RANGE.
        #
        # The first ER-2 call happens before the arm has moved, from a wide
        # view of the whole counter, and GOTCHAS is explicit about what that
        # costs: "Wide-shot pointing lands ~5.7 cm off. Wide shot for
        # semantics, cropped or wrist-camera view for geometry. Two-stage."
        # Everything this pipeline has done so far was stage one only.
        #
        # By the time `grasp()` runs the gripper is centimetres from the
        # object, so a fresh capture is a close view of the thing about to be
        # grasped -- the geometry stage. Re-asking here corrects the wide
        # shot's error at the exact moment it matters, instead of carrying it
        # all the way into the close.
        #
        # Additive: any failure leaves `live_grasp_target` exactly as the
        # offset re-anchor computed it above.
        if self.live_er_grasp:
            try:
                second = self._live_er_grasp_pose(object_name, side)
                if second is not None:
                    previous = live_grasp_target
                    live_grasp_target = second.xyz
                    self._last_grasp_quat[object_name] = tuple(
                        second.quaternion_wxyz
                    )
                    shift = (
                        math.dist(previous, second.xyz)
                        if previous is not None
                        else None
                    )
                    self._log_phase(
                        "live_er_grasp_second_look",
                        True,
                        object=object_name,
                        shift_from_first_m=(
                            round(shift, 4) if shift is not None else None
                        ),
                        **second.as_log(),
                    )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARN: second-look ER-2 failed ({exc!r}); keeping the "
                    "re-anchored target",
                    flush=True,
                )

        if live_grasp_target is not None:
            recenter_ik_stats: dict[str, Any] = {}
            recenter_ok = False
            # The re-center is an ACCURACY improvement, never a
            # precondition: this method's own history (sec 24-25) is that a
            # hard crash here is strictly worse than the drift it fixes.
            # ValueError is the documented one (`world target lies outside
            # CartesianTargetTracker limits`); KeyError/AttributeError cover
            # a world whose helper seams are not all present. Every failure
            # is recorded and logged with ok=False rather than swallowed, so
            # a silently-skipped re-center is visible in the run log.
            try:
                # Reproduce the orientation `descend` actually commanded.
                # Forcing yaw=0 here rotated the wrist off the ranked
                # candidate's chosen grasp direction in the last instant
                # before closing -- for the cup this is 0.0 anyway, so its
                # behaviour is unchanged.
                #
                # 2026-08-14: rebuilding it as rpy(pi, 0, yaw) kept only the
                # YAW. Once reach() can command a tilted approach that is
                # actively destructive: it snapped the wrist from a 45-degree
                # approach back to straight down in the last instant before
                # closing, driving the pads through the object instead of
                # onto it. Use the quaternion reach() actually sent; the
                # rebuild remains the fallback for any object reach() has not
                # recorded one for, which is byte-identical to the old path.
                top_down = self._last_grasp_quat.get(object_name) or m[
                    "_quaternion_from_rpy"
                ](math.pi, 0.0, self._last_grasp_yaw.get(object_name, 0.0))
                # Abort the moment the re-center starts SHOVING the object
                # instead of converging on it. Same shape as the gentle
                # ramp's tracking guard, and for the same measured reason:
                # this is 800 ticks of servoing right at the object, and
                # nothing was watching what the object did during them.
                #
                # Measured (`outputs/keep_live_er_run5.log`, plate2): z held
                # at 0.7472 / 0.7583 / 0.7598 / 0.7596 through navigate,
                # standoff, ramp and descend, then read **0.0314** -- the
                # floor -- at recenter, having also slid 14.5 cm in x. The
                # re-center never converged either (`recenter_pos_err_m`
                # 0.0408 after 800 IK-clean ticks), because it was pushing
                # the object away as fast as it approached it.
                #
                # The tolerance is derived, not fitted: it is the gripper's
                # own jaw span. Once the object has moved further than the
                # jaws can open, the grasp this re-center is refining cannot
                # succeed no matter how long it servos, so continuing can
                # only make the scene worse.
                recenter_obj_start = self.object_position(object_name)

                def _recenter_step() -> None:
                    self._tick()
                    moved = math.dist(
                        self.object_position(object_name), recenter_obj_start
                    )
                    if moved > RECENTER_MAX_OBJECT_PUSH_M:
                        raise _ObjectPushedAway(moved)

                try:
                    recenter_ok = self.arms.reach(
                        side,
                        live_grasp_target,
                        top_down,
                        step=_recenter_step,
                        dt=self.sim.cfg.dt,
                        timeout_s=4.0,
                        position_tolerance_m=0.015,
                        ik_stats=recenter_ik_stats,
                    )
                except _ObjectPushedAway as pushed:
                    recenter_ok = False
                    self._log_phase(
                        "recenter_aborted_object_pushed",
                        False,
                        object=object_name,
                        moved_m=round(pushed.moved_m, 4),
                        limit_m=RECENTER_MAX_OBJECT_PUSH_M,
                    )
                ee_after = self.arms.ee_world_poses()[
                    0 if side == "left" else 1
                ][0]
                self._log_phase(
                    "recenter",
                    bool(recenter_ok),
                    target=[round(v, 3) for v in live_grasp_target],
                    live_obj=[round(v, 3) for v in live_obj],
                    ee_after=[round(v, 3) for v in ee_after],
                    recenter_pos_err_m=round(
                        math.dist(live_grasp_target, ee_after), 4
                    ),
                    ik=recenter_ik_stats,
                )
            except (ValueError, KeyError, AttributeError, TypeError) as exc:
                recenter_ik_stats["exception"] = f"{type(exc).__name__}: {exc}"
                print(
                    f"RECENTER_SKIPPED object={object_name!r} "
                    f"reason={type(exc).__name__}: {exc}",
                    flush=True,
                )

        # Give the close enough authority to actually close.
        #
        # Gripper stiffness has never been set anywhere in this codebase, so
        # it runs on the asset's authored 3.0. Torque is `stiffness * error`,
        # so even a full 1.0 rad of error yields ~3 N*m against an authored
        # 50 N*m effort limit -- the limit is never reached, and
        # set_gripper_effort_scale can only scale DOWN.
        #
        # Measured, aspire_1: the close commands 0 for 300 ticks and the
        # joint does not move at all, ending at 1.0039 rad -- 0.1 rad PAST
        # the 0.9 it was opened to, i.e. wedged further open by contact.
        # Positioning was fine in that run (pad midpoint 0.0472 m from the
        # grasp point), so the only thing left between it and a grasp was
        # closing force.
        #
        # NOT n>=3 verified. Logged so a run says what it used.
        close_stiffness = p.get(
            "close_gripper_stiffness", CLOSE_GRIPPER_STIFFNESS
        )
        if close_stiffness:
            try:
                self.arms.set_gripper_stiffness(side, close_stiffness)
                print(
                    f"GRIPPER_CLOSE_STIFFNESS side={side!r} "
                    f"stiffness={close_stiffness}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(f"WARN: gripper stiffness set failed ({exc!r})", flush=True)

        close_telemetry: dict[str, Any] = {}
        # Owner directive (2026-08-19): "don't move the base while picking
        # up the object" -- measured, real drift confirmed the complaint:
        # base_anchor_err_m reached 0.223m during a single-arm `close`
        # phase this session (vs <=0.05m in reach()'s own descend/pregrasp
        # phases, where the same hold mechanism is not fighting gripper
        # contact reaction force). The existing continuous hold
        # (`_tick()`'s `base_twist_toward` call, position_kp=4.0,
        # max_linear_mps=0.25) already runs every tick but is tuned for
        # ordinary navigation settle, not for resisting a sudden physical
        # push from the gripper closing against an object. Scoped,
        # reversible boost for exactly the close+settle window, restored
        # unconditionally in `finally` -- do not widen this to reach()'s
        # own phases without separately re-verifying those, they were
        # tuned against the current gains.
        _prior_hold_kp = self._base_hold_kp
        _prior_hold_max_mps = self._base_hold_max_mps
        self._base_hold_kp = 12.0
        self._base_hold_max_mps = 0.5
        holding = self.arms.grasp(
            side,
            step=self._tick,
            dt=self.sim.cfg.dt,
            settle_seconds=p.get("grasp_settle_seconds", 1.5),
            ramp_seconds=p.get("grasp_ramp_seconds", 1.0),
            close_effort_scale=p.get("close_effort_scale"),
            # REV13 T4 (plans/SYNC.md 2026-08-06): tried defaulting this
            # ON as a fix for T3's finding (sustained contact ground out
            # over ~150 ticks because the ramp kept forcing toward fully
            # closed). REVERTED: a real GPU re-run showed freezing on the
            # FIRST tick where measured lags commanded by >0.03 rad fires
            # on ordinary servo-response lag at the start of the ramp,
            # not just genuine object contact -- it froze the target at
            # 0.923 rad (tick 4, essentially still open) and the gripper
            # never closed far enough to hold anything at all
            # (`object_rise_m: 0.0`, worse than T3's un-fixed run).
            #
            # REV13 T4-followup (plans/SYNC.md 2026-08-07): comparing that
            # exact failure's raw tick data against T3's real contact
            # showed the discriminator T4 missed -- T3's genuine contact
            # was detected at commanded target 0.647 rad (deep into the
            # ramp, almost exactly the proven upper hold 0.6472); T4's
            # false trigger was at target 0.923 rad (barely past
            # GRIPPER_OPEN_RAD=0.9, before the ramp had done anything).
            # `contact_freeze_max_target_rad` (arms.py,
            # `run_gripper_close_ramp`) gates the freeze on that signal:
            # only freeze when contact is detected at or below that value.
            # Explicit opt-in here (default still OFF) pending live GPU
            # verification of THIS refined trigger, same caution T4 itself
            # used before it was ever defaulted on.
            hold_target_on_contact=p.get(
                "close_hold_on_contact", self.close_hold_on_contact
            ),
            contact_freeze_max_target_rad=p.get(
                "close_contact_freeze_max_target_rad",
                DEFAULT_CONTACT_FREEZE_MAX_TARGET_RAD,
            ),
            # 2026-08-16: comparing only against the immediately-previous
            # tick (the arms.py default, 1) missed a real, sustained cup
            # contact because the measured position jitters a few
            # thousandths of a radian tick to tick under resistance
            # (verify_grasp_lift.py cup runs,
            # outputs/task3_verify_grasp_lift/close_trace/
            # close_ramp_ticks.json) -- every uptick from that jitter read
            # as "closing" and reset the stall counter before it reached
            # `stall_ticks_required`. 10 ticks (0.05s) filters that noise
            # while still reading the genuine multi-tick closing descent
            # as closing, not stalled -- CPU-proven
            # (test_run_gripper_close_ramp_stall_lookback_filters_jitter)
            # and GPU-confirmed to actually engage the freeze where the
            # default did not. NOT YET n>=3 verified on this pipeline.
            stall_lookback_ticks=p.get("close_stall_lookback_ticks", 10),
            telemetry=close_telemetry,
        )
        self._base_hold_kp = _prior_hold_kp
        self._base_hold_max_mps = _prior_hold_max_mps
        gripper_rad = self.arms.gripper_position(side)
        ee_pos = self.arms.ee_world_poses()[0 if side == "left" else 1][0]
        # 2026-08-14: `_last_grasp_offset` is now recorded in PAD space (see
        # tcp_target_for_pads), so the hold check has to compare the pads,
        # not the wrist -- otherwise it carries the 18.6 mm tool offset as
        # permanent error. Falls back to the wrist when the pads cannot be
        # measured, which is the previous behaviour exactly.
        pad_ids = self._pad_body_ids(side)
        if len(pad_ids) == 2:
            mid = [0.0, 0.0, 0.0]
            for _, idx in pad_ids:
                bp = self.robot.data.body_pos_w[0][idx]
                for a in range(3):
                    mid[a] += float(bp[a]) / 2.0
            ee_pos = tuple(mid)
            # Log where the PADS actually ended up against the object. The
            # tool-offset correction (b079936) shifts them along the tool
            # axis, and the pre-existing standoff constants were fitted
            # against the UNCORRECTED frame, so the two can double-count.
            # This is the measurement that settles the sign and size of
            # that interaction instead of arguing about it.
            _obj_now = self.object_position(object_name)
            print(
                "PAD_VS_OBJECT "
                f"object={object_name!r} "
                f"pad_mid=[{mid[0]:.4f},{mid[1]:.4f},{mid[2]:.4f}] "
                f"obj=[{_obj_now[0]:.4f},{_obj_now[1]:.4f},{_obj_now[2]:.4f}] "
                f"pad_minus_obj_z={mid[2] - _obj_now[2]:+.4f} "
                f"tool_offset={self.tool_offset(side)}",
                flush=True,
            )
        object_pos = self.object_position(object_name)
        # M0 (ACTIVE_BRIEF sec 3/5): the raw object-origin<->wrist distance
        # ALWAYS includes the commanded standoff (0.068-0.10 m along the
        # tool axis, plus any x/y rim offset) -- that made the old gate
        # arithmetically unsatisfiable even for a perfect grasp. Compare the
        # wrist instead against the grasp frame: the object's CURRENT
        # position shifted by the same offset used to command the descend
        # (or the cup's fresher recenter offset above). What's left is the
        # real alignment error -- IK residual plus any drift since descend.
        offset = self._last_grasp_offset.get(object_name, (0.0, 0.0, 0.0))
        grasp_frame_point = (
            object_pos[0] + offset[0],
            object_pos[1] + offset[1],
            object_pos[2] + offset[2],
        )
        dist = math.dist(grasp_frame_point, ee_pos)
        raw_dist = math.dist(object_pos, ee_pos)
        follows_ee = vgl.object_follows_end_effector(
            grasp_frame_point,
            ee_pos,
            max_distance_m=p.get(
                "max_held_object_distance_m",
                config.THRESHOLDS.GRASP_HELD_MAX_DIST_M,
            ),
        )
        if holding and follows_ee:
            self._held = object_name
            self._held_side = side
        else:
            self._held = None
        # REV14 T4 hypothesis 1 (plans/SONNET_START_2026-08-07_
        # REV14_SINGLE_VM.md S3): "a grasp with contact:True /
        # object_follows_ee:True still never becomes load-bearing" has
        # three live hypotheses -- authored effort ceiling too low,
        # contact geometry, or friction. This number has never been
        # printed anywhere in this project's history; logging it is the
        # cheapest possible first step (zero new API surface --
        # `_default_gripper_effort_limits` is already computed at
        # DualArmController.__init__ and used by set_gripper_effort_scale/
        # restore_gripper_effort_limit). Per-tick COMMANDED/MEASURED
        # effort during close+lift would fully discriminate the
        # hypothesis but needs an Isaac Lab ArticulationData field
        # (likely `applied_torque`) that is not used anywhere else in
        # this codebase -- deliberately NOT guessed here; verify the
        # real attribute name against a live Isaac Lab session before
        # adding that part, rather than risk crashing tick 0 of a real
        # GPU episode on a wrong field name.
        authored_effort_limit_nm = self.arms._default_gripper_effort_limits[
            side
        ]
        # WHERE THE PADS ACTUALLY ARE relative to the object, at the moment
        # the close finishes. Every close in this session ends the same way
        # -- `close_contact_tick: 8` regardless of object or pose, and
        # `gripper_position_rad` travelling all the way to ~0 -- which says
        # the jaws met each other rather than the object, and that whatever
        # the wrist was doing, the grasp feature was never between the pads.
        # Nothing logged so far can distinguish "pads straddled the rim and
        # it slipped" from "pads closed beside the rim", and those need
        # opposite fixes. This is the measurement that separates them:
        # per-pad world positions, their separation, and each pad's distance
        # to the object.
        pad_telemetry: dict[str, Any] = {}
        try:
            pad_ids = self._pad_body_ids(side)
            positions = self.robot.data.body_pos_w
            pads = {
                name: tuple(
                    round(float(v), 4) for v in positions[0, index].tolist()
                )
                for name, index in pad_ids
            }
            pad_points = list(pads.values())
            pad_telemetry["pads_world"] = pads
            if len(pad_points) == 2:
                pad_telemetry["pad_separation_m"] = round(
                    math.dist(pad_points[0], pad_points[1]), 4
                )
                midpoint = tuple(
                    (a + b) / 2.0
                    for a, b in zip(pad_points[0], pad_points[1])
                )
                pad_telemetry["pad_midpoint_to_object_m"] = round(
                    math.dist(midpoint, object_pos), 4
                )
                # Against the GRASP POINT too: the pads are supposed to
                # straddle the feature ER-2 chose, not the object's origin,
                # and for a rim grasp those are several cm apart.
                pad_telemetry["pad_midpoint_to_grasp_point_m"] = round(
                    math.dist(midpoint, grasp_frame_point), 4
                )
                pad_telemetry["pad_to_object_m"] = [
                    round(math.dist(pt, object_pos), 4) for pt in pad_points
                ]
        except Exception as exc:  # noqa: BLE001
            pad_telemetry["pad_telemetry_error"] = repr(exc)

        # The phase's ok is the HONEST hold: the gripper stopped somewhere
        # an object could be holding it open AND the object is actually
        # travelling with the end-effector. `holding` alone is just
        # `gripper_holds_object(gripper_rad)`, a position-band check that
        # knows nothing about where the object is, and it reports success
        # for a gripper that stopped on nothing.
        #
        # Measured (run 20, spoon2): `holding` True at
        # `gripper_position_rad 0.0527` -- inside the 0.05-1.05 band -- while
        # `object_follows_ee` was False, the object sat 0.1254 m away, and
        # the new pad telemetry put the pad midpoint **0.1254 m from the
        # grasp point and 0.1643 m from the object**, pads 0.034 m apart.
        # The jaws had closed on nothing, 12 cm from the thing they were
        # sent for, and the phase called it a success.
        #
        # `holding` is still returned unchanged below, so every downstream
        # consumer (outcomes.classify_grasp's `contact` gate especially)
        # sees exactly what it saw before. Only the diagnostic verdict
        # changes, and it changes toward the truth.
        # A hold also requires the jaws to actually be CLOSED on something.
        # `gripper_holds_object`'s upper bound is now derived from the
        # joint's own authored USD limit (~1.0 rad, task3_autonomy/arms.py
        # `_gripper_position_upper_limit`) rather than the old hardcoded 1.05,
        # which sat past the mechanical limit and scored a jammed-fully-open
        # gripper (measured 1.0145) as holding.
        #
        # 2026-08-21: the second guard here used to be
        # `gripper_rad < GRIPPER_OPEN_RAD` (0.82), which did NOT make this
        # independent -- the band's own upper bound is the authored open
        # limit 0.8203, so both gates admitted a gripper at 0.819, i.e.
        # open and closed on nothing. `gripper_holds_object` now subtracts
        # a derived contact margin from the open limit, so it is the real
        # check; requiring the reading to be strictly below the measured
        # open position is kept as a cheap sanity bound on top of it.
        open_rad = self.arms._gripper_position_upper_limit(side)
        honest_hold = bool(holding and follows_ee and gripper_rad < open_rad)

        # HOLD THE OBJECT'S OWN WIDTH once we actually have it, instead of
        # continuing to command fully-closed.
        #
        # The close ramp's commanded target is 0 (shut) and nothing stops
        # commanding it after contact, so the jaws keep squeezing after a
        # good grasp until they slip past the object. `hold_target_on_contact`
        # was built for this and its `contact_freeze_max_target_rad = 0.65`
        # gate never fires here, because contact is reported at tick ~8 while
        # the target is still near the open 0.9 -- see that constant's own
        # comment for why the gate is right to be suspicious of such an early
        # contact.
        #
        # Measured, run 24, spoon2: `close ok=True` at
        # `gripper_position_rad 0.4781`, and by the FIRST carry sample
        # (tick 14750, before the base had gone anywhere) the gripper read
        # 0.001 and the object was already 0.2373 m away. The grasp was not
        # shaken loose by the drive -- it was squeezed out before the drive
        # began.
        #
        # The commanded width is derived, not chosen: the measured opening
        # the object itself is holding, minus one `DEFAULT_CONTACT_ERROR_RAD`
        # of deflection -- i.e. exactly the amount of squeeze this codebase
        # already calls "contact". Clamped at 0 so a thin object can never
        # command a negative target.
        # ...but only when the jaws are actually CLOSED on something.
        #
        # `gripper_holds_object`'s upper bound is now derived from the
        # joint's authored USD limit rather than the old hardcoded 1.05 (see
        # the comment above). Requiring the measured opening to be narrower
        # than the gripper's own open position is still kept as a second,
        # independent guard -- it costs nothing when the grasp is real.
        if honest_hold and gripper_rad < open_rad:
            try:
                from task3_autonomy.arms import DEFAULT_CONTACT_ERROR_RAD

                # Hold EXACTLY the width the object stopped the jaws at.
                #
                # Subtracting DEFAULT_CONTACT_ERROR_RAD (0.03) was an
                # over-squeeze that could exceed the object's whole width:
                # measured, cup_loop_3, the close stalled correctly on the
                # cup wall at 0.0618 rad, this latch then commanded 0.0318 --
                # half again tighter -- and by the carry's first sample the
                # gripper read 0.0002 with the cup 0.50 m away. The latch was
                # undoing the stall-freeze that had just done the right
                # thing.
                #
                # No margin is needed: the ramp froze because the jaws are
                # ALREADY pressing on the object, so holding that same target
                # maintains the force that stopped them. Anything tighter is
                # a squeeze the object did not ask for.
                hold_rad = max(0.0, gripper_rad)
                self.arms.set_gripper(side, hold_rad)
                self.arms.command()
                self._held_gripper_rad = hold_rad
                print(
                    "GRIPPER_HOLDING_OBJECT_WIDTH "
                    f"object={object_name!r} side={side!r} "
                    f"measured_rad={gripper_rad:.4f} "
                    f"commanded_rad={hold_rad:.4f}",
                    flush=True,
                )
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARN: could not latch gripper width ({exc!r})",
                    flush=True,
                )
        self._log_phase(
            "close",
            honest_hold,
            gripper_hold_predicate=bool(holding),
            gripper_position_rad=round(gripper_rad, 4),
            object_ee_dist_m=round(dist, 4),
            object_ee_dist_raw_m=round(raw_dist, 4),
            object_follows_ee=follows_ee,
            close_tick_count=close_telemetry.get("tick_count"),
            close_contact_tick=close_telemetry.get("contact_tick"),
            close_outcome=close_telemetry.get("outcome"),
            authored_effort_limit_nm=round(authored_effort_limit_nm, 4),
            **pad_telemetry,
        )
        return {
            "gripper_rad": round(gripper_rad, 4),
            "authored_effort_limit_nm": round(authored_effort_limit_nm, 4),
            # REV13 T2 (plans/SYNC.md 2026-08-06): tick-level close
            # telemetry -- per-tick commanded target/measured position/
            # error, the first tick contact was detected (if any), the
            # final residual, and an `outcome` classification
            # (`closed_no_contact` / `contact_sustained` / `contact_lost`).
            # Built to make a close that never starts and a close that
            # starts and slips distinguishable, per REV13 T1's finding
            # that no existing log could tell them apart.
            "close_telemetry": close_telemetry,
            # REV12 follow-up (T7 finding, plans/SYNC.md 2026-08-06):
            # this was hardcoded True regardless of `holding` (the real
            # signal from self.arms.grasp() -- whether the gripper close
            # command actually converged/contacted). That silently
            # defeated outcomes.classify_grasp's first gate (`contacted
            # = bool(metrics.get("contact", False))`) -- a caller could
            # never see a real MISS via this path, only WEAK_GRASP or
            # SUCCESS, even when the gripper never closed at all
            # (observed directly: T7 episodes 103/104 had
            # gripper_rad ~0.04-0.08 -- barely closed -- yet the old
            # hardcoded True meant `contacted` was always satisfied).
            "contact": bool(holding),
            "object_follows_ee": bool(follows_ee),
            "object_ee_dist_m": round(dist, 4),
            "object_ee_dist_raw_m": round(raw_dist, 4),
        }

    def _load_ranked_grasp_plan(
        self, object_name: str, side: str
    ) -> tuple[list[tuple[Any, Any]], tuple[float, float, float] | None]:
        """R9 T4: candidates for `reach_and_grasp_ranked`'s fall-through
        loop -- ranked entries for `side`, feasible first (by rank, which
        the contract already guarantees is ordered feasible-before-
        infeasible), each paired with its candidate's position/yaw/tilt
        via `candidate_id`. Returns `([], None)` (never raises) whenever
        the files are absent or malformed -- VM B has not written this
        object's candidates yet, the object has no ranked plan, or a
        write raced a read -- so the caller's hardcoded-pose fallback is
        always reachable.

        REV12 T5: also returns `candidate_file.object_pose` -- the
        object's pose AT CANDIDATE-GENERATION TIME, needed by the caller
        to re-anchor each candidate's absolute position to where the
        object actually is now (`task3_autonomy.grasp_reanchor`)."""
        try:
            candidate_file = load_candidates(object_name)
            ranked_file = load_ranked(object_name)
        except (FileNotFoundError, GraspContractError):
            return [], None
        by_id = {c.id: c for c in candidate_file.candidates}
        pairs = [
            (entry, by_id[entry.candidate_id])
            for entry in ranked_file.ranked
            if entry.side == side
            and entry.feasible
            and entry.candidate_id in by_id
        ]
        pairs.sort(key=lambda pair: pair[0].rank)
        return pairs, candidate_file.object_pose

    def reach_and_grasp_ranked(
        self, side: str, object_name: str, *, max_attempts: int = 4, **p
    ) -> dict:
        """R9 T4 (plans/LOOP_PROMPT_VM_A_REV9.md): try ranked grasp
        candidates in order, falling through to the next on failure,
        instead of committing to the single hardcoded pose and failing
        the object outright when it misses. Every attempt's real outcome
        (predicted feasible vs. actual `object_follows_ee`) is appended to
        `grasp_memory.jsonl` -- the learning-loop record.

        No ranked file for `object_name` (e.g. the cup path, which VM B
        has never generated ER candidates for) -> exactly one attempt at
        the ORIGINAL hardcoded pose, byte-identical to calling
        `reach()`/`grasp()` directly. Ranked candidates exhausted without
        a real hold -> one final hardcoded-pose attempt, same as today,
        rather than failing the object outright -- the hardcoded pose is
        the last resort, never removed.
        """
        full_plan, recorded_object_pose = self._load_ranked_grasp_plan(
            object_name, side
        )
        plan = full_plan[:max_attempts]
        attempts: list[dict[str, Any]] = []

        for rank_i, (ranked_entry, candidate) in enumerate(plan):
            # 2026-08-14: the finest-grained of the three deadline checks --
            # one ranked candidate is a full navigate + reach + grasp, ~9 min
            # on the measured 2026-08-14 runs, and max_attempts=4 of them run
            # inside a SINGLE skill invocation. Without this, an abort could
            # not land for ~36 min. See config.check_stage_deadline.
            config.check_stage_deadline(
                self, f"ranked candidate {rank_i} of {object_name!r}"
            )
            # REV12 T5: re-anchor to where the object ACTUALLY is now,
            # not the absolute position frozen at candidate-generation
            # time -- a failed grasp can knock an object off the
            # counter, or leave it flung far away, before a later
            # candidate ever gets tried (plans/SYNC.md PHYSICAL
            # REALITIES). `recorded_object_pose` is always set here
            # (non-None) whenever `plan` is non-empty -- both come from
            # the same successful `_load_ranked_grasp_plan` call.
            live_object_pose = self.object_position(object_name)
            reanchor = reanchor_candidate(
                candidate.position, recorded_object_pose, live_object_pose
            )
            if reanchor.action is not ReanchorAction.PROCEED:
                self._append_grasp_attempt_memory(
                    object_name,
                    candidate_id=candidate.id,
                    side=side,
                    stance_xy=ranked_entry.stance_xy,
                    stance_yaw=ranked_entry.stance_yaw,
                    predicted_feasible=ranked_entry.feasible,
                    object_follows_ee=False,
                    position_error_m=999.0,
                    source=f"reach_and_grasp_ranked_{reanchor.action.value}",
                )
                attempts.append(
                    {
                        "candidate_id": candidate.id,
                        "rank": ranked_entry.rank,
                        "abandoned": True,
                        "reanchor_action": reanchor.action.value,
                        "reanchor_reason": reanchor.reason,
                        "delta_xy_m": round(reanchor.delta_xy_m, 4),
                        "delta_z_m": round(reanchor.delta_z_m, 4),
                    }
                )
                continue
            grasp_target = reanchor.translated_position
            reach_result = self.reach(
                side,
                object_name,
                stance_xy_override=ranked_entry.stance_xy,
                stance_yaw_override=ranked_entry.stance_yaw,
                grasp_xyz_override=grasp_target,
                grasp_yaw_override=candidate.yaw_rad,
                **p,
            )
            # 2026-08-14: do not close a gripper on an approach that was
            # abandoned. reach() returns reason="reach_unreachable" when the
            # pregrasp/standoff/gentle-ramp could not be executed, but this
            # call used to run anyway -- and grasp()'s own pre-close
            # re-center then commands a fresh reach toward the object from
            # whatever pose the failed approach left the arm in.
            #
            # Measured (`outputs/cup_ramp.log`): the new ramp guard aborted
            # correctly with the cup SAFE on the counter at
            # [-4.148,-1.736,0.754], and the unconditional grasp() that
            # followed flung it to [-2.83,-0.78,0.034] anyway. Skipping to
            # the next ranked candidate is both safer and what the ranked
            # loop is for.
            if reach_result.get("reason") == "reach_unreachable":
                print(
                    "GRASP_SKIPPED_UNREACHABLE "
                    f"object={object_name!r} candidate={candidate.id} "
                    f"rank={ranked_entry.rank}",
                    flush=True,
                )
                self._append_grasp_attempt_memory(
                    object_name,
                    candidate_id=candidate.id,
                    side=side,
                    stance_xy=ranked_entry.stance_xy,
                    stance_yaw=ranked_entry.stance_yaw,
                    predicted_feasible=ranked_entry.feasible,
                    object_follows_ee=False,
                    position_error_m=reach_result.get(
                        "position_error_m", 999.0
                    ),
                    source="reach_and_grasp_ranked_unreachable",
                )
                attempts.append(
                    {
                        "candidate_id": candidate.id,
                        "rank": ranked_entry.rank,
                        "held": False,
                        "reason": "reach_unreachable",
                    }
                )
                continue
            grasp_result = self.grasp(side, object_name, **p)
            telemetry_held = bool(grasp_result.get("object_follows_ee"))
            # Owner directive (2026-08-19): telemetry alone (contact force /
            # object_follows_ee) has been wrong before -- a grip only
            # counts as real once a SECOND camera (this side's own wrist
            # cam, independent of whatever camera picked the candidate)
            # visually confirms the object is actually between the pads.
            # Do not proceed (this function returning "held" is what lets
            # the caller move on to lift/transport) on telemetry alone.
            cam_check = (
                self.verify_grasp_by_wrist_camera(side, object_name)
                if telemetry_held
                else {
                    "verified": False,
                    "pixel_frac": 0.0,
                    "reason": "skipped_telemetry_already_false",
                }
            )
            held = telemetry_held and cam_check["verified"]
            grasp_result["camera_verified"] = cam_check
            self._append_grasp_attempt_memory(
                object_name,
                candidate_id=candidate.id,
                side=side,
                stance_xy=ranked_entry.stance_xy,
                stance_yaw=ranked_entry.stance_yaw,
                predicted_feasible=ranked_entry.feasible,
                object_follows_ee=held,
                position_error_m=reach_result.get("position_error_m", 999.0),
                source="reach_and_grasp_ranked",
            )
            attempts.append(
                {
                    "candidate_id": candidate.id,
                    "rank": ranked_entry.rank,
                    "reach": reach_result,
                    "grasp": grasp_result,
                    "telemetry_held": telemetry_held,
                    "camera_verified": cam_check,
                }
            )
            if held:
                return {
                    "side": side,
                    "used_ranked_candidate_id": candidate.id,
                    "attempts": attempts,
                    "reach": reach_result,
                    "grasp": grasp_result,
                    "fell_back_to_hardcoded": False,
                }

        # No ranked plan, or every ranked candidate failed to hold: the
        # hardcoded pose is the last resort, never removed (hard rule --
        # this is what keeps the cup path, which has no ranked file,
        # scoring exactly as it does today).
        reach_result = self.reach(side, object_name, **p)
        grasp_result = self.grasp(side, object_name, **p)
        held = bool(grasp_result.get("object_follows_ee"))
        if plan:
            # Only log the fallback as its own memory entry when ranked
            # candidates were actually tried and exhausted -- the pure
            # no-ranked-file case (e.g. cup) has no candidate_id to
            # attribute the attempt to and would just be noise.
            self._append_grasp_attempt_memory(
                object_name,
                candidate_id=-1,
                side=side,
                stance_xy=(float("nan"), float("nan")),
                stance_yaw=0.0,
                predicted_feasible=False,
                object_follows_ee=held,
                position_error_m=reach_result.get("position_error_m", 999.0),
                source="reach_and_grasp_ranked_hardcoded_fallback",
            )
        attempts.append(
            {
                "candidate_id": None,
                "rank": None,
                "reach": reach_result,
                "grasp": grasp_result,
            }
        )
        return {
            "side": side,
            "used_ranked_candidate_id": None,
            "attempts": attempts,
            "reach": reach_result,
            "grasp": grasp_result,
            "fell_back_to_hardcoded": True,
        }

    def _append_grasp_attempt_memory(
        self,
        object_name: str,
        *,
        candidate_id: int,
        side: str,
        stance_xy: tuple[float, float],
        stance_yaw: float,
        predicted_feasible: bool,
        object_follows_ee: bool,
        position_error_m: float,
        source: str,
    ) -> None:
        # Never let a memory-log write failure (e.g. read-only filesystem
        # in some test/container configuration) abort a real grasp
        # attempt -- this is a record of what happened, not a gate on
        # whether it happened.
        with contextlib.suppress(OSError):
            append_grasp_memory(
                GraspMemoryEntry(
                    object=object_name,
                    candidate_id=candidate_id,
                    side=side,
                    stance_xy=tuple(stance_xy),
                    stance_yaw=float(stance_yaw),
                    predicted_feasible=bool(predicted_feasible),
                    object_follows_ee=bool(object_follows_ee),
                    position_error_m=float(position_error_m),
                    utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    source=source,
                )
            )

    def _servo_bimanual(
        self, right_position, left_position, quat, *, budget_s, tol_m=0.02
    ) -> bool:
        """Drive both arms to their own Cartesian targets in the same tick
        loop. Ported verbatim from verify_grasp_lift.py's `servo_bimanual`
        closure (GPU-verified 2026-08-16: reached tray targets in 69 ticks
        at ~1.5-1.8cm error, and cleared the cup pregrasp/descend legs of
        the first real end-to-end bimanual lift, n=3)."""
        for _ in range(math.ceil(budget_s / self.sim.cfg.dt)):
            self.arms.set_arm_target("right", right_position, quat)
            self.arms.set_arm_target("left", left_position, quat)
            result = self.arms.command()
            self._tick()
            right_error = self.arms.position_error("right", right_position)
            left_error = self.arms.position_error("left", left_position)
            if (
                result.right_succeeded
                and result.left_succeeded
                and right_error <= tol_m
                and left_error <= tol_m
            ):
                return True
        return (
            self.arms.position_error("right", right_position) <= tol_m
            and self.arms.position_error("left", left_position) <= tol_m
        )

    def reach_bimanual(self, object_name, **p) -> dict:
        """Bimanual pregrasp+descend: both arms converge on opposite sides
        of the object and hold there, ready for grasp_bimanual() to close.

        Ported from verify_grasp_lift.py's proven `--bimanual-cup` sequence
        (GPU-verified n=3 on the robotiq gripper, real navigation from
        spawn, 2026-08-16: cup lifted 0.316m, held 3.0s -- the first real
        end-to-end lift in this project's history). This reuses THIS
        world's own navigate_to()/_rotate_to() for the drive (NavigateTo
        already routes via the door internally -- task3_autonomy/skills.py
        -- so a single hop to the stance suffices, unlike that script's own
        separate multi-leg CORRIDOR_STOP/ROTATE_SPOT/rotate_west dance,
        which predates route_via_door's fix and is not needed here).

        Default stance/offsets are the exact literal values that script
        proved -- untouched, unre-derived. `_stance_for()` was not reused
        because it is tuned for the single-arm approach direction, and this
        one call sequence is already n=3 verified; kwargs are honoured so a
        caller CAN override without a code change.
        """
        self._active_object = object_name
        if self._object_has_fallen(object_name):
            self._log_phase(
                "reach_bimanual_skipped_object_has_fallen",
                False,
                object=object_name,
            )
            return {
                "ok": False,
                "reason": "object_has_fallen_off_its_surface",
            }

        m = self._m
        vgl = m["vgl"]

        stance_xy = p.get("stance_xy_override", (-3.32, -1.72))
        stance_yaw = p.get("stance_yaw_override", math.pi)
        settled_pose = self.adapter.pose()
        stance_gap_m = math.dist(
            (settled_pose.x, settled_pose.y), tuple(stance_xy)
        )
        if stance_gap_m > NAVIGATE_ARRIVAL_TOLERANCE_M:
            self.navigate_to(
                *stance_xy,
                max_linear_mps=0.25,
                budget_s=p.get("navigate_budget_s", 45.0),
                min_creep_mps=0.08,
            )
        self._rotate_to(stance_yaw)
        settled = self.adapter.pose()
        self._base_hold_anchor = (settled.x, settled.y)

        top_down = m["_quaternion_from_rpy"](TOP_DOWN_ROLL_RAD, 0.0, 0.0)

        x_offset = p.get("cup_rim_x_offset", vgl.CUP_RIM_X_OFFSET)
        y_sep = p.get("cup_y_separation", 0.055)
        right_y = p.get("cup_right_y_offset", y_sep)
        left_y = p.get("cup_left_y_offset", -y_sep)
        pregrasp_z = p.get("pregrasp_z", vgl.PREGRASP_Z)

        obj_xy_z = self.object_position(object_name)
        right_pregrasp = (
            obj_xy_z[0] + x_offset,
            obj_xy_z[1] + right_y,
            pregrasp_z,
        )
        left_pregrasp = (
            obj_xy_z[0] + x_offset,
            obj_xy_z[1] + left_y,
            pregrasp_z,
        )
        self.arms.set_gripper("left", GRIPPER_OPEN_RAD)
        self.arms.set_gripper("right", GRIPPER_OPEN_RAD)
        pregrasp_ok = self._servo_bimanual(
            right_pregrasp, left_pregrasp, top_down, budget_s=10.0
        )
        self._log_phase(
            "reach_bimanual_pregrasp",
            pregrasp_ok,
            right_target=[round(v, 3) for v in right_pregrasp],
            left_target=[round(v, 3) for v in left_pregrasp],
        )
        if not pregrasp_ok:
            return {"ok": False, "reason": "pregrasp_unreachable"}

        live_obj = self.object_position(object_name)  # re-read post-settle
        z_offset = p.get(
            "cup_grasp_z_offset", vgl.GRASP_HEIGHT_ABOVE_CUP_ORIGIN
        )
        right_grasp = (
            live_obj[0] + x_offset,
            live_obj[1] + right_y,
            live_obj[2] + z_offset,
        )
        left_grasp = (
            live_obj[0] + x_offset,
            live_obj[1] + left_y,
            live_obj[2] + z_offset,
        )

        # 2026-08-17: root-caused by scripts/task3/probe_bimanual_cup_
        # contact_geometry.py (plans/PROGRESS.md) -- a single fast IK-
        # tracked servo straight to the grasp target (the previous version
        # of this descend) measurably pushed the cup 16cm before the arms
        # finished converging, so the pre-computed target went stale by
        # arrival. A plain re-center-before-close wasn't enough either
        # (the cup kept moving DURING that recenter too). This ports the
        # single-arm path's own answer to the identical problem
        # (`reach()`'s `descend_standoff` + `descend_gentle_ramp`, this
        # file ~L3429): approach fast to a STANDOFF above the object where
        # there is no contact yet, then cover the last `gentle_descend_m`
        # at a bounded, rate-limited velocity so momentum -- and the
        # contact impulse that transfers to a light, freely-movable cup --
        # is small, instead of arriving at full IK-tracking speed. NOT YET
        # GPU-verified for bimanual.
        gentle_descend_m = p.get("gentle_descend_m", GENTLE_DESCEND_M)
        gentle_descend_seconds = p.get(
            "gentle_descend_seconds", GENTLE_DESCEND_SECONDS
        )
        right_standoff = (
            right_grasp[0],
            right_grasp[1],
            right_grasp[2] + gentle_descend_m,
        )
        left_standoff = (
            left_grasp[0],
            left_grasp[1],
            left_grasp[2] + gentle_descend_m,
        )
        standoff_ok = self._servo_bimanual(
            right_standoff, left_standoff, top_down, budget_s=8.0
        )
        self._log_phase(
            "reach_bimanual_standoff",
            standoff_ok,
            right_target=[round(v, 3) for v in right_standoff],
            left_target=[round(v, 3) for v in left_standoff],
        )

        ramp_ticks = max(
            1, round(gentle_descend_seconds / self.sim.cfg.dt)
        )
        ramp_tolerance_m = gentle_descend_m
        ramp_deviation_m = {"right": 0.0, "left": 0.0}
        ramp_aborted_tick = None
        # 2026-08-17: interpolate from the ARMS' ACTUAL achieved position,
        # not the intended right_standoff/left_standoff coordinates.
        # _servo_bimanual requires BOTH arms under a 2cm tolerance in the
        # same tick (harder to satisfy than single-arm reach()'s own
        # standoff, which this ramp was ported from) -- measured on a real
        # GPU run (outputs/task3_bimanual_carry/run2/log.txt):
        # reach_bimanual_standoff logged ok=False with each wrist still
        # 5.8-7.4cm from its intended standoff. Interpolating from that
        # unreached intended coordinate meant tick 1's "commanded" point
        # was already ~0.07m from the real wrist position, so the ramp's
        # own (deliberately tight, see reach()'s identical check) tolerance
        # tripped almost immediately (tick 75/600) on a residual standoff
        # gap, not on anything the ramp itself did. Starting from the real
        # position makes the deviation check measure only what the ramp
        # motion causes, which is what it was designed to catch.
        right_ramp_start = self.arms.ee_world_poses()[1][0]
        left_ramp_start = self.arms.ee_world_poses()[0][0]

        # 2026-08-17 (owner directive: close-on-contact): the video from
        # run2/run3 (outputs/task3_bimanual_carry/) showed the ramp
        # tripping its own deviation abort while both wrist cameras were
        # already right on the cup, not diverged into open space -- the
        # live hypothesis is that "deviation" was real PhysX contact
        # resistance, which a pure position-tracking check cannot tell
        # apart from a genuine miss. The gripper ContactSensor added this
        # session (see _gripper_contact_force_n) can. Reuses
        # grasp_transport.CONTACT_FORCE_MIN_N (0.01N), the same
        # already-proven "detected vs below_threshold" cut this project
        # uses for the single-arm cage-band verdict, rather than inventing
        # a new number. When the sensor is unavailable (wrong gripper
        # profile, missing prim -- guarded, never a false-safe zero), this
        # falls back to the exact pre-existing deviation-abort behaviour.
        from task3_autonomy.grasp_transport import CONTACT_FORCE_MIN_N

        contact_sensing_available = any(
            self._gripper_contact_sensors.get(s) is not None
            for s in ("right", "left")
        )
        contact_stopped = {"right": False, "left": False}
        contact_force_n = {"right": None, "left": None}
        right_hold_commanded = None
        left_hold_commanded = None
        for tick in range(1, ramp_ticks + 1):
            frac = tick / ramp_ticks
            if contact_stopped["right"]:
                right_commanded = right_hold_commanded
            else:
                right_commanded = tuple(
                    right_ramp_start[i]
                    + (right_grasp[i] - right_ramp_start[i]) * frac
                    for i in range(3)
                )
            if contact_stopped["left"]:
                left_commanded = left_hold_commanded
            else:
                left_commanded = tuple(
                    left_ramp_start[i]
                    + (left_grasp[i] - left_ramp_start[i]) * frac
                    for i in range(3)
                )
            self.arms.set_arm_target("right", right_commanded, top_down)
            self.arms.set_arm_target("left", left_commanded, top_down)
            self.arms.command()
            self._tick()
            right_now = self.arms.ee_world_poses()[1][0]
            left_now = self.arms.ee_world_poses()[0][0]
            ramp_deviation_m = {
                "right": math.dist(right_commanded, right_now),
                "left": math.dist(left_commanded, left_now),
            }
            if contact_sensing_available:
                contact_force_n["right"] = self._gripper_contact_force_n(
                    "right"
                )
                contact_force_n["left"] = self._gripper_contact_force_n(
                    "left"
                )
                if (
                    not contact_stopped["right"]
                    and contact_force_n["right"] is not None
                    and contact_force_n["right"] > CONTACT_FORCE_MIN_N
                ):
                    contact_stopped["right"] = True
                    right_hold_commanded = right_commanded
                if (
                    not contact_stopped["left"]
                    and contact_force_n["left"] is not None
                    and contact_force_n["left"] > CONTACT_FORCE_MIN_N
                ):
                    contact_stopped["left"] = True
                    left_hold_commanded = left_commanded
                if contact_stopped["right"] and contact_stopped["left"]:
                    break
                if (
                    not contact_stopped["right"]
                    and ramp_deviation_m["right"] > ramp_tolerance_m
                ) or (
                    not contact_stopped["left"]
                    and ramp_deviation_m["left"] > ramp_tolerance_m
                ):
                    ramp_aborted_tick = tick
                    break
            elif (
                ramp_deviation_m["right"] > ramp_tolerance_m
                or ramp_deviation_m["left"] > ramp_tolerance_m
            ):
                ramp_aborted_tick = tick
                break
        strict_reach = ramp_aborted_tick is None
        self._log_phase(
            "reach_bimanual_gentle_ramp",
            strict_reach,
            ramp_ticks=ramp_ticks,
            ramp_seconds=round(gentle_descend_seconds, 3),
            ramp_deviation_m={
                k: round(v, 4) for k, v in ramp_deviation_m.items()
            },
            ramp_tolerance_m=round(ramp_tolerance_m, 4),
            ramp_aborted_tick=ramp_aborted_tick,
            contact_sensing_available=contact_sensing_available,
            contact_stopped=dict(contact_stopped),
            contact_force_n=dict(contact_force_n),
        )

        right_err = self.arms.position_error("right", right_grasp)
        left_err = self.arms.position_error("left", left_grasp)
        contact_tolerance = p.get("contact_tolerance_m", 0.10)
        ok = strict_reach or (
            right_err <= contact_tolerance and left_err <= contact_tolerance
        )
        self._last_grasp_offset_bimanual = {
            "right": (
                right_grasp[0] - live_obj[0],
                right_grasp[1] - live_obj[1],
                right_grasp[2] - live_obj[2],
            ),
            "left": (
                left_grasp[0] - live_obj[0],
                left_grasp[1] - live_obj[1],
                left_grasp[2] - live_obj[2],
            ),
        }
        self._log_phase(
            "reach_bimanual_descend",
            ok,
            strict_reach=strict_reach,
            position_error_m={
                "right": round(right_err, 4),
                "left": round(left_err, 4),
            },
            contact_stopped=dict(contact_stopped),
        )
        return {
            "ok": ok,
            "strict_reach": strict_reach,
            "position_error_m": {"right": right_err, "left": left_err},
            "contact_stopped": dict(contact_stopped),
        }

    def grasp_bimanual(self, object_name, **p) -> dict:
        """Close both grippers together on an object reach_bimanual() has
        already positioned both arms around. Ported from verify_grasp_lift.
        py's proven bimanual close ramp (same session/evidence as
        reach_bimanual's docstring)."""
        self._active_object = object_name
        if self._object_has_fallen(object_name):
            self._log_phase(
                "grasp_bimanual_skipped_object_has_fallen",
                False,
                object=object_name,
            )
            return {
                "held": False,
                "scored": False,
                "reason": "object_has_fallen_off_its_surface",
            }

        from task3_autonomy.arms import GRIPPER_CLOSED_RAD, linear_ramp_target
        from task3_autonomy.grasp_transport import CONTACT_FORCE_MIN_N

        # RE-CENTER ON THE LIVE CUP POSE RIGHT BEFORE CLOSE -- the same
        # "LEVER 1" fix grasp() already applies for the single-arm path
        # (see its own comment), ported here after real GPU measurement
        # (scripts/task3/probe_bimanual_cup_contact_geometry.py,
        # plans/PROGRESS.md) showed WHY the bimanual close is only ~50%
        # reliable: reach_bimanual's own commanded descend targets are
        # computed from a live cup read taken BEFORE the descend servo
        # runs, but the descend motion itself -- two arms converging
        # simultaneously from opposite sides -- measurably nudges a light,
        # freely-movable cup before both pads finish converging. One
        # probed run moved the cup ~16cm in Y during reach_bimanual alone,
        # confirmed against a live tensor read (object_position) versus a
        # stale BBoxCache read frozen at the cup's ORIGINAL spawn pose (the
        # two disagreed by exactly that 16cm, and the arms' real achieved
        # pad positions landed near the STALE/spawn location, i.e. where
        # the cup no longer was). grasp_bimanual then closed on wherever
        # the descend target originally was, not on the cup's true current
        # position -- explaining the coin-flip outcome this session
        # measured (2 holds, 1 near-miss, 2 clean misses at otherwise
        # identical reach_bimanual telemetry). This does not change the
        # close ramp itself -- it just gives it a fresher target,
        # analogous to grasp()'s single-arm re-center.
        offsets = getattr(self, "_last_grasp_offset_bimanual", None)
        if offsets is not None:
            m = self._m
            top_down = m["_quaternion_from_rpy"](TOP_DOWN_ROLL_RAD, 0.0, 0.0)
            live_obj = self.object_position(object_name)
            right_recenter = (
                live_obj[0] + offsets["right"][0],
                live_obj[1] + offsets["right"][1],
                live_obj[2] + offsets["right"][2],
            )
            left_recenter = (
                live_obj[0] + offsets["left"][0],
                live_obj[1] + offsets["left"][1],
                live_obj[2] + offsets["left"][2],
            )
            recenter_ok = self._servo_bimanual(
                right_recenter, left_recenter, top_down, budget_s=4.0
            )
            self._log_phase(
                "grasp_bimanual_recenter",
                recenter_ok,
                right_target=[round(v, 3) for v in right_recenter],
                left_target=[round(v, 3) for v in left_recenter],
                live_obj=[round(v, 3) for v in live_obj],
            )

        # Gentle first: CLOSE_GRIPPER_STIFFNESS=60.0 is the proven value
        # that reliably catches the object (n>=4 across this session, both
        # standalone and inside the real pipeline). 2026-08-16 GPU evidence
        # (plans/PROGRESS.md) shows this is NOT enough to survive a carry
        # -- the close ramp's commanded target sits at GRIPPER_CLOSED_RAD
        # =0.0 for the back half of grasp_settle_seconds while measured
        # position stalls around 0.04-0.05 rad (the cup wall physically
        # blocking further travel), a persistent ~0.05 rad error the whole
        # hold, so torque = stiffness * error is the real lever on squeeze
        # force. At 60.0 that's ~3 N*m against a 50 N*m authored ceiling
        # (`authored_effort_limit_nm: 50.0`, measured this session) -- 6%
        # of available torque, plenty for a static 3s hold but not enough
        # friction margin for a real drive (two carry attempts at 0.3 and
        # 0.15 m/s, ruling out carry speed, both showed the object
        # separate from a `gripper_position_rad` that stayed essentially
        # flat -- not the jaws pushing open, but the object sliding out of
        # a fixed-but-shallow squeeze).
        #
        # A single higher-stiffness close was tried first and made things
        # WORSE, not better (GPU-measured): stiffness=300.0 from the start
        # collapsed both sides to ~0.0002 rad -- essentially fully closed
        # on nothing, a miss. The likely mechanism: with two independent
        # grippers closing on opposite sides of a light, freely-movable
        # cup, a too-aggressive close can shove the object away/off-axis
        # before both sides seal on it, before either pad has real contact
        # to resist against. So the FIRST close stays at the proven gentle
        # 60.0 -- unchanged from before -- and stiffness is only raised
        # AFTER a real contact is confirmed (`holding`), to firm up a grip
        # that's already correctly seated rather than to force one closed.
        for side in ("left", "right"):
            try:
                self.arms.set_gripper_stiffness(side, CLOSE_GRIPPER_STIFFNESS)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"WARN: bimanual close stiffness failed side={side!r} "
                    f"({exc!r})",
                    flush=True,
                )

        grasp_ramp_seconds = p.get("grasp_ramp_seconds", 1.0)
        grasp_settle_seconds = p.get("grasp_settle_seconds", 1.5)
        right_start = self.arms.gripper_position("right")
        left_start = self.arms.gripper_position("left")
        close_ticks = math.ceil(grasp_settle_seconds / self.sim.cfg.dt)
        ramp_ticks = math.ceil(grasp_ramp_seconds / self.sim.cfg.dt)

        # 2026-08-17 (owner directive: close-on-contact): stop advancing
        # EACH side's commanded close target independently, the instant
        # that side's real ContactSensor reading crosses
        # grasp_transport.CONTACT_FORCE_MIN_N -- instead of always ramping
        # both sides blindly to GRIPPER_CLOSED_RAD=0.0 regardless of where
        # (or whether) either pad actually met the cup. Holds the side's
        # last commanded position (still re-issued every tick so the
        # stiffness torque keeps acting, per this method's existing
        # gentle-then-firm mechanism) while the other side keeps closing if
        # it hasn't made contact yet. Falls back to the exact prior
        # behaviour (ramp straight to GRIPPER_CLOSED_RAD) when the sensor
        # is unavailable, same guard as reach_bimanual's identical check.
        contact_sensing_available = any(
            self._gripper_contact_sensors.get(s) is not None
            for s in ("right", "left")
        )
        close_contact_stopped = {"right": False, "left": False}
        close_hold_target = {"right": None, "left": None}
        for close_tick in range(close_ticks):
            if contact_sensing_available:
                if close_contact_stopped["right"]:
                    right_target = close_hold_target["right"]
                else:
                    right_target = linear_ramp_target(
                        right_start,
                        GRIPPER_CLOSED_RAD,
                        close_tick + 1,
                        ramp_ticks,
                    )
                if close_contact_stopped["left"]:
                    left_target = close_hold_target["left"]
                else:
                    left_target = linear_ramp_target(
                        left_start,
                        GRIPPER_CLOSED_RAD,
                        close_tick + 1,
                        ramp_ticks,
                    )
            else:
                right_target = linear_ramp_target(
                    right_start, GRIPPER_CLOSED_RAD, close_tick + 1, ramp_ticks
                )
                left_target = linear_ramp_target(
                    left_start, GRIPPER_CLOSED_RAD, close_tick + 1, ramp_ticks
                )
            self.arms.set_gripper("right", right_target)
            self.arms.set_gripper("left", left_target)
            self.arms.command()
            self._tick()
            if contact_sensing_available:
                if not close_contact_stopped["right"]:
                    force = self._gripper_contact_force_n("right")
                    if force is not None and force > CONTACT_FORCE_MIN_N:
                        close_contact_stopped["right"] = True
                        close_hold_target["right"] = right_target
                if not close_contact_stopped["left"]:
                    force = self._gripper_contact_force_n("left")
                    if force is not None and force > CONTACT_FORCE_MIN_N:
                        close_contact_stopped["left"] = True
                        close_hold_target["left"] = left_target

        gripper_holds_object = self._m["gripper_holds_object"]
        right_pos = self.arms.gripper_position("right")
        left_pos = self.arms.gripper_position("left")
        holding = gripper_holds_object(
            right_pos,
            max_position_rad=self.arms._gripper_position_upper_limit(
                "right"
            ),
        ) and gripper_holds_object(
            left_pos,
            max_position_rad=self.arms._gripper_position_upper_limit("left"),
        )

        # FIRM UP a confirmed hold before the carry gets a chance to test
        # it. Raising stiffness now increases torque = stiffness * error
        # at whatever position the object already settled the pads at --
        # it does not command a NEW, deeper target, so it cannot push the
        # already-seated grip past the object the way the too-aggressive
        # single-stage close did above.
        if holding:
            firm_stiffness = p.get("firm_close_stiffness", 300.0)
            firm_seconds = p.get("firm_close_seconds", 1.0)
            for side in ("left", "right"):
                try:
                    self.arms.set_gripper_stiffness(side, firm_stiffness)
                except Exception as exc:  # noqa: BLE001
                    print(
                        f"WARN: bimanual firm-close stiffness failed "
                        f"side={side!r} ({exc!r})",
                        flush=True,
                    )
            for _ in range(math.ceil(firm_seconds / self.sim.cfg.dt)):
                self.arms.command()
                self._tick()
            right_pos = self.arms.gripper_position("right")
            left_pos = self.arms.gripper_position("left")
            holding = gripper_holds_object(
                right_pos,
                max_position_rad=self.arms._gripper_position_upper_limit(
                    "right"
                ),
            ) and gripper_holds_object(
                left_pos,
                max_position_rad=self.arms._gripper_position_upper_limit(
                    "left"
                ),
            )

        if holding:
            self._held = object_name
            self._held_side = None
            self._held_sides = ("left", "right")
            self._held_gripper_rad_bimanual = {
                "left": left_pos,
                "right": right_pos,
            }
        else:
            self._held = None
            self._held_sides = None
            self._held_gripper_rad_bimanual = None
        self._log_phase(
            "grasp_bimanual_close",
            holding,
            gripper_position_rad={
                "right": round(right_pos, 4),
                "left": round(left_pos, 4),
            },
            close_contact_stopped=dict(close_contact_stopped),
        )
        return {
            "held": holding,
            "scored": holding,
            "gripper_position_rad": {"right": right_pos, "left": left_pos},
            "close_contact_stopped": dict(close_contact_stopped),
        }

    def lift_bimanual(self, dz, **p) -> dict:
        """Raise both held arms (and the spine) together. Ported from
        verify_grasp_lift.py's proven bimanual lift ramp (same session/
        evidence as reach_bimanual's docstring: cup rose 0.316m, held 3s)."""
        right_pose = self.arms.ee_world_poses()[1]
        left_pose = self.arms.ee_world_poses()[0]
        timeout_s = p.get("timeout_s", 6.0)
        ramp_seconds = p.get("ramp_seconds", 3.0)
        spine_assist_m = p.get("spine_assist_m", 0.12)
        start_spine = self.arms.spine
        lift_ticks = math.ceil(timeout_s / self.sim.cfg.dt)
        ramp_ticks = math.ceil(ramp_seconds / self.sim.cfg.dt)
        for lift_tick in range(lift_ticks):
            alpha = min(1.0, (lift_tick + 1) / ramp_ticks)
            self.arms.spine = start_spine + spine_assist_m * alpha
            self.arms.set_arm_target(
                "right",
                (
                    right_pose[0][0],
                    right_pose[0][1],
                    right_pose[0][2] + dz * alpha,
                ),
                right_pose[1],
            )
            self.arms.set_arm_target(
                "left",
                (
                    left_pose[0][0],
                    left_pose[0][1],
                    left_pose[0][2] + dz * alpha,
                ),
                left_pose[1],
            )
            self.arms.command()
            self._tick()
        object_name = self._held
        self._log_phase("lift_bimanual", True, object=object_name)
        return {"ok": True}

    def lift(self, side, dz, **p) -> dict:
        object_name = self._held
        # Single read, reused below -- object_position() is a live sensor
        # query (a per-tick-advancing sequence in at least one test double),
        # so reading it twice for "the same instant" is not just wasteful,
        # it can silently observe two different values.
        object_pos_before = (
            self.object_position(object_name) if object_name else None
        )
        z_before = object_pos_before[2] if object_pos_before else None
        # P5.1: baseline for hold()'s three_predicate_hold -- see
        # _pre_lift_baseline's own comment in __init__/reset().
        if object_name is not None:
            self._pre_lift_baseline[object_name] = (
                self.arms.ee_world_poses()[0 if side == "left" else 1][0],
                object_pos_before,
            )
        # REV13 T4-followup-2 (plans/SYNC.md 2026-08-07): 9/9 episodes with
        # a telemetrically "sustained" close still failed to lift -- the
        # close-loop mechanism was never the bottleneck. Per-tick tracking
        # here, mirroring T2's close_telemetry, so a slip during the lift
        # motion itself is observable instead of inferred from a single
        # before/after rise measurement.
        offset = (
            self._last_grasp_offset.get(object_name, (0.0, 0.0, 0.0))
            if object_name
            else (0.0, 0.0, 0.0)
        )
        max_dist = p.get(
            "max_held_object_distance_m",
            config.THRESHOLDS.GRASP_HELD_MAX_DIST_M,
        )
        vgl = self._m["vgl"]
        lift_telemetry: list[dict] = []

        def _on_tick(tick: int) -> None:
            if object_name is None:
                return
            object_pos = self.object_position(object_name)
            ee_pos = self.arms.ee_world_poses()[0 if side == "left" else 1][0]
            grasp_frame_point = (
                object_pos[0] + offset[0],
                object_pos[1] + offset[1],
                object_pos[2] + offset[2],
            )
            follows = vgl.object_follows_end_effector(
                grasp_frame_point, ee_pos, max_distance_m=max_dist
            )
            lift_telemetry.append(
                {
                    "tick": tick,
                    "object_z": round(object_pos[2], 5),
                    "ee_z": round(ee_pos[2], 5),
                    "object_follows_ee": bool(follows),
                }
            )

        lift_ok = self.arms.lift(
            side,
            dz,
            step=self._tick,
            dt=self.sim.cfg.dt,
            timeout_s=p.get("timeout_s", 6.0),
            position_tolerance_m=p.get(
                "position_tolerance_m",
                config.THRESHOLDS.lift_position_tolerance_m,
            ),
            spine_assist_m=p.get("spine_assist_m", 0.12),
            on_tick=_on_tick,
        )
        z_after = (
            self.object_position(object_name)[2] if object_name else z_before
        )
        rise = (z_after - z_before) if object_name else 0.0
        first_slip_tick = next(
            (
                row["tick"]
                for row in lift_telemetry
                if not row["object_follows_ee"]
            ),
            None,
        )
        # A LIFT THAT LEAVES THE OBJECT BEHIND IS NOT A LIFT.
        #
        # `lift_ok` above is `arms.lift()`'s return: whether the ARM reached
        # its commanded height. It says nothing about the object, so the
        # phase reported success while the payload never moved.
        #
        # Measured, allobj_7, spoon2: phase lift ok=True with
        # `object_rise_m -0.0` and `lift_first_slip_tick 47`. The EE rose
        # from z 0.8733 to 0.9403 -- a real 6.7 cm arm motion -- while the
        # object sat at 0.7556 before and after. The arm lifted; the spoon
        # did not.
        #
        # `scripts/task3/run_stage1_setup.py` has always scored its own lift
        # the honest way (`lift_ok = obj_rise >= MIN_LIFT_M`); this adopts
        # that rule rather than inventing a new one. The arm's own verdict
        # is kept alongside as `arm_lift_ok`, so a failure still says WHICH
        # half failed: arm short of target, or object not coming with it.
        object_lifted = bool(
            object_name is not None
            and rise >= config.THRESHOLDS.min_lift_m
        )
        # No held object means there is nothing to weigh the verdict
        # against, so fall back to exactly the previous behaviour.
        honest_lift_ok = (
            bool(lift_ok) if object_name is None else object_lifted
        )
        self._log_phase(
            "lift",
            honest_lift_ok,
            object_rise_m=round(rise, 4),
            arm_lift_ok=bool(lift_ok),
            min_lift_m=config.THRESHOLDS.min_lift_m,
            lift_first_slip_tick=first_slip_tick,
        )
        return {
            "object_rise_m": round(rise, 4),
            "ik_ok": bool(honest_lift_ok),
            "arm_lift_ok": bool(lift_ok),
            "lift_telemetry": lift_telemetry,
            "lift_first_slip_tick": first_slip_tick,
        }

    def hold(self, seconds, **p) -> dict:
        vgl = self._m["vgl"]
        object_name = self._held
        if object_name is None:
            return {
                "z_drop_m": 1.0,
                "held_seconds": 0.0,
                "required_seconds": seconds,
            }
        start_z = self.object_position(object_name)[2]
        side = p.get("side", "right")
        hold_pose = self.arms.ee_world_poses()[0 if side == "left" else 1]
        max_dist = p.get(
            "max_held_object_distance_m",
            config.THRESHOLDS.GRASP_HELD_MAX_DIST_M,
        )
        needed_ticks = int(seconds / self.sim.cfg.dt)
        held_ticks = 0
        min_z = start_z
        # Same M0 fix as grasp(): compare the wrist against the object's
        # current pose shifted by the standoff recorded at descend/recenter
        # time, not the raw object origin (which always sits the standoff
        # distance below the wrist even for a real hold).
        offset = self._last_grasp_offset.get(object_name, (0.0, 0.0, 0.0))
        for _ in range(needed_ticks + int(2.0 / self.sim.cfg.dt)):
            self.arms.set_arm_target(side, hold_pose[0], hold_pose[1])
            self.arms.command()
            self._tick()
            object_pos = self.object_position(object_name)
            min_z = min(min_z, object_pos[2])
            grasp_frame_point = (
                object_pos[0] + offset[0],
                object_pos[1] + offset[1],
                object_pos[2] + offset[2],
            )
            follows = vgl.object_follows_end_effector(
                grasp_frame_point, hold_pose[0], max_distance_m=max_dist
            )
            if follows:
                held_ticks += 1
                if held_ticks >= needed_ticks:
                    break
            else:
                held_ticks = 0
        drop = start_z - min_z

        # P5.1 (2026-08-11): the six-key evidence outcomes.classify_hold
        # needs to run verification.three_predicate_hold instead of the old
        # z_drop_m/held_seconds-only check (task3_pipeline/outcomes.py,
        # merged from sprint/audit -- "did not fall and time elapsed" also
        # passes for an object resting on a surface the gripper never
        # closed on).
        #
        # hold()'s own loop commands a STATIONARY target throughout (see
        # `self.arms.set_arm_target(side, hold_pose[0], hold_pose[1])`
        # above, same target every tick) -- a rise/ee-delta measured only
        # within hold()'s own window would be ~0 even for a genuinely
        # successful hold, which would make three_predicate_hold's
        # follow-delta and lift predicates fail SUCCESSFUL holds, not just
        # bad ones. So the start references come from lift()'s own
        # pre-lift baseline (_pre_lift_baseline, captured before the
        # object left the table) -- the evidence spans the real grasp+lift
        # motion, evaluated once hold() confirms it stuck. Falls back to
        # hold()'s own start state when lift() was never called for this
        # object (e.g. a direct hold() test), so this never crashes on a
        # missing baseline -- it just measures a near-zero delta, which
        # three_predicate_hold will correctly call not-yet-proven rather
        # than silently skip.
        ee_pos_end = self.arms.ee_world_poses()[0 if side == "left" else 1][0]
        object_pos_end = self.object_position(object_name)
        baseline = self._pre_lift_baseline.get(object_name)
        if baseline is not None:
            ee_pos_start, object_pos_start = baseline
        else:
            ee_pos_start = hold_pose[0]
            object_pos_start = (object_pos_end[0], object_pos_end[1], start_z)

        self._log_phase(
            "hold", held_ticks >= needed_ticks, held_ticks=held_ticks
        )
        return {
            "z_drop_m": round(drop, 4),
            "held_seconds": round(held_ticks * self.sim.cfg.dt, 3),
            "required_seconds": seconds,
            "gripper_position_rad": round(self.arms.gripper_position(side), 4),
            "ee_pos_start": tuple(round(v, 4) for v in ee_pos_start),
            "ee_pos_end": tuple(round(v, 4) for v in ee_pos_end),
            "object_pos_start": tuple(round(v, 4) for v in object_pos_start),
            "object_pos_end": tuple(round(v, 4) for v in object_pos_end),
            "object_rise_m": round(object_pos_end[2] - object_pos_start[2], 4),
        }

    def place(self, side, world_pose, **p) -> dict:
        m = self._m
        # Carry the object out on the wrist pose it was GRASPED with, not a
        # rebuilt straight-down one. While every approach was top-down these
        # were the same quaternion; once reach() can command a tilted
        # approach they are not, and forcing vertical here rotates whatever
        # is in the jaws by the tilt angle on the way to the table -- a plate
        # held at 45 degrees is levered out of the pads before it is ever
        # released. Falls back to the old constant when nothing has been
        # grasped this episode, which is byte-identical.
        top_down = self._last_grasp_quat.get(self._active_object) or m[
            "_quaternion_from_rpy"
        ](math.pi, 0.0, 0.0)
        ok = self.arms.place(
            side,
            world_pose,
            top_down,
            step=self._tick,
            dt=self.sim.cfg.dt,
            timeout_s=p.get("timeout_s", 8.0),
        )
        release_ok = self.arms.release(
            side, step=self._tick, dt=self.sim.cfg.dt, timeout_s=2.0
        )
        if self._held is not None:
            self._held = None
        self._log_phase(
            "place",
            ok and release_ok,
            target=[round(v, 3) for v in world_pose],
        )
        return {"scored": bool(ok and release_ok)}

    def _ik_feasible_sides(
        self, target_xyz: tuple[float, float, float]
    ) -> tuple[bool, bool] | None:
        """R9 T3: a real IK feasibility query for `target_xyz`, both arms,
        against the robot's ACTUAL current base pose -- the same solver
        `DualArmController.command()` calls every production tick
        (mirrors `task3_autonomy.perception_targets.screen_candidate_ik`'s
        own pattern; not a re-derived approximation). Returns
        `(left_feasible, right_feasible)`, or `None` if a live query is
        not possible (`self.arms` unset -- pre-reset/mock/off-GPU worlds)
        so callers fall back to distance instead of crashing."""
        if self.arms is None or self.robot is None:
            return None
        # Local import, not self._m: `teleop_targets` is CPU-pure (no Isaac
        # at module scope, verified: only `math`/`teleop_commands`), so
        # this stays callable from CPU tests that build `world.arms`
        # directly without ever calling reset() (which is what populates
        # `self._m`).
        from teleop_targets import _quaternion_from_rpy

        top_down = _quaternion_from_rpy(TOP_DOWN_ROLL_RAD, 0.0, 0.0)
        root_position, root_orientation = self.arms._root_pose(self.robot)
        result = self.arms._ik.solve(
            target_xyz,
            target_xyz,
            top_down,
            top_down,
            spine_position=self.arms.spine,
            base_position=root_position,
            base_orientation_wxyz=root_orientation,
        )
        return bool(result.left_succeeded), bool(result.right_succeeded)

    def _select_arm_side(self, object_name: str) -> str:
        """R9 T3 (plans/LOOP_PROMPT_VM_A_REV9.md): pick the arm side by a
        REAL IK feasibility query (`_ik_feasible_sides`) instead of
        Euclidean distance -- the R7 sweep disproved distance as the
        metric (0% feasible at `dx<=+0.10`, well inside the ~0.855m reach
        envelope; VM B's own live bowl2 attempt confirmed it: same
        ER-derived pose, RIGHT arm 0/1600/1200/1200 real IK-ok ticks,
        LEFT 191/1200/1200, `object_follows_ee: true`).

        Distance (`_arm_base_relative`, the old P5 metric) survives only
        as a cheap tie-break when the feasibility query agrees on both
        sides (both feasible or neither) or is unavailable. Falls back to
        "right" (the unchanged original default) when body lookup fails
        entirely -- preserved deliberately: off-GPU/mock worlds and
        `test_pipeline.py`'s
        `test_select_arm_side_defaults_to_right_when_body_lookup_misses`
        depend on this exact fallback, and a selection miss must never
        abort a stage."""
        obj_xyz = self.object_position(object_name)
        right_rel = self._arm_base_relative("right", obj_xyz)
        left_rel = self._arm_base_relative("left", obj_xyz)
        if right_rel is None or left_rel is None:
            return "right"

        feasible = self._ik_feasible_sides(obj_xyz)
        if feasible is not None:
            left_ok, right_ok = feasible
            if left_ok != right_ok:
                return "left" if left_ok else "right"

        return "left" if left_rel[1] < right_rel[1] else "right"

    def carry_object_to(self, object_name, x, y, z=None, **p) -> dict:
        """Drive the base to (x, y) while re-issuing the held relative arm
        target each tick (so a genuinely grasped object travels with the
        gripper), then release. This is a controlled carry/place, not a
        physics exploit -- it follows a real, verified grasp (see grasp())."""
        side = p.get("side")
        if side is None:
            # THE ARM THAT IS ACTUALLY HOLDING IT WINS.
            #
            # This defaulted to "right" whenever select_nearer_arm_side was
            # off, while every grasp in this pipeline is made with the LEFT
            # arm -- so the carry re-issued the wrong arm's held pose, and
            # the left arm holding the object was never commanded at all.
            #
            # Measured, allobj_1: close ok=True on the left at
            # gripper_position_rad 0.4598 with object_follows_ee True, then
            # the carry reported obj_to_ee 0.3572 and grip 0.0004. Both
            # numbers are the RIGHT arm's: left_ee to object was 0.104 m,
            # right_ee to object 0.365 m, and 0.0004 is the empty right
            # gripper. The object had not moved -- it was still on the
            # counter at z=0.7556, 1.8 cm from where it was grasped. The
            # carry was simply looking at, and driving, the wrong arm.
            if self._held == object_name and self._held_side:
                side = self._held_side
            elif self.select_nearer_arm_side:
                side = self._select_arm_side(object_name)
            else:
                side = "right"
        method = p.get("method", "grasp_place")

        # C0 (plan CUROBO_PIVOT_PLAN_2026-07-28.md sec 5 / handoff sec 48):
        # score_stage4_cleanup only checks the object's final bounds/height,
        # not that anything ever held it -- config.CLEANUP_GRID's
        # "base_carry"/"controlled_slide" methods are legal, ungated paths
        # that never needed grasp() to have succeeded. Route them to a real
        # contact-push instead of requiring self._held. "grasp_place" (the
        # default) is unchanged below: it still requires a real, verified
        # hold.
        # A VERIFIED HOLD OUTRANKS THE REQUESTED METHOD.
        #
        # This check used to come after the method branch, so a "base_carry"
        # request pushed the object even when the gripper was already
        # holding it -- throwing away the one thing the whole pipeline
        # exists to achieve. Measured, run 21, spoon2: `close ok=True` under
        # the strict verdict (`gripper_hold_predicate True`,
        # `object_follows_ee True`, `gripper_position_rad 0.0881`, pad
        # midpoint 0.0657 m from the grasp point) followed immediately by
        # `push_object_to_retract_tuck` and a full push sequence.
        #
        # Fixed here rather than by reordering CLEANUP_GRID, because the
        # grid's "base_carry"-first order is CORRECT for the regime it was
        # chosen in: `test_cleanup_retry_grid_real_ordering_covers_both_
        # stances_within_budget` records that Stage 4's scored command runs
        # with `--skip-grasp`, under which `grasp_place` is a guaranteed
        # no-op that would waste one of only four retry slots. Both regimes
        # are now served: with nothing held the grid behaves exactly as
        # before, and with something held we carry it instead of shoving it.
        if self._held == object_name:
            print(
                "CARRY_HONOURING_VERIFIED_HOLD "
                f"object={object_name!r} requested_method={method!r} "
                "note='holding it, so carrying rather than pushing'",
                flush=True,
            )
        elif method in ("base_carry", "controlled_slide"):
            # P5 finding: an explicit side= kwarg (now a real call pattern
            # since side selection can be overridden) was still sitting in
            # **p here, colliding with the positional `side` just resolved
            # above -- TypeError: multiple values for argument 'side'. Never
            # triggered before this task because no caller ever passed
            # side= explicitly; a real bug, not a hypothetical one.
            push_kwargs = {k: v for k, v in p.items() if k != "side"}
            return self._push_object_to(
                side, object_name, x, y, z, **push_kwargs
            )

        if self._held != object_name:
            # No verified hold -- nothing to honestly carry.
            return {"scored": False, "reason": "no verified hold on object"}

        # A grasp_bimanual() hold means BOTH arms are gripping the object --
        # re-issuing only `side`'s relative pose every tick (the single-arm
        # path below) would leave the other arm's hand frozen in world
        # space while the base drives out from under it, tearing the object
        # out exactly the way the single-arm carry bug this function's own
        # history describes did before it re-commanded the gripper. Carry
        # every held side, not just one.
        held_sides = self._held_sides or (side,)
        held_poses_relative = {
            s: self.arms.arm_pose_relative(s) for s in held_sides
        }
        held_pose_relative = held_poses_relative[
            side if side in held_poses_relative else held_sides[0]
        ]
        self._base_hold_anchor = None
        max_linear = p.get("max_linear_mps", 0.3)
        budget_s = p.get("budget_s", 40.0)
        # min_creep_mps: this call never had it, and it is the third place in
        # this codebase to be caught by the same defect. `position_kp *
        # distance` decays toward zero as the target nears, while the wheel
        # drives have a ~2 s velocity-tracking lag (DRIVE_DAMPING=500), so a
        # commanded speed that shrinks faster than the wheels can track never
        # lets real motion start -- and the ProgressWatchdog then reports a
        # stall the base never had. `navigate_to` was fixed for exactly this
        # in 957d886 and `RotateTo` earlier today (`min_creep_radps`).
        #
        # Measured here, run 22, immediately after
        # CARRY_HONOURING_VERIFIED_HOLD on a genuinely held spoon2:
        # `carry_object_to ok=False stalled=True`, target (-1.6, 1.0), object
        # (-4.1862,-1.6238,0.7617) -> (-4.186,-1.624,0.762). The object did
        # not move AT ALL: the drive never began, so this was not a grasp
        # slipping en route.
        #
        # A carry needs this more than a free drive does, not less -- it
        # starts from rest with an extended arm and an object's mass on the
        # end of it. 0.08 m/s is the value already proven at `reach()`'s own
        # stance approach, not a fresh guess.
        min_creep = p.get("min_creep_mps", 0.08)
        skill = self._m["NavigateTo"](
            (x, y),
            None,
            max_linear_mps=max_linear,
            min_creep_mps=min_creep,
        )
        # (tick, gripper_rad, object-to-ee distance) sampled through the
        # drive -- see the sampling block in the loop below for why.
        carry_trace: list[tuple[int, float, float]] = []
        watchdog = ProgressWatchdog()
        stalled = False
        for _ in range(int(budget_s / self.sim.cfg.dt)):
            pose = self.adapter.pose()
            if watchdog.sample(self._tick_count, pose.x, pose.y):
                stalled = True
                self.adapter.apply_twist(0.0, 0.0)
                self._tick()
                break
            vx, vy, done = skill.compute(pose)
            for held_side in held_sides:
                self.arms.set_arm_target_relative(
                    held_side,
                    held_poses_relative[held_side].position,
                    held_poses_relative[held_side].orientation_wxyz,
                )
            # KEEP COMMANDING THE GRIP. `set_arm_target_relative` rebuilds
            # the tracker's target and takes the gripper with it, so the
            # held width is discarded on the first tick of the carry and the
            # jaws spring open.
            #
            # Measured, allobj_2, once the carry was finally driving the arm
            # that actually holds the object: the trace opens at
            # (7750, 0.4601, 0.1182) -- the hold intact -- and by the very
            # next sample reads (8000, 1.0028, 0.1041): 1.0028 rad is past
            # the gripper's own 0.9 open position, and the object simply
            # stays where it was. Nothing dropped it; the hand let go.
            if self._held_sides and self._held_gripper_rad_bimanual:
                for held_side in held_sides:
                    self.arms.set_gripper(
                        held_side,
                        self._held_gripper_rad_bimanual[held_side],
                    )
            elif self._held_gripper_rad is not None:
                self.arms.set_gripper(side, self._held_gripper_rad)
            # ...AND ACTUALLY SEND IT. `set_arm_target_relative` and
            # `set_gripper` only mutate the tracker's targets; `command()`
            # is the call that solves IK and writes
            # `set_joint_position_target`. Without it this loop wrote
            # nothing to the robot for the whole drive: the arm stayed
            # frozen in JOINT space while the base drove out from under it,
            # and the held-width re-command above was a no-op.
            #
            # Measured, allobj_3, spoon2 -- a genuinely held object
            # (`gripper_position_rad 0.4598`, `object_follows_ee True`,
            # latched at `commanded_rad=0.4598`): the trace opens intact at
            # (7750, 0.4601, 0.1182) and reads (8000, 1.0028, 0.1041) by
            # the next sample. 1.0028 is the gripper's `gripper_max` limit,
            # not a slipping grasp -- with the arm rigid and the base
            # driving away, the object levered the jaws open to their stop.
            # The cup showed the same signature from the first sample
            # (24000, 1.0046, 0.0584) with the object left behind as
            # obj_to_ee grew 0.058 -> 0.474 m.
            #
            # Every other motion path in this file already calls
            # `command()` inside its tick loop; this loop was the only one
            # that did not. Wrapped because an IK failure mid-drive should
            # cost tracking on that tick, not abort a carry that is
            # otherwise going fine.
            try:
                self.arms.command()
            except Exception as exc:  # noqa: BLE001
                print(f"CARRY_ARM_COMMAND_FAILED {exc!r}", flush=True)
            # Does the GRIP survive the drive? Run 23 held spoon2
            # (`gripper_position_rad 0.4781`, `object_follows_ee True`),
            # drove the base 1.7 m, and left the object exactly where it was
            # grasped -- and nothing recorded when or how it was lost. This
            # loop re-commands the arm every tick and never the gripper, so
            # the two candidate explanations are "the grip opened" and "the
            # arm's relative-pose correction pulled the object out", and
            # they need opposite fixes.
            #
            # Sampled, not per-tick: a 40 s carry is 8000 ticks and a log
            # line each would drown the run.
            if carry_trace is not None and len(carry_trace) < 40:
                if self._tick_count % 250 == 0:
                    carry_trace.append(
                        (
                            self._tick_count,
                            round(self.arms.gripper_position(side), 4),
                            round(
                                math.dist(
                                    self.object_position(object_name),
                                    self.arms.ee_world_poses()[
                                        0 if side == "left" else 1
                                    ][0],
                                ),
                                4,
                            ),
                        )
                    )
            if done:
                self.adapter.apply_twist(0.0, 0.0)
                self._tick()
                break
            self.adapter.apply_twist(vx, vy)
            self._tick()

        release_ok = all(
            self.arms.release(
                held_side, step=self._tick, dt=self.sim.cfg.dt, timeout_s=2.0
            )
            for held_side in held_sides
        )
        self._held = None
        self._held_sides = None
        self._held_gripper_rad_bimanual = None
        for _ in range(round(1.0 / self.sim.cfg.dt)):
            self._tick()
        final = self.object_position(object_name)
        target_z = z if z is not None else config.SINK_TABLETOP_Z
        scored = (
            not stalled
            and release_ok
            and math.hypot(final[0] - x, final[1] - y) <= 0.5
            and final[2] >= target_z - 0.05
        )
        self._log_phase(
            "carry_object_to",
            scored,
            target=[round(x, 3), round(y, 3)],
            final=[round(v, 3) for v in final],
            stalled=stalled,
            # (tick, gripper_rad, object_to_ee_m) through the drive. Reading
            # it: a gripper_rad collapsing toward 0 means the grip opened; a
            # steady gripper_rad with a growing distance means the object was
            # pulled or shaken out of jaws that stayed shut.
            carry_trace=carry_trace,
        )
        result = {"scored": bool(scored)}
        if stalled:
            result["stalled"] = True
            result["pose_trace"] = watchdog.pose_trace
        return result

    def _ensure_perception_push_targets(self) -> None:
        """Q3 (SYNC 22-24): populate self._perception_push_target_cache
        ONCE per episode via a single batched ER call, for
        PUSH_PERCEPTION_OBJECTS only. Mirrors scripts/task3/
        probe_perception_targets.py's proven GATE N2/N4 pattern (same
        SceneCamera, same call_er, same build_ranked_candidates/
        best_feasible), just called from inside a live push attempt
        instead of a standalone probe script. Never raises and never
        aborts the stage -- any failure (no API key, ER call error, self-
        check miss, no feasible candidate) just leaves the cache empty
        and _push_object_to() falls back to its existing geometry,
        exactly like every other perception seam in this project.
        """
        self._perception_push_attempted = True
        objects = [
            o
            for o in PUSH_PERCEPTION_OBJECTS
            if self.stage4_objects is None or o in self.stage4_objects
        ]
        if not objects:
            return
        try:
            from probe_gemini_er_vs_ground_truth import (
                SceneCamera,
                _api_keys,
                call_er,
            )

            from task3_autonomy.perception_targets import (
                best_feasible,
                build_ranked_candidates,
                er_grasp_targets_batched,
            )

            # Matches probe_perception_targets.py's own resilience: no ER
            # keys does NOT mean no perception -- build_ranked_candidates
            # always also tries the BBOX geometric fallback (the object's
            # own real USD bounding box, no API/network dependency, still
            # screened through the real IK solver), "the night cannot
            # stall" per that module's own docstring. Bailing out entirely
            # here would silently make this flag a no-op whenever a key is
            # missing/rate-limited instead of degrading gracefully.
            try:
                api_keys = _api_keys()
            except SystemExit as exc:
                print(
                    f"PUSH_PERCEPTION no ER keys ({exc}) -- BBOX-only",
                    flush=True,
                )
                api_keys = []

            cam = SceneCamera(resolution=(640, 360))
            y0, x0, _ = cam.project(CAMERA_LOOK_AT)
            selfcheck_err = math.hypot(y0 - 500.0, x0 - 500.0)
            if selfcheck_err > 60.0:
                print(
                    "PUSH_PERCEPTION selfcheck_err="
                    f"{selfcheck_err:.1f} > 60.0 -- skipping, not trusting "
                    "the camera projection",
                    flush=True,
                )
                return

            seen_frames = self._frames_written
            ticks = 0
            while self._frames_written <= seen_frames and ticks < 300:
                self._tick()
                ticks += 1
            frame_path = (
                self.frames_dir / f"rgb_{self._frames_written - 1:04d}.png"
            )

            er_by_object: dict[str, dict] = {}
            if api_keys:
                try:
                    er_by_object, latency_s, key_suffix = (
                        er_grasp_targets_batched(
                            frame_path, objects, api_keys, call_er
                        )
                    )
                    print(
                        f"PUSH_PERCEPTION_ER_RESPONSE "
                        f"latency_s={latency_s:.2f} "
                        f"key=...{key_suffix} "
                        f"objects={list(er_by_object)}",
                        flush=True,
                    )
                except Exception as exc:  # noqa: BLE001 -- fall to BBOX-only
                    print(
                        f"PUSH_PERCEPTION_ER_CALL_FAILED error={exc}",
                        flush=True,
                    )

            for name in objects:
                live_obj = self.object_position(name)
                _, _, gt_depth = cam.project(live_obj)
                candidates = build_ranked_candidates(
                    self, name, er_by_object.get(name), cam, gt_depth
                )
                winner = best_feasible(candidates)
                if winner is not None:
                    self._perception_push_target_cache[name] = winner.xyz
                    print(
                        f"PUSH_PERCEPTION_TARGET object={name} "
                        f"source={winner.source} xyz={winner.xyz}",
                        flush=True,
                    )
        except Exception as exc:  # noqa: BLE001 -- diagnostic-only seam,
            # never let a perception failure abort a live scoring episode.
            print(f"PUSH_PERCEPTION_FAILED error={exc}", flush=True)

    def _reach_limit_exceeded(self, side, target) -> bool:
        """Q2/Q4 shared gate: True if `target` is already past
        PUSH_APPROACH_REACH_LIMIT_M for `side`'s arm base. Reuses the same
        `_arm_base_relative()` telemetry helper the C2.5 logging already
        calls -- returns False (never blocks) off-GPU or when the body
        lookup fails, exactly like the rest of that telemetry.

        T5 (LOOP_PROMPT_VM_A_REV4.md): `self.reach_gate_enabled` (default
        True, unchanged behavior) lets this A/B test be run without it --
        Q2/Q4's gates are correct engineering that may also be refusing
        the exact out-of-reach attempts that produced the project's only
        point (handoff sec 105). Behind a flag so the comparison is a real
        code-path toggle, not a guess.
        """
        if not self.reach_gate_enabled:
            return False
        rel = self._arm_base_relative(side, target)
        return rel is not None and rel[1] > PUSH_APPROACH_REACH_LIMIT_M

    def _already_scored_push_result(self, object_name: str) -> dict | None:
        """T3 (LOOP_PROMPT_VM_A_REV4.md): if ``object_name`` already
        satisfies the scorer's own predicate, freeze here and report
        success without touching it again. Handoff sec 105's audit found
        the winning run's cup survived THREE further failed push attempts
        after it had already scored at tick 35794 -- any of which could
        just as easily have knocked it back out of the sink rect or off
        the counter. Returns the success dict to return immediately, or
        None if the object does not yet score.
        """
        pos = self.object_position(object_name)
        if not config.scores_in_sink(*pos):
            return None
        self._log_phase(
            "push_already_scored", True, obj=[round(v, 3) for v in pos]
        )
        return {"scored": True, "already_scored": True}

    def _push_object_to(self, side, object_name, x, y, z, **p) -> dict:
        """C0's real physics push (no grasp): open the gripper, position it
        just behind the object relative to the push direction, then drive
        the base toward (x, y) while holding that contact pose RIGID
        relative to the base -- the same mechanism carry_object_to already
        uses to carry a genuinely held object (a fixed base-relative arm
        target, reissued every tick via set_arm_target_relative), just with
        a pushing contact pose instead of a holding one. The object moves
        because the base drives into it with the gripper in the way, via
        real PhysX contact -- no kinematic attach, no teleport, matching
        this project's standing "standard physics only" rule.

        "base_carry" and "controlled_slide" (config.CLEANUP_GRID) both
        route here today -- they are implemented identically for now (one
        real push mechanism), not two. A future session could give
        "base_carry" a genuinely different (arm-free, base-body) mechanism;
        this session did not have time to build and verify a second one
        without risking an unpredictable base-vs-furniture collision, and
        C0 only requires ONE working path to score.
        """
        # REVIEW #10 (handoff sec 86): reach() -- now removed from
        # plan_stage4's push path -- was the only place that set
        # self._active_object, which _log_phase reads to put a live
        # obj: [...] coordinate into every phase log line. Without this,
        # object-position telemetry goes dark for every push attempt.
        self._active_object = object_name
        already = self._already_scored_push_result(object_name)
        if already is not None:
            return already
        # Q3: one ER call per episode, not one per retry attempt.
        if (
            self.push_perception_targets
            and not self._perception_push_attempted
        ):
            self._ensure_perception_push_targets()
        # handoff sec 89: a 25 deg ROLL tilt off pure top-down (changes
        # pointing direction) was GPU-verified and REFUTED -- push_contact
        # still failed 0/150 in all 5 grid attempts, and push_approach saw
        # no material change either. R6 T2 (plans/SYNC.md 2026-08-04
        # ~17:54 UTC) additionally GPU-tested the other orientation DOF --
        # wrist YAW aligned to each attempt's `stance_yaw` instead of a
        # fixed world yaw=0 -- and got byte-identical ik_ok_ticks/
        # ik_fail_ticks (0/150 on all 4 real attempts) to the unaligned
        # baseline despite the orientation genuinely differing (confirmed
        # via non-zero base yaw at those ticks). REFUTED and reverted,
        # consistent with sec 89's own precedent of not carrying an
        # unproven change forward once its falsification check fires.
        # Both tested orientation DOFs (tilt-off-vertical, wrist-twist)
        # now show no measured benefit to push_approach's IK convergence.
        top_down = self._m["_quaternion_from_rpy"](TOP_DOWN_ROLL_RAD, 0.0, 0.0)

        # Known, fully-open gripper regardless of what a preceding failed
        # grasp() attempt left it at (plan_stage4 still runs grasp() first,
        # unconditionally, ahead of this method).
        self.arms.release(
            side, step=self._tick, dt=self.sim.cfg.dt, timeout_s=2.0
        )

        # handoff sec 89: a real floor event was traced to THIS point on a
        # retry attempt -- the previous attempt's failed push_contact/ramp
        # can leave the gripper near or inside the object, and nothing
        # retracted it before the navigate_to below starts driving the
        # base toward the next stance (~0.35 m of travel observed dragging
        # the end effector ~1 m). Tuck the arm to the SAME TRANSIT_ARM_POSE
        # reset() (:378) and feed_hold() (:1711) already use before their
        # own base travel -- reusing a proven mechanism, not inventing new
        # geometry -- so no attempt starts a drive with a stale, possibly
        # interpenetrating arm target.
        #
        # N1 (SYNC 21/23): this exact tuck is the mechanism implicated in
        # cup's mid-episode floor event (tick 68641-72251, no navigate_to
        # ran in that window). ramp_arm_pose is a raw joint-space LERP with
        # no Cartesian guard -- the interpolated path can sweep close to or
        # through a bystander object even though the start (possibly near
        # the just-pushed object) and end (TRANSIT_ARM_POSE) poses are each
        # individually fine. Trace every tick's EE distance to every
        # spawned object here, per-tick, not just before/after -- that is
        # what proves the sweep instead of assuming it.
        objs_before_tuck = {
            name: self.object_position(name) for name in self.object_views
        }
        min_dist: dict[str, float] = {
            name: float("inf") for name in self.object_views
        }
        min_dist_tick: dict[str, int] = {
            name: -1 for name in self.object_views
        }

        def _sample_tuck_clearance(tick: int) -> None:
            left_ee, right_ee = self.arms.ee_world_poses()
            for ee in (left_ee[0], right_ee[0]):
                for name, pos in objs_before_tuck.items():
                    d = math.dist(
                        [float(ee[0]), float(ee[1]), float(ee[2])], pos
                    )
                    if d < min_dist[name]:
                        min_dist[name] = d
                        min_dist_tick[name] = tick

        self._m["ramp_arm_pose"](
            self.robot,
            self._m["TRANSIT_ARM_POSE"],
            step=self._tick,
            on_tick=_sample_tuck_clearance,
        )
        self.arms.sync_targets_from_measured()

        objs_after_tuck = {
            name: self.object_position(name) for name in self.object_views
        }
        tuck_z_delta = {
            name: round(
                objs_after_tuck[name][2] - objs_before_tuck[name][2], 4
            )
            for name in self.object_views
        }
        self._log_phase(
            "push_object_to_retract_tuck",
            all(abs(d) < 0.02 for d in tuck_z_delta.values()),
            tuck_z_delta=tuck_z_delta,
            tuck_min_dist_to_object_m={
                k: round(v, 4) for k, v in min_dist.items()
            },
            tuck_min_dist_tick={k: v for k, v in min_dist_tick.items()},
        )

        # GATE B1 follow-up (handoff sec 72): this method used to compute
        # its push-contact target purely from the object's current XY and
        # attempt it via arm IK from WHATEVER base position/yaw a preceding
        # grasp()/carry_object_to() attempt happened to leave the robot at
        # -- unlike reach(), which always re-stances+navigates+rotates
        # before any arm motion. Real logs (sec 72) showed push_approach/
        # push_contact failing almost every tick (ik_fail_ticks at or near
        # 100% of the attempt's budget) even for targets only ~0.1-0.2m
        # from the object, consistent with the base's yaw having drifted
        # (carry_object_to's own drive loop passes yaw=None to NavigateTo,
        # so nothing holds a heading during it) such that a small XY offset
        # ends up outside the arm's current directional reach envelope.
        # Re-establish the SAME validated stance/orientation reach() itself
        # would use for this object before attempting the push, instead of
        # trusting whatever the preceding attempt left behind.
        # T5a (plans/LOOP_PROMPT_VM_A_REV5.md, ADDED 2026-08-04): tried
        # validating the stance against `contact_z + approach_clearance`
        # (the real, higher point push_approach reaches for) instead of
        # the bare contact height -- reverted after appearing to regress
        # navigate_to arrival, but that attribution was ITSELF corrected
        # (plans/SYNC.md 2026-08-04 ~14:45 UTC): the reverted code
        # produces byte-identical results to the "regressed" run, and the
        # ORIGINAL "confirmed-good" baseline never actually had
        # navigate_to arrive either (`ok: False` on all 9/9 attempts,
        # `terminal_error_m` 2.1-3.3m -- just unmeasured at the time).
        # navigate_to has never arrived at a curobo_stance_for candidate
        # in any run recorded since T4 shipped; this is unrelated to
        # T5a's height-validation line either way. `push_stance_navigate_
        # budget_s` below is the well-evidenced next lever (T4's fix
        # picks candidates ~2-3m away; the old 25s budget may simply be
        # too short), added as a kwarg so it can be A/B'd without a code
        # change -- default unchanged (25.0s) until GPU-verified.
        live_obj_for_stance = self.object_position(object_name)
        contact_z_for_stance = live_obj_for_stance[2] + p.get(
            "push_contact_height_offset_m", 0.0
        )
        stance_xy, stance_yaw = self._stance_for(
            (live_obj_for_stance[0], live_obj_for_stance[1]),
            p.get("approach_stance", "east"),
            contact_z=contact_z_for_stance,
            stance_radius_m=p.get(
                "push_stance_radius_m", PUSH_STANCE_RADIUS_M
            ),
            # The push stance is the one caller that genuinely needs the
            # radius to grow: at PUSH_STANCE_RADIUS_M the whole annulus is
            # inside the counter's inflated footprint (0 of 180 angles
            # clear), so without a ceiling to grow toward there is no legal
            # stance at all. Named here rather than defaulted globally --
            # the grasp path shares this function and cannot afford it.
            stance_max_radius_m=PUSH_STANCE_GROWTH_CEILING_M,
        )
        # 2026-08-09 (O1 investigation): see navigate_to_avoiding_island's
        # docstring -- a direct navigate_to(*stance_xy) here drove straight
        # through the kitchen island's real PhysX collider, GPU-confirmed
        # 3/3.
        self.navigate_to_avoiding_island(
            *stance_xy,
            max_linear_mps=0.25,
            budget_s=p.get(
                "push_stance_navigate_budget_s",
                self.push_stance_navigate_budget_s,
            ),
        )
        self._rotate_to(stance_yaw)

        obj_pos = self.object_position(object_name)
        dx, dy = x - obj_pos[0], y - obj_pos[1]
        dist = math.hypot(dx, dy)
        ux, uy = (dx / dist, dy / dist) if dist > 1e-6 else (1.0, 0.0)

        behind_offset = p.get("push_behind_offset_m", 0.06)
        perception_target = (
            self._perception_push_target_cache.get(object_name)
            if self.push_perception_targets
            and object_name in PUSH_PERCEPTION_OBJECTS
            else None
        )
        if perception_target is not None:
            contact_z = perception_target[2] + p.get(
                "push_contact_height_offset_m", 0.0
            )
        else:
            contact_z = obj_pos[2] + p.get("push_contact_height_offset_m", 0.0)
        approach_clearance = p.get("push_approach_clearance_m", 0.15)
        behind_xy = (
            obj_pos[0] - ux * behind_offset,
            obj_pos[1] - uy * behind_offset,
        )

        # Approach from above first so the gripper doesn't clip through the
        # object on the way in (same two-stage pattern reach() uses for the
        # grasp approach), then descend to contact height. set_arm_target
        # (called inside reach()) raises ValueError if the target lies
        # outside the CartesianTargetTracker's workspace limits (handoff
        # sec 47.3 / plan sec 1.3) -- this call was NOT guarded until this
        # fix, and did escape uncaught live (handoff sec 48: aborted a real
        # GPU episode with total_score 0). Never let it escape and abort
        # the whole stage (plan sec 6 point 4) -- same guard as the
        # contact-phase reach() below.
        # C2.5 Part A telemetry contract (_log_phase :531, _arm_base_relative
        # :448): a phase only gets `target_norm_from_arm_base_m` -- the
        # measurement GATE C2.5A (sec 64) closed the whole stance-reachability
        # question with, 100% correlation against FR3's ~0.855m reach -- if
        # its `target=` kwarg carries THREE components. push_approach and
        # push_contact were logging the bare 2-element `behind_xy` instead of
        # the 3D pose they actually reach for, so `_arm_base_relative`'s
        # `len < 3` guard (written for navigate_to's genuinely-2D BASE
        # targets) silently dropped them, and the one number that
        # discriminates "unreachable" from "reachable but mistracked" has
        # been dark for exactly the two phases that keep failing (sec 88
        # finding 2). push_standoff (:1369) logs its 3-vector and does get
        # it. Log the real target so the norm comes back for free.
        approach_target = (
            behind_xy[0],
            behind_xy[1],
            contact_z + approach_clearance,
        )
        # Q2 pre-flight gate: a target already past the measured reach
        # ceiling before the reach() call even starts will not converge
        # inside it either -- committing to the full 8s budget just holds
        # the arm straining directly above the object instead (the
        # mechanism this session traced to cup's counter-edge fall).
        # `_arm_base_relative` returns None off-GPU (mock/test worlds,
        # `self.robot is None`) or when body lookup fails, so this check
        # degrades to a no-op exactly like the rest of the C2.5 telemetry.
        if self._reach_limit_exceeded(side, approach_target):
            # T5a: GATE T5a asks for the refused target's
            # target_norm_from_arm_base_m explicitly -- previously this
            # rejection logged NO number, only the reason string, so the
            # over-ceiling MARGIN was invisible (rejected-by-1cm and
            # rejected-by-50cm looked identical in the log).
            approach_rel = self._arm_base_relative(side, approach_target)
            self._log_phase(
                "push_approach",
                False,
                target=[round(v, 3) for v in approach_target],
                target_norm_from_arm_base_m=(
                    round(approach_rel[1], 4) if approach_rel else None
                ),
                reason="beyond_reach_limit_preflight",
            )
            return {
                "scored": False,
                "reason": "push_approach_beyond_reach_limit",
            }
        # T3 (plans/LOOP_PROMPT_VM_A_REV5.md): T2 traced a live episode
        # (2026-08-04 07:40 UTC, plans/SYNC.md) where THIS reach() call --
        # unlike push_contact's below, which already passes
        # zero_success_bail_ticks (GATE B1 fling finding, handoff sec
        # 78/80) -- ran its full 1600-tick budget after its IK started
        # diverging (ik_fail_ticks=312 from tick 1288/1600) and swept the
        # arm through a ~2.5 rad joint change, flinging the object 0.72m
        # off the table. Not a sign bug in the push direction (verified in
        # the same trace: behind_xy/aim math is correct) -- a missing
        # bail-out guard on the one reach() call in this method that never
        # had one.
        approach_ik: dict[str, Any] = {}
        try:
            approach_ok = self.arms.reach(
                side,
                approach_target,
                top_down,
                step=self._tick,
                dt=self.sim.cfg.dt,
                timeout_s=8.0,
                ik_stats=approach_ik,
                zero_success_bail_ticks=p.get(
                    "push_approach_zero_success_bail_ticks", 150
                ),
                # 2026-08-09 (O1 investigation, GPU-confirmed): a
                # succeeding-every-tick reach() here can still internally
                # flip to a wildly different joint solution mid-servo and
                # fling the object -- see arms.py::reach's own docstring
                # for the confirming trace. 0.5 rad/tick is well above any
                # normal smooth-tracking delta (early convergence bursts
                # observed this session stayed under ~0.1 rad/tick) but
                # far below the observed thrash magnitude (~2.9 rad),
                # conservative against false-positive aborts.
                max_joint_delta_rad=p.get(
                    "push_approach_max_joint_delta_rad", 0.5
                ),
            )
        except ValueError as exc:
            self._log_phase("push_approach", False, reason=str(exc))
            return {"scored": False, "reason": "push_approach_unreachable"}
        self._log_phase(
            "push_approach",
            approach_ok,
            target=[round(v, 3) for v in approach_target],
            ik=approach_ik,
            side=side,
        )
        already = self._already_scored_push_result(object_name)
        if already is not None:
            return already

        # GATE B1 crash finding (handoff sec 73): a violent push_contact
        # flung an object 1.3m and the process later segfaulted (downstream
        # CUDA/PhysX corruption from the impact). push_contact used to drive
        # straight in at full IK-servo rate from approach_clearance height --
        # the exact "shove" pattern reach()'s own docstring already
        # diagnosed and fixed for the grasp descent via a tick-by-tick
        # rate-limited ramp (GENTLE_DESCEND_M/GENTLE_DESCEND_SECONDS). Give
        # the push contact approach the same treatment instead of a second
        # full-speed reach() call.
        gentle_descend_m = p.get("push_gentle_descend_m", GENTLE_DESCEND_M)
        gentle_descend_seconds = p.get(
            "push_gentle_descend_seconds", GENTLE_DESCEND_SECONDS
        )
        if gentle_descend_m > 0.0 and approach_clearance > gentle_descend_m:
            ramp_start_z = contact_z + gentle_descend_m
            ramp_start_target = (behind_xy[0], behind_xy[1], ramp_start_z)
            # T3: same missing-guard class as push_approach above -- this
            # reach() call had no zero_success_bail_ticks either.
            ramp_start_ik: dict[str, Any] = {}
            try:
                ramp_start_ok = self.arms.reach(
                    side,
                    ramp_start_target,
                    top_down,
                    step=self._tick,
                    dt=self.sim.cfg.dt,
                    timeout_s=6.0,
                    ik_stats=ramp_start_ik,
                    zero_success_bail_ticks=p.get(
                        "push_standoff_zero_success_bail_ticks", 150
                    ),
                    # 2026-08-09: same joint-thrash guard as push_approach
                    # above -- see that call site's comment / arms.py's
                    # reach() docstring for the confirming GPU trace.
                    max_joint_delta_rad=p.get(
                        "push_standoff_max_joint_delta_rad", 0.5
                    ),
                )
            except ValueError as exc:
                self._log_phase("push_standoff", False, reason=str(exc))
                return {
                    "scored": False,
                    "reason": "push_standoff_unreachable",
                }
            self._log_phase(
                "push_standoff",
                ramp_start_ok,
                target=[round(v, 3) for v in ramp_start_target],
                ik=ramp_start_ik,
                side=side,
            )
            ramp_ticks = max(
                1, round(gentle_descend_seconds / self.sim.cfg.dt)
            )
            # handoff sec 90: this loop used to call set_arm_target/command
            # blindly for the full ramp_ticks budget with no IK-success
            # check at all -- unlike every reach() call in this method,
            # which all get the zero_success_bail_ticks protection (sec
            # 78/80). A real GPU run showed this is exactly the gap it
            # looks like: the object's z had already collapsed from 0.778
            # to 0.35 by the END of this loop (before push_contact even
            # ran), with the end effector ~1 m from where it started --
            # a violent, uncontrolled excursion this loop had no way to
            # detect or stop. Track consecutive IK failures the same way
            # `dual_arm_lula._solve_arm` reports them and bail out early
            # (freezing the arm target where it stands, matching what a
            # failed reach() call already does) rather than grinding
            # through the remaining ticks once the solver is clearly not
            # tracking the intended path.
            ramp_bail_ticks = p.get(
                "push_gentle_ramp_zero_success_bail_ticks", 150
            )
            # 2026-08-09 (O1 investigation, GPU-confirmed): this loop
            # never goes through arms.py::reach(), so it never got that
            # method's joint-thrash guard -- and a real GPU run
            # (spoon2_run13_seed7_retrybump.log) showed the object flung
            # onto the floor (z: 0.75 -> 0.007) DURING this exact ramp
            # despite reach()'s own guard already protecting
            # push_approach/standoff/contact. Same fix, same threshold,
            # applied here directly since this loop has its own
            # hand-rolled tick body instead of calling reach().
            max_joint_delta_rad = p.get(
                "push_gentle_ramp_max_joint_delta_rad", 0.5
            )
            prev_commanded_joints: list[float] | None = None
            joint_thrash_tick: int | None = None
            joint_thrash_delta_rad: float | None = None
            consecutive_ik_fail_ticks = 0
            ramp_bailed = False
            try:
                for tick in range(1, ramp_ticks + 1):
                    z = ramp_start_z + (contact_z - ramp_start_z) * (
                        tick / ramp_ticks
                    )
                    self.arms.set_arm_target(
                        side, (behind_xy[0], behind_xy[1], z), top_down
                    )
                    ik_result = self.arms.command()
                    self._tick()
                    tick_succeeded = (
                        ik_result.left_succeeded
                        if side == "left"
                        else ik_result.right_succeeded
                    )
                    if tick_succeeded:
                        consecutive_ik_fail_ticks = 0
                    else:
                        consecutive_ik_fail_ticks += 1
                        if consecutive_ik_fail_ticks >= ramp_bail_ticks:
                            ramp_bailed = True
                            break
                    commanded_now = self.arms.commanded_arm_joint_positions(
                        side
                    )
                    if prev_commanded_joints is not None:
                        delta = max(
                            abs(c - p_)
                            for c, p_ in zip(
                                commanded_now, prev_commanded_joints
                            )
                        )
                        if delta > max_joint_delta_rad:
                            joint_thrash_tick = tick
                            joint_thrash_delta_rad = round(delta, 4)
                            ramp_bailed = True
                            break
                    prev_commanded_joints = commanded_now
            except ValueError as exc:
                self._log_phase("push_gentle_ramp", False, reason=str(exc))
                return {
                    "scored": False,
                    "reason": "push_gentle_ramp_unreachable",
                }
            self._log_phase(
                "push_gentle_ramp",
                not ramp_bailed,
                ramp_ticks=ramp_ticks,
                ramp_seconds=round(gentle_descend_seconds, 3),
                bailed=ramp_bailed,
                joint_thrash_bailed=joint_thrash_tick is not None,
                joint_thrash_tick=joint_thrash_tick,
                joint_thrash_delta_rad=joint_thrash_delta_rad,
            )
            if ramp_bailed:
                return {
                    "scored": False,
                    "reason": (
                        "push_gentle_ramp_joint_thrash"
                        if joint_thrash_tick is not None
                        else "push_gentle_ramp_ik_diverged"
                    ),
                }

        # set_arm_target (called inside reach()) raises ValueError if the
        # target lies outside the CartesianTargetTracker's workspace limits
        # (handoff sec 47.3 / plan sec 1.3) -- never let that escape and
        # abort the whole stage (plan sec 6 point 4).
        #
        # GATE B1 fling finding (handoff sec 78/80): a `push_contact` whose
        # IK never converges (ik_ok_ticks staying at 0) is not held
        # motionless while it fails -- dual_arm_lula._solve_arm freezes the
        # commanded joint target at the last successful (pre-contact) pose,
        # so a failing attempt sits there, possibly interpenetrating the
        # object, for the full budget. Real logs showed the object flung
        # and floored between one such failing attempt and the next. Bail
        # out fast (well short of the 1200-tick budget) once it's clear no
        # tick has succeeded at all -- a genuinely reachable target starts
        # succeeding almost immediately (cup's push_approach, sec 78),
        # even when it still needs the full window to shrink under
        # tolerance.
        # Same 2-vs-3 component telemetry fix as push_approach above -- this
        # is the phase whose reachability is actually in question.
        contact_target = (behind_xy[0], behind_xy[1], contact_z)
        if self._reach_limit_exceeded(side, contact_target):
            # T5a: same rejection-margin telemetry as push_approach above.
            contact_rel = self._arm_base_relative(side, contact_target)
            self._log_phase(
                "push_contact",
                False,
                target=[round(v, 3) for v in contact_target],
                target_norm_from_arm_base_m=(
                    round(contact_rel[1], 4) if contact_rel else None
                ),
                reason="beyond_reach_limit_preflight",
            )
            return {
                "scored": False,
                "reason": "push_contact_beyond_reach_limit",
            }
        descend_ik: dict[str, Any] = {}
        try:
            contact_ok = self.arms.reach(
                side,
                contact_target,
                top_down,
                step=self._tick,
                dt=self.sim.cfg.dt,
                timeout_s=6.0,
                ik_stats=descend_ik,
                zero_success_bail_ticks=p.get(
                    "push_contact_zero_success_bail_ticks", 150
                ),
                # 2026-08-09: same joint-thrash guard as push_approach --
                # see that call site's comment / arms.py's reach()
                # docstring for the confirming GPU trace.
                max_joint_delta_rad=p.get(
                    "push_contact_max_joint_delta_rad", 0.5
                ),
            )
        except ValueError as exc:
            self._log_phase("push_contact", False, reason=str(exc))
            return {"scored": False, "reason": "push_contact_unreachable"}
        self._log_phase(
            "push_contact",
            contact_ok,
            target=[round(v, 3) for v in contact_target],
            ik=descend_ik,
            side=side,
        )
        if not contact_ok:
            # 2026-08-09 (O1 investigation): push_contact's target
            # (contact_target, above) is numerically IDENTICAL to
            # push_gentle_ramp's own final target -- both are computed
            # from the same contact_z/behind_xy, never recomputed between
            # the two calls. A failed reach() here does not necessarily
            # mean the target is unreachable: GPU-confirmed
            # (spoon2_run7_seed7_westwallfix.log) a failed attempt logged
            # `ik_fail_ticks: 150/440` -- only the LAST 150 CONSECUTIVE
            # ticks failed (the bail threshold), meaning ~290 real,
            # independently-converging ticks happened first, most likely
            # real contact-force perturbation (the gripper is now
            # touching the object) making the exact final pose harder to
            # hold right at the end, not a target the solver never found.
            # Discarding that real progress and aborting the whole push
            # wastes it. Fall through to the arm's current pose ONLY when
            # there is real evidence it converged at some point
            # (`ik_ok_ticks > 0`) AND the final position error is still
            # small -- never on a target that never converged even once
            # or ended up far away, which still aborts exactly as before
            # (same safety the existing bail-out already provides).
            ik_ok_ticks = descend_ik.get("ik_ok_ticks", 0)
            per_axis_err = descend_ik.get("per_axis_err_m")
            final_err_m = (
                math.dist(per_axis_err, (0.0, 0.0, 0.0))
                if per_axis_err is not None
                else None
            )
            fallback_max_err_m = p.get("push_contact_fallback_max_err_m", 0.05)
            if (
                ik_ok_ticks > 0
                and final_err_m is not None
                and final_err_m <= fallback_max_err_m
            ):
                self._log_phase(
                    "push_contact_fallback",
                    True,
                    ik_ok_ticks=ik_ok_ticks,
                    final_err_m=round(final_err_m, 4),
                    reason=(
                        "reach()_did_not_fully_converge_but_had_real_"
                        "ik_successes_and_ended_close"
                    ),
                )
            else:
                return {"scored": False, "reason": "push_contact_not_reached"}

        push_pose_relative = self.arms.arm_pose_relative(side)
        self._base_hold_anchor = None
        max_linear = p.get("max_linear_mps", 0.2)
        budget_s = p.get("budget_s", 30.0)
        skill = self._m["NavigateTo"]((x, y), None, max_linear_mps=max_linear)
        watchdog = ProgressWatchdog()
        stalled = False
        # sec 74/76: this loop used to read the object's position only at
        # the very end (`final = self.object_position(...)` below), so a
        # fling/drop (sec 73) and a floored object (sec 76 F4) were
        # indistinguishable from a clean push until the last line. Sample
        # the object's own position on the same cadence ProgressWatchdog
        # already uses for the base, so a future run can show WHERE in the
        # drive the object left the gripper's control.
        object_trace: list[tuple[int, float, float, float]] = []
        try:
            for _ in range(int(budget_s / self.sim.cfg.dt)):
                pose = self.adapter.pose()
                if self._tick_count % watchdog.sample_every_ticks == 0:
                    ox, oy, oz = self.object_position(object_name)
                    object_trace.append(
                        (
                            self._tick_count,
                            round(ox, 4),
                            round(oy, 4),
                            round(oz, 4),
                        )
                    )
                if watchdog.sample(self._tick_count, pose.x, pose.y):
                    stalled = True
                    self.adapter.apply_twist(0.0, 0.0)
                    self._tick()
                    break
                vx, vy, done = skill.compute(pose)
                self.arms.set_arm_target_relative(
                    side,
                    push_pose_relative.position,
                    push_pose_relative.orientation_wxyz,
                )
                # Same defect as `carry_object_to` had, found by auditing
                # every tick loop in this file that commands a target:
                # `set_arm_target_relative` only mutates the tracker, so
                # without `command()` the pushing arm was frozen in JOINT
                # space while the base drove, instead of holding its pose
                # relative to the base. The contact pose this loop exists
                # to maintain was never actually maintained.
                #
                # Wrapped rather than left to the enclosing `except
                # ValueError`, which deliberately ENDS the push: an IK
                # failure on one tick should cost tracking on that tick,
                # not abort a push that is otherwise going fine.
                try:
                    self.arms.command()
                except Exception as exc:  # noqa: BLE001
                    print(f"PUSH_ARM_COMMAND_FAILED {exc!r}", flush=True)
                if done:
                    self.adapter.apply_twist(0.0, 0.0)
                    self._tick()
                    break
                self.adapter.apply_twist(vx, vy)
                self._tick()
        except ValueError as exc:
            self.adapter.apply_twist(0.0, 0.0)
            self._tick()
            self._log_phase(
                "push_drive", False, reason=str(exc), object_trace=object_trace
            )
            return {"scored": False, "reason": "push_drive_tracker_limit"}

        # Retract straight up off the object, away from further contact.
        # Scoring below only reads the OBJECT's final pose, not the arm's,
        # so a failed/unreachable retract must not abort this result --
        # guard it the same way as the other two reach() calls above.
        ee_now = self.arms.ee_world_poses()[0 if side == "left" else 1][0]
        try:
            self.arms.reach(
                side,
                (ee_now[0], ee_now[1], ee_now[2] + approach_clearance),
                top_down,
                step=self._tick,
                dt=self.sim.cfg.dt,
                timeout_s=4.0,
            )
        except ValueError as exc:
            self._log_phase("push_retract", False, reason=str(exc))
        for _ in range(round(1.0 / self.sim.cfg.dt)):
            self._tick()

        final = self.object_position(object_name)
        target_z = z if z is not None else config.SINK_TABLETOP_Z

        # REVIEW #9 (handoff sec 76): judge this push by the SAME predicate
        # the official scorer uses, not by hand-picked tolerances. When the
        # destination is the sink (every scoring caller -- plan_stage4's
        # cleanup step passes SINK_CENTER_XY/SINK_TABLETOP_Z), defer to
        # config.scores_in_sink(); the old `hypot(...) <= 0.5` accepted an
        # object up to 0.5 m from the sink centre when the sink's own
        # half-diagonal is ~0.29 m, and `>= target_z - 0.05` accepted 0.697
        # when the scorer demands 0.74699.
        #
        # `stalled` is also no longer part of this conjunction: the official
        # scorer reads the OBJECT's final pose and nothing else, so a base
        # stall after the object is already in the sink is a scored success,
        # not a failure. It stays in the log/result for diagnosis.
        if config.scores_in_sink(x, y, target_z):
            scored = config.scores_in_sink(final[0], final[1], final[2])
        else:
            scored = (
                math.hypot(final[0] - x, final[1] - y) <= 0.5
                and final[2] >= target_z - 0.05
            )
        self._log_phase(
            "push_object_to",
            scored,
            target=[round(x, 3), round(y, 3)],
            final=[round(v, 3) for v in final],
            stalled=stalled,
            scorer_exact=config.scores_in_sink(x, y, target_z),
            object_trace=object_trace,
        )
        result = {"scored": bool(scored)}
        if stalled:
            result["stalled"] = True
            result["pose_trace"] = watchdog.pose_trace
        result["object_trace"] = object_trace
        return result

    # ------------------------------------------------------------------ #
    # Bimanual feeding (Stage 2) / bean recovery (Stage 3) -- P0.8
    # ------------------------------------------------------------------ #

    def _bean_poses(self) -> list[tuple[float, float, float]]:
        """Verbatim port of run_stage2_feeding.py's ``bean_poses()`` closure
        (a pure UsdGeom prim-transform read, not manipulation) -- loose
        ``bean_*`` prims are not part of ``self.object_names`` so they have
        no ``RigidPrim`` view of their own."""
        from pxr import Usd, UsdGeom

        paths = []
        for prim in Usd.PrimRange(
            self.sim.stage.GetPrimAtPath("/World/Task3")
        ):
            name = str(prim.GetName())
            if name.startswith("bean_") and prim.IsA(UsdGeom.Xformable):
                paths.append(str(prim.GetPath()))
        paths.sort()
        result = []
        for path in paths:
            prim = self.sim.stage.GetPrimAtPath(path)
            if prim:
                matrix = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                )
                t = matrix.ExtractTranslation()
                result.append((float(t[0]), float(t[1]), float(t[2])))
        return result

    def _head_pose(self) -> tuple[float, float, float]:
        """Verbatim port of run_stage2_feeding.py's ``head_pose()``."""
        from pxr import Usd, UsdGeom

        head_path = self._m["resolve_prim_path"](self.sim.stage, "head")
        matrix = UsdGeom.Xformable(
            self.sim.stage.GetPrimAtPath(head_path)
        ).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        t = matrix.ExtractTranslation()
        return (float(t[0]), float(t[1]), float(t[2]))

    def _head_contact_force_n(self) -> float | None:
        """P0.9: mirrors run_stage1_setup.py's gripper/base
        ``*_contact_force_n()`` helpers exactly (same sensor API, same
        magnitude computation). Returns None when the sensor isn't wired --
        never a false-safe 0.0 -- so a missing reading and a genuine
        zero-force reading are never conflated (see
        ``grasp_transport.contact_force_state``'s identical distinction)."""
        if self._head_contact_sensor is None:
            return None
        try:
            self._head_contact_sensor.update(self.sim.cfg.dt)
            forces = self._head_contact_sensor.data.net_forces_w
            if forces is None:
                return None
            vector = forces.detach().cpu().tolist()[0][0]
            return round(math.sqrt(sum(c * c for c in vector)), 5)
        except Exception:  # pragma: no cover - GPU/API dependent
            return None

    def _gripper_contact_force_n(self, side: str) -> float | None:
        """Same contract as ``_head_contact_force_n``, for the per-side
        gripper sensor added 2026-08-17 for close-on-contact. None means
        the sensor isn't wired (wrong gripper profile, missing prim) --
        never a false-safe 0.0."""
        sensor = getattr(self, "_gripper_contact_sensors", {}).get(side)
        if sensor is None:
            return None
        try:
            sensor.update(self.sim.cfg.dt)
            forces = sensor.data.net_forces_w
            if forces is None:
                return None
            vector = forces.detach().cpu().tolist()[0][0]
            return round(math.sqrt(sum(c * c for c in vector)), 5)
        except Exception:  # pragma: no cover - GPU/API dependent
            return None

    def scoop(self, side, **p) -> dict:
        """Thin adapter over run_stage2_feeding.py Phase 3b (scoop_enter /
        scoop_lift): dip the held spoon into the bowl, then report beans on
        the spoon. Reuses ``reach()``/``grasp()`` for pickup (this pipeline's
        Stage 2 plan does not call them separately -- see stages.py) and
        ``grading``'s bean-on-spoon geometry unchanged; only the bowl-dip
        motion itself (ported verbatim from the script above) is new to
        this file.

        [Never run on real Isaac before P0.8 -- Phase-1 Worker A (handoff
        sec 17.6) is the first GPU exercise of this method.]
        """
        m = self._m
        spoon_name = "spoon2"
        self._active_object = spoon_name

        if self._held != spoon_name:
            self.reach(side, spoon_name, **p)
            self.grasp(side, spoon_name, **p)
            if self._held != spoon_name:
                self._log_phase("scoop", False, reason="spoon_not_held")
                return {"beans_on_spoon": 0, "scored": False}

        top_down = m["_quaternion_from_rpy"](TOP_DOWN_ROLL_RAD, 0.0, 0.0)
        entry_pitch_rad = math.radians(p.get("entry_pitch_deg", 30.0))
        scoop_quat = m["_quaternion_from_rpy"](
            math.pi + entry_pitch_rad, 0.0, 0.0
        )
        bowl_pos = (
            self.object_position("bowl2")
            if "bowl2" in self.object_views
            else None
        )
        scoop_ok = False
        if bowl_pos is not None:
            bowl_target = (bowl_pos[0], bowl_pos[1], bowl_pos[2] + 0.02)
            scoop_ok = self.arms.reach(
                side,
                bowl_target,
                scoop_quat,
                step=self._tick,
                dt=self.sim.cfg.dt,
                timeout_s=5.0,
                position_tolerance_m=0.04,
            )
            self._log_phase(
                "scoop_enter",
                scoop_ok,
                target=[round(v, 3) for v in bowl_target],
            )
            if scoop_ok:
                scoop_lift = (bowl_pos[0], bowl_pos[1], bowl_pos[2] + 0.10)
                scoop_ok = self.arms.reach(
                    side,
                    scoop_lift,
                    top_down,
                    step=self._tick,
                    dt=self.sim.cfg.dt,
                    timeout_s=5.0,
                    position_tolerance_m=0.04,
                )
                self._log_phase(
                    "scoop_lift",
                    scoop_ok,
                    target=[round(v, 3) for v in scoop_lift],
                )
        else:
            self._log_phase("scoop_no_bowl", False)

        beans_on_spoon = self._count_beans_on_spoon_now(spoon_name)
        self._log_phase(
            "scoop_result", scoop_ok, beans_on_spoon=beans_on_spoon
        )
        return {
            "beans_on_spoon": beans_on_spoon,
            "scored": beans_on_spoon >= 4,
        }

    def feed_hold(self, seconds: float, **p) -> dict:
        """Thin adapter over run_stage2_feeding.py Phases 5-6 (navigate to
        the dining head via the proven doorway route, present the spoon,
        hold >= seconds). The navigation/servo call sequence is new to this
        file; the door-crossing route (``route_via_door``), the feed-zone
        offsets (``HEAD_Z_OFFSET_M`` etc.) and the hold state machine
        (``grading.update_feed_hold``) are all imported unchanged.

        [Never run on real Isaac before P0.8 -- see scoop()'s docstring.]
        """
        m = self._m
        stage2 = m["stage2"]
        side = p.get("side", "right")
        spoon_name = "spoon2"
        if self._held != spoon_name:
            result = {
                "held_seconds": 0.0,
                "required_seconds": seconds,
                "z_drop_m": 1.0,
                "beans_left": 0,
                "smooth": False,
            }
            self._last_feed_result = result
            return result

        self._base_hold_anchor = None
        m["ramp_arm_pose"](self.robot, m["TRANSIT_ARM_POSE"], step=self._tick)
        self.arms.sync_targets_from_measured()
        start_xy = (self.adapter.pose().x, self.adapter.pose().y)
        route = m["route_via_door"](start_xy, stage2.DINING_TARGET)
        nav_ok = True
        for waypoint in route[1:]:
            self.navigate_to(*waypoint, max_linear_mps=0.35, budget_s=45.0)
            nav_ok = (
                math.hypot(
                    waypoint[0] - self.adapter.pose().x,
                    waypoint[1] - self.adapter.pose().y,
                )
                <= 0.15
            )
            if waypoint == (m["TASK3_DOOR_X"], m["TASK3_KITCHEN_LANE_Y"]):
                self._rotate_to(math.pi / 2.0, budget_s=10.0)
            if not nav_ok:
                break
        self._log_phase(
            "navigate_dining", nav_ok, route=[list(w) for w in route]
        )
        if not nav_ok:
            result = {
                "held_seconds": 0.0,
                "required_seconds": seconds,
                "z_drop_m": 1.0,
                "beans_left": self._count_beans_on_spoon_now(spoon_name),
                "smooth": False,
            }
            self._last_feed_result = result
            return result

        head = self._head_pose()
        feed_z = head[2] + stage2.HEAD_Z_OFFSET_M
        spoon_insertion = (
            head[0],
            head[1] + stage2.INSERTION_Y_OFFSET_M,
            feed_z,
        )
        feed_quat = m["_quaternion_from_rpy"](math.pi, 0.0, 0.0)
        insertion_ok = self.arms.reach(
            side,
            spoon_insertion,
            feed_quat,
            step=self._tick,
            dt=self.sim.cfg.dt,
            timeout_s=8.0,
            position_tolerance_m=0.04,
        )
        # P0.9: peak head-contact force over the approach + hold, the
        # safety measurement no Stage-2 point may be banked without (sec
        # 1/16.10(D)). None (not 0.0) means the sensor was never wired --
        # see _head_contact_force_n()'s own contract.
        peak_head_force_n: float | None = self._head_contact_force_n()
        self._log_phase(
            "feed_insertion",
            insertion_ok,
            target=[round(v, 3) for v in spoon_insertion],
            head_contact_force_n=peak_head_force_n,
        )

        feed_state = m["grading"].FeedHoldState()
        needed_ticks = int(seconds / self.sim.cfg.dt)
        recovery_ticks = math.ceil(15.0 / self.sim.cfg.dt)
        hold_ticks = 0
        for _ in range(needed_ticks + recovery_ticks):
            self.arms.set_arm_target(side, spoon_insertion, feed_quat)
            self.arms.command()
            self._tick()
            reading = self._head_contact_force_n()
            if reading is not None:
                peak_head_force_n = (
                    reading
                    if peak_head_force_n is None
                    else max(peak_head_force_n, reading)
                )
            bean_count = self._count_beans_on_spoon_now(spoon_name)
            ee_pos = self.arms.ee_world_poses()[0 if side == "left" else 1][0]
            in_zone = math.dist(ee_pos, (head[0], head[1], feed_z)) <= 0.35
            feed_state = m["grading"].update_feed_hold(
                feed_state,
                bean_count=bean_count,
                in_feed_zone=in_zone,
                dt=self.sim.cfg.dt,
            )
            if feed_state.completed:
                hold_ticks = needed_ticks
                break
            if feed_state.hold_seconds > 0:
                hold_ticks = int(feed_state.hold_seconds / self.sim.cfg.dt)

        beans_left = self._count_beans_on_spoon_now(spoon_name)
        self._log_phase(
            "feed_hold",
            feed_state.completed,
            held_seconds=round(hold_ticks * self.sim.cfg.dt, 3),
            beans_left=beans_left,
            peak_head_force_n=peak_head_force_n,
        )
        result = {
            "held_seconds": round(hold_ticks * self.sim.cfg.dt, 3),
            "required_seconds": seconds,
            "z_drop_m": 0.0 if feed_state.completed else 0.2,
            "beans_left": beans_left,
            "smooth": True,
            "peak_head_force_n": peak_head_force_n,
        }
        self._last_feed_result = result
        return result

    def _count_beans_on_spoon_now(self, spoon_name: str) -> int:
        spoon_pos = self.object_position(spoon_name)
        return sum(
            1
            for bean_pos in self._bean_poses()
            if self._m["stage2"].bean_is_on_spoon(bean_pos, spoon_pos)
        )

    def pour(self, side, x, y, **p) -> dict:
        """Stage 3 (Bean Recovery). No pour primitive exists anywhere in
        this repo to adapt -- ``probe_stage3_bean_settle.py`` only measures
        passive gravity settling with zero manipulation (handoff sec 17).
        This is therefore genuinely new, intentionally minimal manipulation
        code, composed only from primitives already used elsewhere in this
        file (``reach()`` for pickup, ``arms.reach``/``set_arm_target`` for
        the tilt): carry the held bowl over the recovery region and tip it
        by rotating the wrist, then score with ``grading``'s real sphere
        scorer (unchanged) -- no kinematic teleport of the beans themselves.

        [Never run on real Isaac before P0.8 -- see scoop()'s docstring.
        This is the least-proven of the three new methods.]
        """
        m = self._m
        bowl_name = "bowl2"
        self._active_object = bowl_name

        if self._held != bowl_name:
            self.reach(side, bowl_name, **p)
            self.grasp(side, bowl_name, **p)
            if self._held != bowl_name:
                self._log_phase("pour", False, reason="bowl_not_held")
                result = {
                    "beans_delivered": 0,
                    "ratio": 0.0,
                    "scored": False,
                    "beans_total": len(self._bean_poses()),
                }
                self._last_pour_result = result
                return result

        pour_height_m = p.get("pour_height_m", 0.08)
        pour_pos = (x, y, config.SINK_TABLETOP_Z + 0.15 + pour_height_m)
        upright = m["_quaternion_from_rpy"](math.pi, 0.0, 0.0)
        over_ok = self.arms.reach(
            side,
            pour_pos,
            upright,
            step=self._tick,
            dt=self.sim.cfg.dt,
            timeout_s=8.0,
            position_tolerance_m=0.05,
        )
        self._log_phase(
            "pour_position", over_ok, target=[round(v, 3) for v in pour_pos]
        )

        tilt_rad = math.radians(100.0)  # past vertical -- tips the bowl
        tilt_quat = m["_quaternion_from_rpy"](math.pi - tilt_rad, 0.0, 0.0)
        tilt_ticks = max(1, round(2.0 / self.sim.cfg.dt))
        for _ in range(tilt_ticks):
            self.arms.set_arm_target(side, pour_pos, tilt_quat)
            self.arms.command()
            self._tick()
        settle_ticks = max(1, round(3.0 / self.sim.cfg.dt))
        for _ in range(settle_ticks):
            self._tick()

        self.arms.set_arm_target(side, pour_pos, upright)
        self.arms.command()
        for _ in range(max(1, round(1.0 / self.sim.cfg.dt))):
            self._tick()
        self.arms.release(
            side, step=self._tick, dt=self.sim.cfg.dt, timeout_s=2.0
        )
        self._held = None

        bean_positions = self._bean_poses()
        beans_total = len(bean_positions)
        beans_inside = m["grading"].count_points_in_sphere(
            [m["grading"].Point3D(*pos) for pos in bean_positions],
            m["grading"].TASK3_BEAN_RECOVERY_REGION,
        )
        ratio = (beans_inside / beans_total) if beans_total else 0.0
        self._log_phase(
            "pour_result",
            beans_inside > 0,
            beans_inside=beans_inside,
            beans_total=beans_total,
            ratio=round(ratio, 4),
        )
        result = {
            "beans_delivered": beans_inside,
            "beans_total": beans_total,
            "ratio": round(ratio, 4),
            "scored": ratio >= config.STAGE3_RATIO_FOR_2PTS,
        }
        self._last_pour_result = result
        return result

    # ------------------------------------------------------------------ #
    # Reads
    # ------------------------------------------------------------------ #

    def object_xy(self, name):
        pos = self.object_position(name)
        return (pos[0], pos[1])

    def object_z(self, name):
        return self.object_position(name)[2]

    def score_stage(self, stage: int):
        for path in (EVALUATION_DIR,):
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))
        import grading

        assert len(config.STAGE1_OBJECTS) == 4, (
            f"config.STAGE1_OBJECTS must have exactly 4 objects, got "
            f"{len(config.STAGE1_OBJECTS)}: {config.STAGE1_OBJECTS}"
        )
        if stage == 1:
            object_positions = {
                name: grading.Point3D(*self.object_position(name))
                for name in config.STAGE1_OBJECTS
            }
            result = grading.score_stage1_table_setup(
                object_positions, object_names=config.STAGE1_OBJECTS
            )
            return (
                result.score,
                result.max_score,
                {
                    "passed": list(result.passed),
                    "failed": list(result.failed),
                    "object_names": list(config.STAGE1_OBJECTS),
                },
            )
        if stage == 4:
            bounds = {
                name: grading.Bounds2D.from_point(self.object_position(name))
                for name in config.STAGE1_OBJECTS
            }
            z = {name: self.object_z(name) for name in config.STAGE1_OBJECTS}
            result = grading.score_stage4_cleanup(
                bounds, z, object_names=config.STAGE1_OBJECTS
            )
            return (
                result.score,
                result.max_score,
                {
                    "passed": list(result.passed),
                    "failed": list(result.failed),
                    "object_names": list(config.STAGE1_OBJECTS),
                },
            )
        if stage == 2:
            # P0.7 (handoff sec 17.2 item 4): wired to the real feed_score,
            # reading feed_hold()'s own measured outcome (score_stage takes
            # no arguments, so this can't be passed in directly).
            if self._last_feed_result is None:
                return 0, 4, {"note": "feed_hold() was never called"}
            r = self._last_feed_result
            score = grading.feed_score(
                beans_left=r["beans_left"],
                hold_seconds=r["held_seconds"],
                smooth=r["smooth"],
                required_hold_seconds=r["required_seconds"],
            )
            return score, 4, dict(r)
        if stage == 3:
            # Same pattern, reading pour()'s own measured outcome.
            if self._last_pour_result is None:
                return 0, 4, {"note": "pour() was never called"}
            r = self._last_pour_result
            score = grading.bean_recovery_score(
                r["beans_delivered"], r["beans_total"]
            )
            return score, 4, dict(r)
        return 0, 4, {"note": f"stage {stage} scorer not wired yet"}
