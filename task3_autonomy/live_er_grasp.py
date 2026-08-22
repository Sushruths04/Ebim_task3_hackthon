# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Ask Gemini Robotics ER-2, live and per attempt, WHERE to grasp and from
WHICH DIRECTION.

This replaces two frozen things at once:

* ``assets/derived/grasp_candidates/*.json`` -- generated 2026-08-11 and read
  at run time ever since, so a "derived" grasp point was really a constant
  that predated three stance fixes.
* the straight-down wrist orientation hardcoded in ``world_isaac.reach()``
  for every object regardless of shape (see ``er_grasp_orientation`` for the
  measurement that says it is 52-84 degrees wrong).

HOW THE ORIENTATION IS ASKED FOR, and why not simply as numbers:

ER-2 is a pointing model. It is good at "which pixel" and unreliable at
"which world-frame bearing in degrees" -- it cannot see the world axes, so
asking for an azimuth directly invites a confident wrong number. So it is
asked for two PIXELS:

    grasp_point      where the finger pads should close
    approach_from    a pixel on the side the gripper should come in from

and both are back-projected **at the grasp point's own depth**. That is the
trick that makes the second point usable: the depth buffer at
``approach_from`` would report whatever lies behind it -- a wall, the floor,
nothing -- but placing both pixels on the same fronto-parallel plane through
the object turns their difference into a real world-frame direction that does
not depend on the scene behind the object at all.

The approach axis then runs from ``approach_from`` toward ``grasp_point``,
which is the direction the gripper travels; its tilt away from straight down
is the one scalar ER-2 is asked for in words, because a picture genuinely
does not determine it (a cup can be taken from directly above or from the
side, and which is right depends on the gripper, not the image).

GOTCHAS.md, ER-2 section: "Wide-shot pointing lands ~5.7 cm off. Wide shot
for semantics, cropped or wrist-camera view (~15 cm) for geometry." A caller
handing this a wide shot should expect a wide shot's accuracy; the
``reach()`` re-anchor and the pre-close re-centre exist to absorb it.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from task3_autonomy.er_grasp_orientation import (
    approach_angles_from_quaternion,
    approach_axis,
    clamp_tilt,
    nearest_equivalent_roll,
    quaternion_from_approach,
)

# Gemini's pointing convention: [y, x], each normalised to 0-1000 over the
# image, y first. This is the model's documented output space and it is NOT
# the (u, v) pixel order the back-projection expects -- the conversion is
# `_pixel_from_point` below, and getting it backwards produces a plausible
# point on the wrong axis, which is the exact failure class GOTCHAS.md warns
# about for depth annotators.
POINT_SCALE = 1000.0

# The response is schema-constrained so the model cannot answer in prose and
# cannot omit a field. `call_er` already forces `application/json`; this
# additionally pins the shape.
GRASP_POSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "grasp_point": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        "approach_from": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        "tilt_deg": {"type": "number"},
        "roll_deg": {"type": "number"},
        "long_axis_point": {
            "type": "array",
            "items": {"type": "number"},
            "minItems": 2,
            "maxItems": 2,
        },
        "reason": {"type": "string"},
    },
    "required": ["grasp_point", "approach_from", "tilt_deg", "roll_deg"],
}


def natural_label(object_name: str) -> str:
    """Scene prim name -> what a person would call it.

    The scene names objects `plate2`, `spoon2`, `simple_tray` -- the trailing
    index disambiguates duplicate prims and means nothing to a vision model,
    and an underscore is not a word boundary it should have to guess at.
    Derived by rule rather than by a lookup table so a new object in the
    scene needs no code change: strip the disambiguating digits, split the
    underscores.
    """
    return object_name.rstrip("0123456789").replace("_", " ").strip() or object_name


def grasp_prompt(object_name: str) -> str:
    """The instruction sent with the frame.

    It states the gripper's real constraint (a two-finger parallel jaw that
    must close on two opposing surfaces) because that, not the object's
    appearance, is what decides whether a rim, a handle or a whole body is
    the right thing to take -- and it asks for the approach as a pixel rather
    than as an angle for the reason in the module docstring.
    """
    return (
        "You are choosing a grasp for a two-finger parallel-jaw gripper on a "
        "mobile robot arm. The jaws close on two opposing surfaces and open "
        f"about 8 cm. Look at the {natural_label(object_name)} in this "
        "image.\n\n"
        "Return JSON with:\n"
        '- "grasp_point": [y, x] normalised 0-1000, the point where the '
        "finger pads should close on the object.\n"
        "  PREFER wrapping the jaws around the whole BODY of the object, "
        "across its narrowest width, if that width is under 8 cm -- a cup, "
        "a mug or a bottle is best taken by closing around its side wall, "
        "not by pinching its rim. A body grasp has two solid opposing "
        "surfaces and is far more secure than a rim pinch on a thin wall.\n"
        "  Only pick a rim, an edge, a handle or a stem when the body is "
        "too wide for the jaws to close around.\n"
        "  Never pick the centre of a wide flat surface.\n"
        '- "approach_from": [y, x] normalised 0-1000, a point in the image '
        "on the side the gripper should travel in FROM to reach that grasp "
        "point without hitting the object or the surface it rests on.\n"
        '- "tilt_deg": 0-90. 0 means the gripper comes straight down from '
        "above; 90 means it comes in horizontally from the side. Pick what "
        "this object's shape actually needs.\n"
        "  An upright cup, mug or bottle standing on a surface should be "
        "taken from DIRECTLY ABOVE, tilt near 0: the open jaws descend "
        "around the outside of it and then close on its walls. Coming in "
        "from the side pushes the object over before the jaws ever reach "
        "around it.\n"
        "  A thin item lying flat, like a spoon, can also be taken from "
        "above at a shallow angle.\n"
        "  Use a side approach only for something with no accessible top, "
        "like a flat plate or tray lying on a counter.\n"
        '- "long_axis_point": [y, x] normalised 0-1000, a point further '
        "along the object's LONGEST dimension from the grasp point -- the "
        "tip of a spoon's handle, the far end of a bottle. The jaws will be "
        "closed perpendicular to the line between the two, so they bite "
        "across the object rather than sliding along it. Give the grasp "
        "point again if the object has no long axis.\n"
        '- "roll_deg": rotation of the jaw line about the approach '
        "direction, so the jaws straddle the narrow dimension of the object "
        "rather than its wide one. Ignored when long_axis_point differs "
        "from grasp_point.\n"
        '- "reason": one short sentence.'
    )


@dataclass(frozen=True)
class ErGraspAnswer:
    """A parsed, validated ER-2 grasp answer in PIXEL space.

    Deliberately stops short of world coordinates: this half needs no camera,
    no Isaac and no GPU, so it is testable on its own, and the world
    conversion below is a separate pure function over it.
    """

    grasp_uv: tuple[float, float]
    approach_uv: tuple[float, float]
    tilt_deg: float
    roll_deg: float
    reason: str = ""
    long_axis_uv: tuple[float, float] | None = None

    def quaternion(self, azimuth_deg: float) -> tuple[float, float, float, float]:
        return quaternion_from_approach(self.tilt_deg, azimuth_deg, self.roll_deg)


def _pixel_from_point(
    point: Any, width: int, height: int
) -> tuple[float, float]:
    """``[y, x]`` normalised 0-1000 -> ``(u, v)`` in pixels.

    Rejects rather than clamps an out-of-range coordinate: a model answer
    outside the image is a wrong answer, and silently clamping it to the
    border would hand back a confident grasp point on the edge of the frame.
    """
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"expected a [y, x] pair, got {point!r}")
    y_norm, x_norm = float(point[0]), float(point[1])
    if not (0.0 <= y_norm <= POINT_SCALE and 0.0 <= x_norm <= POINT_SCALE):
        raise ValueError(
            f"point {point!r} is outside the 0-{POINT_SCALE:.0f} range"
        )
    return (x_norm / POINT_SCALE * width, y_norm / POINT_SCALE * height)


def parse_grasp_answer(
    raw: Any, width: int, height: int
) -> ErGraspAnswer:
    """Validate ER-2's JSON into an :class:`ErGraspAnswer`.

    Raises ``ValueError`` on anything malformed. Callers treat that as "no
    live answer this attempt" and fall back -- never as a reason to abort the
    grasp, matching ``_perception_grasp_target``'s existing contract.
    """
    if isinstance(raw, str):
        raw = json.loads(raw)
    if isinstance(raw, list):  # the model occasionally wraps it in an array
        if not raw:
            raise ValueError("empty response array")
        raw = raw[0]
    if not isinstance(raw, dict):
        raise ValueError(f"expected a JSON object, got {type(raw).__name__}")

    grasp_uv = _pixel_from_point(raw.get("grasp_point"), width, height)
    approach_uv = _pixel_from_point(raw.get("approach_from"), width, height)
    tilt_deg = clamp_tilt(float(raw.get("tilt_deg", 0.0)))
    roll_deg = float(raw.get("roll_deg", 0.0))
    if not math.isfinite(roll_deg):
        raise ValueError("roll_deg is not finite")
    long_axis_uv = None
    if raw.get("long_axis_point") is not None:
        try:
            candidate = _pixel_from_point(
                raw.get("long_axis_point"), width, height
            )
            if math.dist(candidate, grasp_uv) > 1.0:
                long_axis_uv = candidate
        except ValueError:
            long_axis_uv = None
    return ErGraspAnswer(
        grasp_uv=grasp_uv,
        approach_uv=approach_uv,
        tilt_deg=tilt_deg,
        roll_deg=roll_deg,
        reason=str(raw.get("reason", "")),
        long_axis_uv=long_axis_uv,
    )


def azimuth_from_world_points(
    grasp_world: tuple[float, float, float],
    approach_world: tuple[float, float, float],
) -> float | None:
    """Compass bearing, in degrees, of travel FROM ``approach_world`` TO
    ``grasp_world``, flattened into the horizontal plane.

    Returns ``None`` when the two points are horizontally coincident -- that
    means ER-2 put ``approach_from`` directly above or below the grasp point,
    i.e. it is describing a top-down approach, and there is no azimuth to
    speak of. The caller should use a zero tilt in that case rather than
    inventing a bearing.
    """
    dx = grasp_world[0] - approach_world[0]
    dy = grasp_world[1] - approach_world[1]
    if math.hypot(dx, dy) < 1e-6:
        return None
    return math.degrees(math.atan2(dy, dx))


def roll_across_long_axis(
    grasp_world: tuple[float, float, float],
    long_axis_world: tuple[float, float, float],
    tilt_deg: float,
    azimuth_deg: float,
) -> float | None:
    """Roll that puts the jaw line PERPENDICULAR to the object's long axis.

    The jaw line is the grasp frame's x axis (see
    `er_grasp_orientation.quaternion_from_approach`, whose base frame is
    `Rz(roll) @ Rx(pi)`). Rolling by `theta` rotates that line about the
    approach axis, so the roll wanted is the one whose x axis is normal to
    the object's long axis projected into the plane the jaws close in.

    Returns None when the long axis is degenerate or points along the
    approach direction, in which case there is no meaningful perpendicular
    and the caller should keep the model's own roll.
    """
    axis = [b - a for a, b in zip(grasp_world, long_axis_world)]
    norm = math.sqrt(sum(v * v for v in axis))
    if norm < 1e-6:
        return None
    axis = [v / norm for v in axis]

    approach = approach_axis(tilt_deg, azimuth_deg)
    # Component of the long axis perpendicular to the approach: the part the
    # jaws can actually straddle.
    dot = sum(a * b for a, b in zip(axis, approach))
    perp = [a - dot * b for a, b in zip(axis, approach)]
    perp_norm = math.sqrt(sum(v * v for v in perp))
    if perp_norm < 1e-3:
        return None
    perp = [v / perp_norm for v in perp]

    # Find the roll whose jaw line (frame x axis) is closest to `perp`, then
    # add 90 degrees so the jaws close ACROSS the object rather than along
    # it. Searched rather than solved: a closed form here would need the
    # quaternion convention inlined, and this runs once per grasp.
    best_roll = None
    best_dot = -2.0
    for step in range(0, 360, 2):
        w, x, y, z = quaternion_from_approach(tilt_deg, azimuth_deg, float(step))
        jaw = (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y + z * w),
            2.0 * (x * z - y * w),
        )
        d = abs(sum(a * b for a, b in zip(jaw, perp)))
        if d > best_dot:
            best_dot, best_roll = d, float(step)
    if best_roll is None:
        return None
    return (best_roll + 90.0 + 180.0) % 360.0 - 180.0


@dataclass(frozen=True)
class LiveGraspPose:
    """The finished answer, in world frame, ready for ``reach()``."""

    xyz: tuple[float, float, float]
    quaternion_wxyz: tuple[float, float, float, float]
    tilt_deg: float
    azimuth_deg: float
    roll_deg: float
    reason: str = ""

    def as_log(self) -> dict[str, Any]:
        """Log fields for `_log_phase(**pose.as_log())`.

        The model's own explanation is keyed `er_reason`, NOT `reason`:
        `_log_phase` callers pass their own `reason=` for a rejection, and a
        bare `reason` here would collide with it as a duplicate keyword
        argument -- a TypeError raised only on the failure path, i.e. exactly
        when the diagnostic is needed most.
        """
        return {
            "xyz": [round(v, 4) for v in self.xyz],
            "tilt_deg": round(self.tilt_deg, 1),
            "azimuth_deg": round(self.azimuth_deg, 1),
            "roll_deg": round(self.roll_deg, 1),
            "er_reason": self.reason,
        }


def grasp_pose_from_answer(
    answer: ErGraspAnswer,
    depth_m: float,
    view_matrix: Any,
    proj_matrix: Any,
    width: int,
    height: int,
) -> LiveGraspPose:
    """Pixel-space answer + the grasp point's depth -> a world pose.

    Both pixels are unprojected at ``depth_m`` -- see the module docstring
    for why the approach pixel must NOT use its own depth reading.

    Uses `perception_grasp.project_to_world` (the camera's own view/
    projection matrices) rather than `sim_camera_perception`'s
    intrinsics + authored-transform route. That is not a style preference,
    it is measured: the matrix path lands grasp points 0.04-0.11 m from
    ground truth on this exact camera, while the authored-transform route
    put ER-2's correctly-identified plate 500 m away along -Y on a camera
    that looks along -X (outputs/keep_live_er_run2.log). A USD camera looks
    down its own -Z, `back_project` returns +Z-forward OpenCV, and composing
    them without that flip is silently wrong. `project_to_world` handles it
    (`cam_z = -depth_m`) and is already proven in the precompute path.
    """
    from task3_autonomy.perception_grasp import project_to_world

    grasp_world = project_to_world(
        answer.grasp_uv[0],
        answer.grasp_uv[1],
        depth_m,
        view_matrix,
        proj_matrix,
        width,
        height,
    )
    approach_world = project_to_world(
        answer.approach_uv[0],
        answer.approach_uv[1],
        depth_m,
        view_matrix,
        proj_matrix,
        width,
        height,
    )
    # Roll from the object's LONG AXIS, so the jaws bite ACROSS it.
    #
    # ASPIRE (NVIDIA GEAR, 2026) lists our exact symptom and its cause:
    # "Grasp succeeds but drops during lift -> gripper half-closed on
    # slippery/ELONGATED object -> try perpendicular yaw or a
    # fingers-along-long-axis approach". Measured here: spoon2 closes at
    # gripper_position_rad 0.0568/0.0933 -- half-closed -- and then loses the
    # object, while ER-2 returned roll_deg 0 nearly every time. Jaws closing
    # along a spoon's stem slide off it; jaws closing across it bite.
    #
    # Derived from the same pointing trick as the approach direction: a
    # second pixel further along the object, unprojected at the SAME depth,
    # gives a real world-frame long axis without trusting the model to
    # produce an angle.
    long_axis_world = None
    if answer.long_axis_uv is not None:
        long_axis_world = project_to_world(
            answer.long_axis_uv[0],
            answer.long_axis_uv[1],
            depth_m,
            view_matrix,
            proj_matrix,
            width,
            height,
        )
    azimuth_deg = azimuth_from_world_points(grasp_world, approach_world)
    tilt_deg = answer.tilt_deg
    if azimuth_deg is None:
        # No horizontal separation: treat it as the top-down grasp it is,
        # rather than tilting toward an arbitrary bearing.
        tilt_deg, azimuth_deg = 0.0, 0.0
    roll_deg = answer.roll_deg
    if long_axis_world is not None:
        derived = roll_across_long_axis(
            grasp_world, long_axis_world, tilt_deg, azimuth_deg
        )
        if derived is not None:
            roll_deg = derived
    return LiveGraspPose(
        xyz=grasp_world,
        quaternion_wxyz=quaternion_from_approach(
            tilt_deg, azimuth_deg, roll_deg
        ),
        tilt_deg=tilt_deg,
        azimuth_deg=azimuth_deg,
        roll_deg=roll_deg,
        reason=answer.reason,
    )


def crop_box_around(
    u: float,
    v: float,
    width: int,
    height: int,
    half_size_px: int,
) -> tuple[int, int, int, int]:
    """A square crop centred on ``(u, v)``, clamped inside the image.

    THE POINT OF CROPPING. GOTCHAS records that a wide shot points ~5.7 cm
    off and prescribes "wide shot for semantics, cropped or wrist-camera
    view for geometry". That advice is quoted in three separate comments in
    this repo and was implemented in none of them, so every grasp has been
    driven by stage one alone -- measured, allobj_7 spoon2: `miss_m 0.0495`
    from the wide shot, against jaws that need centimetre accuracy.

    Cropping is preferred over a fresh close-range capture because the
    close-range capture does not work here: at contact the head camera sees
    the robot's own arm, and ER-2 said so in as many words -- "The image
    does not contain a spoon, but rather a close-up of a robotic arm joint
    or tool changer interface" -- which the miss check then rejected at
    0.3555 m. The crop re-asks about the SAME unobstructed pixels at higher
    effective resolution, so it cannot introduce an occlusion that the wide
    shot did not already have.

    The box is clamped, never allowed to leave the image, and is always a
    valid non-empty region so the caller never has to special-case an edge
    object.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    half = max(1, int(half_size_px))
    x0 = int(round(u)) - half
    y0 = int(round(v)) - half
    x1 = x0 + 2 * half
    y1 = y0 + 2 * half
    # Shift a box that overhangs, rather than shrinking it: a consistent
    # crop SIZE keeps the model's effective resolution the same wherever
    # the object sits, which is the whole reason for cropping.
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > width:
        x0 -= x1 - width
        x1 = width
    if y1 > height:
        y0 -= y1 - height
        y1 = height
    x0 = max(0, x0)
    y0 = max(0, y0)
    return (x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))


def uv_from_crop(
    u_crop: float, v_crop: float, box: tuple[int, int, int, int]
) -> tuple[float, float]:
    """Map a pixel in the CROPPED image back to the full frame.

    Forgetting this is the whole risk of cropping: ER-2's answer is
    normalised to whatever image it was shown, so a crop's coordinates are
    a confident, plausible-looking answer about the wrong part of the
    scene -- the same failure class that put a correctly-identified plate
    500 m away earlier in this project.
    """
    x0, y0, _x1, _y1 = box
    return (float(u_crop) + float(x0), float(v_crop) + float(y0))


def answer_in_full_frame(
    answer: "ErGraspAnswer", box: tuple[int, int, int, int]
) -> "ErGraspAnswer":
    """``answer`` re-expressed in full-frame pixels. Angles are unchanged.

    Tilt, roll and the reason text are properties of the object and the
    approach, not of the framing, so only the pixel fields move.
    """
    return ErGraspAnswer(
        grasp_uv=uv_from_crop(*answer.grasp_uv, box),
        approach_uv=uv_from_crop(*answer.approach_uv, box),
        tilt_deg=answer.tilt_deg,
        roll_deg=answer.roll_deg,
        reason=answer.reason,
        long_axis_uv=(
            uv_from_crop(*answer.long_axis_uv, box)
            if answer.long_axis_uv is not None
            else None
        ),
    )


def request_grasp_answer(
    image_path: Path,
    object_name: str,
    width: int,
    height: int,
    *,
    api_keys: list[str] | None = None,
    timeout_s: int = 30,
) -> ErGraspAnswer:
    """One live ER-2 round trip. Raises on any failure; callers fall back.

    Imported lazily so this module stays importable (and its pure half stays
    testable) on a machine with no API key and no network.
    """
    import sys

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.task3.probe_gemini_er_vs_ground_truth import _api_keys, call_er

    keys = api_keys if api_keys is not None else _api_keys()
    raw, _latency_s, _key = call_er(
        image_path,
        grasp_prompt(object_name),
        keys,
        response_schema=GRASP_POSE_SCHEMA,
        timeout_s=timeout_s,
    )
    return parse_grasp_answer(raw, width, height)


def describe(quaternion_wxyz: tuple[float, float, float, float]) -> str:
    """Human-readable tilt/azimuth/roll for a commanded quaternion.

    Every run should be able to say, in the log, what orientation it actually
    commanded in the same units the organisers' demonstrations were measured
    in -- 52-84 degrees of tilt. A quaternion in a log line is unreadable and
    that is part of why the top-down constant survived so long.
    """
    tilt, azimuth, roll = approach_angles_from_quaternion(quaternion_wxyz)
    return f"tilt={tilt:.1f} azimuth={azimuth:.1f} roll={roll:.1f}"
