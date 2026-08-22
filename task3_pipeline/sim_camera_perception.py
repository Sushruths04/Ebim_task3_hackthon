# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""M1: the real perception backend (ACTIVE_BRIEF sec 5 M1 -- "build now,
not later"; Opus verdict on BLOCKER 2: build this IN PARALLEL with M3
training, never blocking on it).

Backend choice: a **pretrained**, zero-shot, open-vocabulary detector
(OWL-ViT, via HuggingFace ``transformers`` -- confirmed installed in the
container this session, plus internet access confirmed reachable to
huggingface.co) prompted with each object's plain-English name ("a cup",
"a bowl", ...), rather than a detector trained from scratch (ACTIVE_BRIEF
sec 10: "do not attempt pixel-based RL from scratch. Pose interface +
pretrained perception" -- this is the pretrained-perception half). COCO-
pretrained detectors were considered and rejected: COCO covers "cup" and
"bowl" but has no "plate" or "spoon" class, so a fixed-vocabulary
detector cannot cover this project's object set at all; an open-
vocabulary model can.

**Scope of what this measures**: position only, not full 6-DOF
orientation. This is a deliberate, load-bearing simplification, not an
oversight -- ``task3_rl/grasp_task.py``'s ``build_observation`` (M3,
already training) only consumes the object's RELATIVE POSITION, not
orientation, so a real backend only needs to measure position error for
GATE M1 to produce numbers M3 can actually use. Orientation estimation
(needed eventually for non-symmetric grasps) is flagged as a fast-follow,
not silently dropped -- every kitchen object here sits upright on a flat
counter in this benchmark, so orientation reduces to a single yaw angle,
which is a much smaller follow-up than full 6-DOF.

**Pure math (this file, CPU-testable)**: pinhole back-projection from a
detected pixel + depth value to a 3-D point in the camera frame, then a
camera-to-world transform. This is the part GATE M1's error measurement
actually depends on being correct, so it is tested here independent of
any model or Isaac Sim session.

**Isaac/model wiring (lazy-imported, GPU-only)**: ``SimCameraPerception``
implements the same ``Perception`` protocol as ``GroundTruthPerception``
(``task3_pipeline/perception.py``) -- swappable, per ACTIVE_BRIEF sec 2's
"train-only backend, same interface" rule. Not verified on GPU as of the
commit that added this file (see ``plans/handoff.md`` -- M3 training was
using the only available GPU at the time); the pure math below IS
verified.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from task3_pipeline.perception import IDENTITY_ORIENTATION_WXYZ, PerceivedPose


@dataclass(frozen=True)
class CameraIntrinsics:
    """Standard pinhole model. fx/fy in pixels, cx/cy the principal
    point in pixels, width/height the image size the model prompts
    should be checked against."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.fx <= 0.0 or self.fy <= 0.0:
            raise ValueError("focal lengths must be positive")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("image dimensions must be positive")


def back_project(
    u: float, v: float, depth_m: float, intrinsics: CameraIntrinsics
) -> tuple[float, float, float]:
    """Pixel (u, v) + a depth reading -> a 3-D point in the CAMERA frame
    (camera looks down +z, x right, y down -- the standard pinhole/
    OpenCV convention, not USD's).

    ``depth_m`` must be the straight-line distance along the optical
    axis (a ``distance_to_image_plane``/z-depth annotator), not the
    Euclidean distance to the camera origin (a
    ``distance_to_camera``/range annotator) -- passing the wrong one in
    silently produces a plausible-looking but systematically wrong
    position, exactly ACTIVE_BRIEF sec 3's recurring failure class in a
    new costume. Callers must know which annotator they used.
    """
    if depth_m <= 0.0:
        raise ValueError("depth_m must be positive")
    x = (u - intrinsics.cx) * depth_m / intrinsics.fx
    y = (v - intrinsics.cy) * depth_m / intrinsics.fy
    return (x, y, depth_m)


def camera_point_to_world(
    point_camera: tuple[float, float, float],
    camera_position_world: tuple[float, float, float],
    camera_orientation_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """Rotate a camera-frame point by the camera's world orientation and
    translate by its world position. Quaternion convention matches the
    rest of this repo (scalar-first wxyz, e.g.
    ``scripts/common/teleop_targets.py``)."""
    w, x, y, z = camera_orientation_wxyz
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("orientation quaternion must be nonzero")
    w, x, y, z = w / norm, x / norm, y / norm, z / norm

    px, py, pz = point_camera
    # Standard quaternion-vector rotation: q * v * q_conjugate, expanded.
    qv = (x, y, z)
    uvx = qv[1] * pz - qv[2] * py
    uvy = qv[2] * px - qv[0] * pz
    uvz = qv[0] * py - qv[1] * px
    uuvx = qv[1] * uvz - qv[2] * uvy
    uuvy = qv[2] * uvx - qv[0] * uvz
    uuvz = qv[0] * uvy - qv[1] * uvx
    rx = px + 2.0 * (w * uvx + uuvx)
    ry = py + 2.0 * (w * uvy + uuvy)
    rz = pz + 2.0 * (w * uvz + uuvz)

    return (
        rx + camera_position_world[0],
        ry + camera_position_world[1],
        rz + camera_position_world[2],
    )


def detection_to_world_position(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    camera_position_world: tuple[float, float, float],
    camera_orientation_wxyz: tuple[float, float, float, float],
) -> tuple[float, float, float]:
    """The full pipeline this module measures: pixel + depth -> world
    position. Composed from the two pure functions above so each step
    stays independently testable."""
    point_camera = back_project(u, v, depth_m, intrinsics)
    return camera_point_to_world(
        point_camera, camera_position_world, camera_orientation_wxyz
    )


def perceived_pose_from_detection(
    u: float,
    v: float,
    depth_m: float,
    intrinsics: CameraIntrinsics,
    camera_position_world: tuple[float, float, float],
    camera_orientation_wxyz: tuple[float, float, float, float],
    *,
    confidence: float,
) -> PerceivedPose:
    """Package a detection into the same ``PerceivedPose`` shape
    ``GroundTruthPerception`` returns (``task3_pipeline/perception.py``)
    -- orientation is identity (see module docstring: not measured yet,
    M3 does not consume it)."""
    position = detection_to_world_position(
        u,
        v,
        depth_m,
        intrinsics,
        camera_position_world,
        camera_orientation_wxyz,
    )
    return PerceivedPose(
        position=position,
        orientation_wxyz=IDENTITY_ORIENTATION_WXYZ,
        confidence=confidence,
        visible=True,
    )


# Points the camera at the counter, not the whole kitchen (unlike
# world_isaac.py's CAMERA_POSITION/CAMERA_LOOK_AT debug camera, which
# frames the full room for --record-video). Object detection needs the
# counter to fill more of the frame than a wide establishing shot does.
# Numbers are a reasoned placement (island counter at COUNTER_XY,
# task3_rl/isaac_grasp_env.py, top z 0.747), NOT GPU-tuned -- flagged in
# the module docstring as unverified.
PERCEPTION_CAMERA_POSITION = (-3.3, -1.753, 1.6)
PERCEPTION_CAMERA_LOOK_AT = (-4.185, -1.753, 0.8)
PERCEPTION_CAMERA_RESOLUTION = (640, 480)
PERCEPTION_CAMERA_FOV_DEG = 60.0

DEFAULT_DETECTION_CONFIDENCE_THRESHOLD = 0.1
NOT_DETECTED_POSITION = (0.0, 0.0, 0.0)


def intrinsics_from_fov(
    resolution: tuple[int, int], fov_deg: float
) -> CameraIntrinsics:
    """Pinhole intrinsics for a synthetic (distortion-free) camera at a
    given horizontal field of view -- the only thing an
    ``omni.replicator`` camera created via ``rep.create.camera`` needs
    beyond position/orientation, since it has no real lens to calibrate."""
    width, height = resolution
    fx = (width / 2.0) / math.tan(math.radians(fov_deg) / 2.0)
    return CameraIntrinsics(
        fx=fx,
        fy=fx,
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )


def _look_at_quaternion_wxyz(
    position: tuple[float, float, float], look_at: tuple[float, float, float]
) -> tuple[float, float, float, float]:
    """Quaternion for a camera at ``position`` looking toward ``look_at``,
    camera -z forward / +y up (the OpenGL/Replicator convention). Used
    only as a fallback if the created camera prim's authored orientation
    cannot be read back directly (the preferred path -- see
    ``SimCameraPerception.__init__``, which queries the actual prim
    transform rather than trusting this re-derivation, since a camera
    API's own internal look-at convention could differ in ways this
    function cannot detect from the outside)."""
    forward = tuple((a - b) for a, b in zip(look_at, position))
    norm = math.sqrt(sum(c * c for c in forward))
    if norm == 0.0:
        raise ValueError("camera position and look_at must differ")
    fx, fy, fz = (c / norm for c in forward)
    # Camera looks down -z; world "up" is +z (USD convention here).
    up = (0.0, 0.0, 1.0)
    # right = forward x up (then re-orthogonalize up = right x forward).
    rx = fy * up[2] - fz * up[1]
    ry = fz * up[0] - fx * up[2]
    rz = fx * up[1] - fy * up[0]
    rnorm = math.sqrt(rx * rx + ry * ry + rz * rz)
    if rnorm == 0.0:
        # forward parallel to up -- pick an arbitrary right vector.
        rx, ry, rz, rnorm = 1.0, 0.0, 0.0, 1.0
    rx, ry, rz = rx / rnorm, ry / rnorm, rz / rnorm
    ux = ry * fz - rz * fy
    uy = rz * fx - rx * fz
    uz = rx * fy - ry * fx
    # Rotation matrix columns: right, up, -forward (camera looks -z).
    m00, m01, m02 = rx, ux, -fx
    m10, m11, m12 = ry, uy, -fy
    m20, m21, m22 = rz, uz, -fz
    trace = m00 + m11 + m22
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s
    return (w, x, y, z)


class SimCameraPerception:
    """The real M1 backend: OWL-ViT zero-shot detection + depth back-
    projection, implementing the same ``Perception`` protocol as
    ``GroundTruthPerception`` (``task3_pipeline/perception.py``).

    **UNVERIFIED on GPU as of the commit that added this class** (see
    ``plans/handoff.md`` -- the only available GPU was running M3
    training when this was written). The pure math it calls
    (``back_project``/``camera_point_to_world``/``intrinsics_from_fov``)
    IS verified (CPU tests, this file). Wiring code below follows
    ``task3_pipeline/world_isaac.py``'s established patterns
    (``omni.replicator.core`` for the camera/annotators, lazy imports
    only reachable after ``AppLauncher``) as closely as possible to
    minimize invented-syntax risk, the same approach that found and fixed
    two real bugs quickly in ``task3_rl/isaac_grasp_env.py`` this
    session -- but the first real GPU run of THIS class should still be
    expected to surface something, per that same session's own lesson.
    """

    def __init__(
        self,
        *,
        object_names: tuple[str, ...],
        camera_position: tuple[
            float, float, float
        ] = PERCEPTION_CAMERA_POSITION,
        camera_look_at: tuple[float, float, float] = PERCEPTION_CAMERA_LOOK_AT,
        resolution: tuple[int, int] = PERCEPTION_CAMERA_RESOLUTION,
        fov_deg: float = PERCEPTION_CAMERA_FOV_DEG,
        confidence_threshold: float = DEFAULT_DETECTION_CONFIDENCE_THRESHOLD,
        camera_prim_path: str | None = None,
    ) -> None:
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        # Confirmed live on GPU (this session's first GPU run of this
        # class): `import omni.replicator.core` raises
        # `ModuleNotFoundError` unless the extension has been enabled
        # first -- world_isaac.py's own `--record-video` path never hit
        # this because some other extension it depends on happens to
        # pull omni.replicator.core in as a side effect; this class has
        # no such dependency, so it must enable it explicitly. Same
        # pattern as `enable_motion_generation_extension`
        # (scripts/scenes/scene_robot_room_keyboard.py) for Lula.
        import omni.kit.app

        app = omni.kit.app.get_app()
        extension_manager = app.get_extension_manager()
        if not extension_manager.is_extension_enabled("omni.replicator.core"):
            enabled = extension_manager.set_extension_enabled_immediate(
                "omni.replicator.core", True
            )
            if enabled is False:
                raise RuntimeError(
                    "Could not enable omni.replicator.core -- "
                    "SimCameraPerception needs it for camera/annotator "
                    "creation."
                )
            # Confirmed live on GPU: enabling immediately still left the
            # extension's OmniGraph node types unregistered on the very
            # next line -- `rep.create.camera` failed with
            # `OmniGraphError: Failed to wrap graph in node given
            # {'graph_path': '/Replicator/SDGPipeline', ...}`.
            # `set_extension_enabled_immediate` starts the extension but
            # its OmniGraph registration runs on a later app update tick,
            # not synchronously within this call -- pump a few ticks so
            # that registration actually completes before this class
            # tries to use it.
            for _ in range(10):
                app.update()

        import omni.replicator.core as rep
        from pxr import UsdGeom

        self._torch = torch
        self.object_names = object_names
        self.confidence_threshold = confidence_threshold
        self.intrinsics = intrinsics_from_fov(resolution, fov_deg)

        # 2026-08-14: `camera_prim_path` points this at one of the robot's
        # THREE real onboard cameras instead of spawning a free-floating one
        # -- `head_Camera` (ZED Mini head rig) and the two D405 wrist
        # cameras `left_Camera` / `right_Camera`. Until now this class
        # always built its own camera at a fixed world pose, so the wrist
        # cameras were never used for grasping at all, and every grasp point
        # came from a WIDE view. plans/GOTCHAS.md is explicit that this is
        # the wrong tool for geometry: "Wide-shot pointing lands ~5.7 cm
        # off. Wide shot for semantics, cropped or wrist-camera view (~15
        # cm) for geometry. Two-stage. One wide point proves nothing."
        # Default None keeps the previous behaviour byte-for-byte.
        if camera_prim_path is not None:
            self._camera_prim = camera_prim_path
        else:
            self._camera_prim = rep.create.camera(
                position=camera_position, look_at=camera_look_at
            )
        self._render_product = rep.create.render_product(
            self._camera_prim, resolution
        )
        self._rgb_annotator = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annotator.attach([self._render_product])
        # distance_to_image_plane, NOT distance_to_camera -- back_project
        # assumes z-depth along the optical axis (see its own docstring);
        # using the range annotator here would silently corrupt every
        # measured position, exactly ACTIVE_BRIEF sec 3's failure class.
        self._depth_annotator = rep.AnnotatorRegistry.get_annotator(
            "distance_to_image_plane"
        )
        self._depth_annotator.attach([self._render_product])

        # Read the camera's OWN authored world transform rather than
        # re-deriving it from position/look_at (see _look_at_quaternion_
        # wxyz's docstring for why) -- rep.create.camera returns a prim
        # path/object; resolve it to a USD prim and query its transform
        # after Replicator has actually placed it.
        camera_prim_path = (
            self._camera_prim
            if isinstance(self._camera_prim, str)
            else self._camera_prim.GetPath().pathString
        )
        from omni.usd import get_context

        stage = get_context().get_stage()
        prim = stage.GetPrimAtPath(camera_prim_path)
        xform = UsdGeom.Xformable(prim)
        world_transform = xform.ComputeLocalToWorldTransform(0)
        translation = world_transform.ExtractTranslation()
        rotation_quat = world_transform.ExtractRotationQuat()
        self.camera_position_world = (
            translation[0],
            translation[1],
            translation[2],
        )
        real = rotation_quat.GetReal()
        imaginary = rotation_quat.GetImaginary()
        self.camera_orientation_wxyz = (
            real,
            imaginary[0],
            imaginary[1],
            imaginary[2],
        )

        model_name = "google/owlvit-base-patch32"
        self._processor = OwlViTProcessor.from_pretrained(model_name)
        self._model = OwlViTForObjectDetection.from_pretrained(model_name)
        self._model.eval()
        self._text_prompts = [f"a photo of a {name}" for name in object_names]

    def perceive(
        self, world: Any, object_names: tuple[str, ...]
    ) -> dict[str, PerceivedPose]:
        from PIL import Image

        torch = self._torch
        rgb = self._rgb_annotator.get_data()
        depth = self._depth_annotator.get_data()
        image = Image.fromarray(rgb[:, :, :3])

        inputs = self._processor(
            text=[self._text_prompts], images=image, return_tensors="pt"
        )
        with torch.no_grad():
            outputs = self._model(**inputs)
        target_sizes = torch.tensor([image.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs,
            threshold=self.confidence_threshold,
            target_sizes=target_sizes,
        )[0]

        poses: dict[str, PerceivedPose] = {}
        best_score_by_label: dict[int, float] = {}
        best_box_by_label: dict[int, tuple[float, float, float, float]] = {}
        for box, score, label in zip(
            results["boxes"], results["scores"], results["labels"]
        ):
            label_idx = int(label.item())
            score_val = float(score.item())
            if score_val > best_score_by_label.get(label_idx, -1.0):
                best_score_by_label[label_idx] = score_val
                best_box_by_label[label_idx] = tuple(box.tolist())

        for idx, name in enumerate(object_names):
            if idx not in best_score_by_label:
                poses[name] = PerceivedPose(
                    position=NOT_DETECTED_POSITION,
                    orientation_wxyz=IDENTITY_ORIENTATION_WXYZ,
                    confidence=0.0,
                    visible=False,
                )
                continue
            x0, y0, x1, y1 = best_box_by_label[idx]
            u = (x0 + x1) / 2.0
            v = (y0 + y1) / 2.0
            depth_m = float(depth[int(v), int(u)])
            if depth_m <= 0.0:
                poses[name] = PerceivedPose(
                    position=NOT_DETECTED_POSITION,
                    orientation_wxyz=IDENTITY_ORIENTATION_WXYZ,
                    confidence=0.0,
                    visible=False,
                )
                continue
            poses[name] = perceived_pose_from_detection(
                u,
                v,
                depth_m,
                self.intrinsics,
                self.camera_position_world,
                self.camera_orientation_wxyz,
                confidence=best_score_by_label[idx],
            )
        return poses
