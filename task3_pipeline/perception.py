# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""M1: the perception boundary (ACTIVE_BRIEF.md sec 2/5).

``perceive()`` is the ONLY object-pose interface the deployed policy may use.
``WorldAdapter.object_xy``/``object_z`` remain privileged-state reads used for
training and for A/B error measurement ONLY -- never in the runtime path Phase
II will exercise, per ACTIVE_BRIEF sec 2's three rules:

1. The policy never sees pixels and never sees privileged state -- it gets
   object pose **relative to the gripper frame** plus proprioception.
2. Perception is a separate, swappable module behind one interface, backed by
   a **pretrained** vision model (never a from-scratch detector).
3. Ground-truth poses are a TRAINING-ONLY backend of this SAME interface, so
   swapping it for the real backend later is a backend swap, not a rewrite.

Two implementations exist so far:

* ``GroundTruthPerception`` (this file) -- wraps any ``WorldAdapter``'s
  ``object_xy``/``object_z`` reads. Confidence is always 1.0 and nothing ever
  drops out, because it IS the ground truth. This is the training-only
  backend from rule 3, and the CPU-testable default while the real backend
  does not exist yet.
* ``SimCameraPerception`` (``task3_pipeline/sim_camera_perception.py``) --
  OWL-ViT (pretrained, zero-shot) detection + depth back-projection. Its
  pure math (pinhole back-projection, camera-to-world transform, FOV-to-
  intrinsics) is CPU-tested; the Isaac/model wiring is written but NOT yet
  GPU-verified as of the commit that added it -- the only available GPU
  was running M3 training at the time (see ``plans/handoff.md``). GATE
  M1's actual subject (the measured pose-error distribution) still needs
  a live run to produce real numbers -- do not report it from the code
  existing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from task3_pipeline.world import WorldAdapter

IDENTITY_ORIENTATION_WXYZ = (1.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class PerceivedPose:
    """One object's 6-D pose plus how much to trust it.

    ``position``/``orientation_wxyz`` are in the WORLD frame here --
    the same frame ``WorldAdapter.object_xy``/``object_z`` already use --
    so a caller can diff a perceived pose against ground truth directly for
    the GATE M1 error measurement. Converting to the gripper-relative frame
    M3's policy actually consumes is the caller's job (it needs the current
    end-effector pose, which ``Perception`` does not have), not this one's.
    """

    position: tuple[float, float, float]
    orientation_wxyz: tuple[float, float, float, float]
    confidence: float
    visible: bool

    def __post_init__(self) -> None:
        if len(self.position) != 3:
            raise ValueError("position must be an (x, y, z) tuple")
        if len(self.orientation_wxyz) != 4:
            raise ValueError("orientation_wxyz must be a (w, x, y, z) tuple")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("confidence must be in [0.0, 1.0]")
        if not self.visible and self.confidence != 0.0:
            raise ValueError(
                "an invisible detection must report confidence 0.0"
            )


@runtime_checkable
class Perception(Protocol):
    """The ONE interface the deployed policy may read object poses through."""

    def perceive(
        self, world: WorldAdapter, object_names: tuple[str, ...]
    ) -> dict[str, PerceivedPose]: ...


class GroundTruthPerception:
    """Training-only backend (ACTIVE_BRIEF sec 2 rule 3). Never confuse this
    with the deployed path -- it exists so M3's training loop and the future
    real backend can be A/B-measured against the exact same call shape."""

    def perceive(
        self, world: WorldAdapter, object_names: tuple[str, ...]
    ) -> dict[str, PerceivedPose]:
        poses: dict[str, PerceivedPose] = {}
        for name in object_names:
            x, y = world.object_xy(name)
            z = world.object_z(name)
            poses[name] = PerceivedPose(
                position=(x, y, z),
                orientation_wxyz=IDENTITY_ORIENTATION_WXYZ,
                confidence=1.0,
                visible=True,
            )
        return poses


def pose_error(
    perceived: PerceivedPose, truth_position: tuple[float, float, float]
) -> float:
    """Euclidean position error -- the measurement GATE M1 requires per
    object (mean/sigma across many perceive() calls), never estimated."""
    return math.dist(perceived.position, truth_position)
