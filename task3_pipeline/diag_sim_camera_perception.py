# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""One-shot diagnostic (not a permanent module): the first live-GPU run of
``SimCameraPerception`` (M1's real camera backend, ``sim_camera_perception.
py``), never verified on GPU because M3 training had exclusive use of the
only available GPU for the rest of this session. Builds the standard
Stage 1 scene (``IsaacWorld``, ``skip_navigation=True`` -- no need to
drive the base for a perception-only check), reads ground-truth object
positions, runs OWL-ViT detection + depth back-projection, and reports
the position error per object against ground truth. Run concurrently
with grasp_run4 training (7 GB / 23 GB in use, single env here needs
very little memory) rather than waiting for the GPU to free up, per
ACTIVE_BRIEF's "never leave a milestone unstarted because another is
blocked" -- M3 isn't blocked here, but M1 doesn't need to wait either.
"""

from __future__ import annotations


def main() -> None:
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": True, "livestream": -1})
    simulation_app = app_launcher.app

    from task3_pipeline import config
    from task3_pipeline.world_isaac import IsaacWorld

    world = IsaacWorld(
        simulation_app=simulation_app,
        object_names=config.STAGE1_OBJECTS,
        skip_navigation=True,
    )
    world.reset(seed=0, head_placement="a")

    ground_truth = {
        name: world.object_position(name) for name in config.STAGE1_OBJECTS
    }
    for name, pos in ground_truth.items():
        print(f"GROUND_TRUTH {name} {pos}", flush=True)

    from task3_pipeline.sim_camera_perception import SimCameraPerception

    perception = SimCameraPerception(object_names=config.STAGE1_OBJECTS)
    print(
        f"CAMERA_POS {perception.camera_position_world} "
        f"CAMERA_QUAT {perception.camera_orientation_wxyz}",
        flush=True,
    )

    # Let the render product produce at least one real frame before
    # reading annotators -- rep annotators can return stale/empty data
    # on the very first app.update() after a render product is created.
    for _ in range(5):
        simulation_app.update()

    poses = perception.perceive(world, config.STAGE1_OBJECTS)
    for name in config.STAGE1_OBJECTS:
        pose = poses[name]
        gt = ground_truth[name]
        err = (
            (pose.position[0] - gt[0]) ** 2
            + (pose.position[1] - gt[1]) ** 2
            + (pose.position[2] - gt[2]) ** 2
        ) ** 0.5
        print(
            f"RESULT {name} visible={pose.visible} "
            f"confidence={pose.confidence:.3f} "
            f"perceived={pose.position} ground_truth={gt} "
            f"error_m={err:.4f}",
            flush=True,
        )

    import os

    os._exit(0)


if __name__ == "__main__":
    main()
