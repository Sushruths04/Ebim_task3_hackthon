#!/usr/bin/env python3
# Copyright (c) 2026 The EBiM Benchmark Contributors
# SPDX-License-Identifier: Apache-2.0

"""Launch the robot room in Isaac Sim with the mobile FR3 placed."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

COMMON_DIR = Path(__file__).resolve().parents[1] / "common"
TASK3_DIR = Path(__file__).resolve().parents[1] / "task3"
TASK3_EVAL_DIR = Path(__file__).resolve().parents[1] / "evaluation" / "task3"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))
if str(TASK3_DIR) not in sys.path:
    sys.path.insert(0, str(TASK3_DIR))
if str(TASK3_EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(TASK3_EVAL_DIR))

from path_utils import asset_path

ISAACSIM_EXPERIENCES = {
    "base": "/isaac-sim/apps/isaacsim.exp.base.kit",
    "full": "/isaac-sim/apps/isaacsim.exp.full.kit",
}
ROS2_BRIDGE_ROOT = Path("/isaac-sim/exts/isaacsim.ros2.bridge")
ROS2_ENV_READY_VAR = "EBIM_ROS2_BRIDGE_ENV_READY"
INSIDE_KIT_ENV_VAR = "EBIM_SCENE_LAUNCH_INSIDE_KIT"
INNER_ARGV_ENV_VAR = "EBIM_SCENE_LAUNCH_ARGV"
ISAACSIM_LAUNCHER = Path("/isaac-sim/isaac-sim.sh")
DEFAULT_BEAN_COLOR = (0.20, 0.12, 0.07)
DEFAULT_BEAN_COUNT = 300
DEFAULT_BEAN_DENSITY = 850.0
BOWL_USD = asset_path("bowl2.usd")
TASK3_BOWL_POSITION = (-4.3, -1.5, 0.74659)
TASK3_HEAD_PLACEMENTS = {
    "A": ((-2.8, 1.7, 0.74659), (0.0, 0.0, 270.0)),
    "B": ((-2.4, 1.7, 0.74659), (0.0, 0.0, 270.0)),
    "C": ((-2.0, 1.7, 0.74659), (0.0, 0.0, 270.0)),
    "D": ((-1.6, 1.7, 0.74659), (0.0, 0.0, 270.0)),
    "E": ((-1.35, 1.95, 0.74659), (0.0, 0.0, 0.0)),
    "F": ((-1.6, 2.2, 0.74659), (0.0, 0.0, 90.0)),
    "G": ((-2.0, 2.2, 0.74659), (0.0, 0.0, 90.0)),
    "H": ((-2.4, 2.2, 0.74659), (0.0, 0.0, 90.0)),
    "I": ((-2.8, 2.2, 0.74659), (0.0, 0.0, 90.0)),
}
INITIAL_VIEW_POSE = (
    (-8.12589, -3.29067, 2.79653),
    (73.13762, 0.0, -50.88313),
)
BEAN_PHYSICS = {
    "radius": 0.0025,
    "half_height": 0.0016,
    "spawn_height": 0.02,
    "spawn_wall_thickness": 0.016,
    "spawn_spacing_scale": 1.2,
    "friction": 0.55,
    "restitution": 0.02,
}
TASK_ROBOT_POSES = {
    "task1": {"position": (4.4, -2.5, 0.0), "yaw": 90.0},
    "task2": {"position": (4.4, 2.6, 0.0), "yaw": -90.0},
    "task3": {"position": (-4.6, 2.7, 0.0), "yaw": -90.0},
}
INITIAL_ROBOT_JOINT_POS = {
    "left_fr3v2_joint1": 0.0,
    "left_fr3v2_joint2": -1.5,
    "left_fr3v2_joint3": 0.0,
    "left_fr3v2_joint4": -2.2,
    "left_fr3v2_joint5": 0.0,
    "left_fr3v2_joint6": 1.5,
    "left_fr3v2_joint7": 0.785,
    "right_fr3v2_joint1": 0.0,
    "right_fr3v2_joint2": -1.5,
    "right_fr3v2_joint3": 0.0,
    "right_fr3v2_joint4": -2.2,
    "right_fr3v2_joint5": 0.0,
    "right_fr3v2_joint6": 1.5,
    "right_fr3v2_joint7": 0.785,
}

# Task 2 Specific
TASK2_TABLE_POSITION = (2.05, 1.95, 0.75)
TASK2_CAMERA_POSITION = (2.087, 1.885, 2.7)
TASK2_OBJECT_SPAWN_CONFIG = {  # relative to table origin
    "thermalpad": {  # 2 deformable meshes + attachment
        "asset_path": "task2_objects/Ram_ThermalPad_Res20_Top.usda",
        "position": (-0.3, 0.0, 0.1),
        "rotation": (0.70711, 0.0, 0.0, 0.70711),
    },
    "thermalpad_base": {  # 1 rigid kinematic mesh
        "asset_path": "task2_objects/sticker_base.usda",
        "position": (-0.31, -0.04, 0.017),
        "rotation": (1.0, 0.0, 0.0, 0.0),
    },
    "board_target": {  # 1 rigid body
        "asset_path": "task2_objects/Ram_Board_Target.usda",
        "position": (0.1, 0.0, 0.0),
        "rotation": (0.70711, 0.0, 0.0, 0.70711),
    },
    "boards": {  # 3 rigid bodies
        "asset_path": "task2_objects/Ram_Board.usda",
        "spawns": [
            {
                "position": (-0.1, 0.0, 0.0),
                "rotation": (0.70711, 0.0, 0.0, 0.70711),
            },
            {
                "position": (0.0, 0.0, 0.0),
                "rotation": (0.70711, 0.0, 0.0, 0.70711),
            },
            {
                "position": (0.2, 0.0, 0.0),
                "rotation": (0.70711, 0.0, 0.0, 0.70711),
            },
        ],
    },
}


def parse_args() -> argparse.Namespace:
    argv = None
    if os.environ.get(INSIDE_KIT_ENV_VAR) == "1":
        raw_argv = os.environ.get(INNER_ARGV_ENV_VAR)
        if raw_argv:
            argv = json.loads(raw_argv)

    parser = argparse.ArgumentParser(
        description=(
            "Launch robot_room.usd with the mobile FR3 in Isaac Sim."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--room-usd",
        type=Path,
        default="robot_room.usd",
        help=("Room USD to reference under asset folder."),
    )
    parser.add_argument(
        "--robot-usd",
        type=Path,
        default=None,
        help="Robot USD to reference. Defaults to the Franka mobile FR3 USD.",
    )
    parser.add_argument(
        "--task",
        choices=tuple(TASK_ROBOT_POSES),
        default="task3",
        help="Task preset used for the robot spawn position.",
    )
    parser.add_argument(
        "--robot-x",
        type=float,
        default=None,
        help="Override the preset robot X position.",
    )
    parser.add_argument(
        "--robot-y",
        type=float,
        default=None,
        help="Override the preset robot Y position.",
    )
    parser.add_argument(
        "--robot-z",
        type=float,
        default=None,
        help="Override the preset robot Z position.",
    )
    parser.add_argument(
        "--robot-yaw",
        type=float,
        default=None,
        help="Override the preset robot yaw in degrees.",
    )
    parser.add_argument(
        "--head-placement",
        type=head_placement_arg,
        default="random",
        help=("Task3 head placement: A-I, or random. Lowercase is accepted."),
    )
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=int,
        default=1,
        help="Number of Isaac Lab environments for keyboard control.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Isaac Lab simulation device used for keyboard control.",
    )
    parser.add_argument(
        "--stabilization-steps",
        type=int,
        default=0,
        help="Initial physics steps before enabling keyboard control.",
    )
    parser.add_argument(
        "--record-teleop",
        action="store_true",
        help="Record sampled teleop targets and phase markers as JSONL.",
    )
    parser.add_argument(
        "--record-dir",
        type=Path,
        default=Path("outputs/task3_teleop"),
        help="Parent directory for recorded teleop probe episodes.",
    )
    parser.add_argument(
        "--episode-name",
        default=None,
        help="Unique output directory name; auto-generated when omitted.",
    )
    parser.add_argument(
        "--record-every-steps",
        type=int,
        default=10,
        help="Record one JSONL sample every N simulation steps.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=0.0,
        help="Stop teleop after this many seconds; 0 means no timeout.",
    )
    parser.add_argument(
        "--livestream",
        action="store_true",
        help="Enable public WebRTC streaming for remote keyboard control.",
    )
    parser.add_argument(
        "--public-ip",
        default=os.environ.get("PUBLIC_IP"),
        help="Public IP advertised by WebRTC; defaults to PUBLIC_IP.",
    )
    parser.add_argument(
        "--dynamic-beans",
        action="store_true",
        help="Enable rigid-body physics for task3 beans in keyboard mode.",
    )
    keyboard_group = parser.add_mutually_exclusive_group()
    keyboard_group.add_argument(
        "--keyboard-control",
        dest="keyboard_control",
        action="store_true",
        default=None,
        help="Run the robot with live WASD/QE keyboard base control.",
    )
    keyboard_group.add_argument(
        "--grasp-friction",
        type=float,
        default=1.2,
        help="static/dynamic friction bound to the gripper pads and the "
             "--publish-object-tf objects; 0 disables. Nothing else in this "
             "scene authors friction for grasping, so contacts otherwise run "
             "on Isaac's ~0.5 default.",
    )
    parser.add_argument(
        "--gripper-stiffness",
        type=float,
        default=None,
        help="override the Robotiq driven-knuckle drive stiffness (default: "
             "robot_actuator_cfg_specs' validated value). The ASSET authors "
             "3.0 with damping 2e-4, an undamped spring that never settles.",
    )
    parser.add_argument(
        "--gripper-damping",
        type=float,
        default=None,
        help="override the driven-knuckle drive damping",
    )
    parser.add_argument(
        "--gripper-max-force",
        type=float,
        default=None,
        help="override the driven-knuckle drive maxForce",
    )
    parser.add_argument(
        "--no-keyboard-control",
        dest="keyboard_control",
        action="store_false",
        help="Load the scene as a passive Isaac Sim viewer.",
    )
    parser.add_argument(
        "--cmd-vel-control",
        action="store_true",
        help=(
            "REV20 P1: drive the base via a /cmd_vel ROS2 subscriber "
            "instead of keyboard input. Requires --ros2-bridge."
        ),
    )
    parser.add_argument(
        "--gripper",
        default=None,
        choices=(None, "robotiq", "panda"),
        help=(
            "Actuator config to use for the gripper group (see "
            "robot_actuator_cfg_specs). Independent of --robot-usd -- "
            "pass both together when loading the real Robotiq asset."
        ),
    )
    parser.add_argument(
        "--autoplay",
        action="store_true",
        help="Start the Isaac Sim timeline immediately after loading.",
    )
    parser.add_argument(
        "--experience",
        choices=tuple(ISAACSIM_EXPERIENCES),
        default="base",
        help="Isaac Sim Kit experience to launch.",
    )
    parser.add_argument(
        "--ros2-bridge",
        choices=("disabled", "fastdds", "cyclonedds"),
        default="disabled",
        help="Enable the Isaac Sim ROS2 bridge with the selected RMW.",
    )
    parser.add_argument(
        "--ros-distro",
        choices=("jazzy", "humble"),
        default="jazzy",
        help="Bundled ROS2 bridge distro to use when ROS2 is enabled.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Launch Isaac Sim without a GUI window.",
    )
    parser.add_argument(
        "--skip-initial-reset",
        action="store_true",
        help=(
            "Skip the initial SimulationContext/InteractiveScene reset. "
            "Useful for scenes whose imported rigid bodies fail during reset."
        ),
    )
    parser.add_argument(
        "--inside-kit",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--gravity-scale",
        type=float,
        default=None,
        help=(
            "Diagnostic only: override gravity magnitude as a multiple of "
            "9.81 m/s^2 (0.0 disables gravity) on the build_stage path, to "
            "test whether a residual joint tracking error is gravity-load "
            "droop. Leaves the default (implicit Isaac Sim gravity) "
            "untouched when omitted."
        ),
    )
    parser.add_argument(
        "--arm-stiffness",
        type=float,
        default=None,
        help=(
            "Diagnostic only: override the arm joints' PhysX drive "
            "stiffness on the build_stage path (default: "
            "robot_actuator_cfg_specs()['arms']['stiffness'])."
        ),
    )
    parser.add_argument(
        "--arm-damping",
        type=float,
        default=None,
        help="Diagnostic only: pair with --arm-stiffness.",
    )
    parser.add_argument(
        "--arm-max-force",
        type=float,
        default=None,
        help=(
            "Diagnostic only: override the arm joints' live PhysX "
            "max_effort via Articulation.set_max_efforts() (only takes "
            "effect with --force-live-gains -- see that flag). Must be "
            "raised in step with --arm-stiffness or a high kp saturates "
            "against this cap and the drive looks stiffness-insensitive "
            "even when it isn't (2026-08-11: an EARLIER version of this "
            "flag's own override went through the disconnected USD path "
            "and its 'barely moved it' result carried no information --"
            " do not trust that claim, it predates the fix)."
        ),
    )
    parser.add_argument(
        "--force-live-gains",
        action="store_true",
        help=(
            "Diagnostic only, DEFAULT OFF: set the arm joints' live PhysX "
            "tensor gains/max-effort via Articulation.set_gains()/"
            "set_max_efforts() -- the API that actually reaches PhysX, "
            "unlike UsdPhysics.DriveAPI writes on this path (GPU-confirmed "
            "no-op, 2026-08-11). PhysX's own auto-computed default "
            "(kps=286478.9, kds=28647.9) is what runs without this flag. "
            "Pair with --arm-stiffness/--arm-damping/--arm-max-force; "
            "forcing gains DOWN to 5000/500 (GPU-tested) made tracking "
            "much worse, consistent with steady-state P-drive droop -- "
            "raising kp ABOVE the default, with max_force raised in step, "
            "is the untested direction that droop model predicts should "
            "help. Off by default so the better-measured PhysX default "
            "runs unless a specific test is being run."
        ),
    )
    parser.add_argument(
        "--log-arm-efforts",
        action="store_true",
        help=(
            "Diagnostic only: periodically log measured joint torque "
            "(Articulation.get_measured_joint_efforts()) and the drive's "
            "own stored setpoint (get_applied_actions()) for the arm "
            "joints, since a real move_group command arrives from an "
            "external ROS2 process well after this script's own main() "
            "returns and can't be timed against a one-shot check."
        ),
    )
    parser.add_argument(
        "--no-extra-graphs",
        action="store_true",
        help=(
            "Skip the odometry and camera OmniGraphs in --cmd-vel-control "
            "mode. Both FAIL to create there, and a half-built graph is the "
            "remaining suspect for why the wheels spin uncommanded at their "
            "velocity limit -- probe_base_drive.py, which drives this base "
            "correctly, builds no OmniGraph at all."
        ),
    )
    parser.add_argument(
        "--ground-collider",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Add a thick ground collider under the room at the measured floor "
            "height. robot_room.usd's floor is 0.5 mm thick, which a 0.05 m "
            "radius wheel tunnels through."
        ),
    )
    parser.add_argument(
        "--floor-friction",
        type=float,
        default=1.0,
        help=(
            "Static/dynamic friction bound to the room, giving the drive "
            "wheels grip. robot_room.usd authors no physics material, and "
            "without this the wheels spin at their velocity limit while the "
            "chassis does not move -- 100%% slip."
        ),
    )
    parser.add_argument(
        "--no-joint-command-graph",
        action="store_true",
        help=(
            "Do not create the /joint_command -> IsaacArticulationController "
            "graph. That node executes on EVERY playback tick, not only when "
            "a message arrives, so it is the last remaining candidate for "
            "what re-applies a zero velocity target to the drive wheels each "
            "frame -- position-controlled joints like the arm would be "
            "unaffected, which matches what is observed."
        ),
    )
    parser.add_argument(
        "--no-base-gain-fix",
        action="store_true",
        help=(
            "Report the live base gains but write nothing. The gain fix runs "
            "AFTER timeline.play(), so set_gains()/set_max_efforts() may be "
            "resetting a drive target that --base-test-spin authored before "
            "play; this separates the two."
        ),
    )
    parser.add_argument(
        "--base-test-spin",
        type=float,
        default=None,
        help=(
            "Diagnostic: author this wheel targetVelocity (rad/s) directly in "
            "USD before play, bypassing ROS, to test whether the base can "
            "drive at all."
        ),
    )
    parser.add_argument(
        "--publish-link-tf",
        nargs="+",
        default=None,
        metavar="LINK_NAME",
        help="Publish these robot link prims' live world poses onto /tf.",
    )
    parser.add_argument(
        "--base-max-force",
        type=float,
        default=None,
        help=(
            "Raise the live max effort on the base's two drive wheels. They "
            "were measured torque-saturated: 0.41 rad/s achieved against a "
            "4.0 rad/s command, the base covering ~0.03 m and stopping in "
            "both directions with a clear /scan. Left alone when unset."
        ),
    )
    parser.add_argument(
        "--drive-damping",
        type=float,
        default=None,
        help=(
            "Live PhysX damping for the two drive wheels on the "
            "--cmd-vel-control path. Unset uses "
            "robot_actuator_cfg_specs()['drive_joints']['damping']. This is "
            "written as a RUNTIME TENSOR WRITE, not left to the "
            "ImplicitActuatorCfg: TmrBaseAdapter (task3_autonomy/skills.py) "
            "already records that the config-path value does not reach the "
            "sim for these joints, and every base-drive script in this repo "
            "that actually moves the base goes through that adapter's "
            "runtime write. This launch path never had one."
        ),
    )
    parser.add_argument(
        "--drive-effort-limit",
        type=float,
        default=None,
        help=(
            "Live PhysX max effort (N.m) for the two drive wheels on the "
            "--cmd-vel-control path. Unset derives a TRACTION-FEASIBLE cap "
            "from the live chassis mass, the bound floor friction and the "
            "wheel radius, instead of the 500 N.m the actuator config "
            "authorises -- 500 N.m through a 0.05 m wheel is 10 kN of rim "
            "force against a contact that can transmit a few hundred, so "
            "the drive breaks traction on the first tick of any command "
            "and limit-cycles instead of accelerating the chassis."
        ),
    )
    parser.add_argument(
        "--drive-armature",
        type=float,
        default=None,
        help=(
            "Live PhysX armature (rotor inertia, kg.m^2) for the two drive "
            "wheels. Unset derives 2 * damping * sim_dt. The asset authors "
            "NO armature, so the solver runs damping 500 against the wheel's "
            "own ~0.003 kg.m^2 at a 5 ms step -- a damping*dt/inertia ratio "
            "of 800, where anything above ~1 oscillates by construction. "
            "Real wheel drives have reflected motor and gearbox inertia; "
            "this is where PhysX takes it. Set 0.0 to reproduce the old "
            "behaviour."
        ),
    )
    parser.add_argument(
        "--steering-effort-limit",
        type=float,
        default=None,
        help=(
            "Live PhysX max effort (N.m) for the two steer axes. Unset "
            "leaves robot_actuator_cfg_specs' 200.0 alone. Test this ONLY "
            "if the steer axes still refuse to track after the caster "
            "rolls are freed: a 1.2 rad error against stiffness 500 "
            "demands 600 N.m and is clipped to 200, so the axis can be "
            "commanding full effort and not moving."
        ),
    )
    parser.add_argument(
        "--steering-stiffness",
        type=float,
        default=None,
        help="Live PhysX stiffness for the two steer axes. Diagnostic only.",
    )
    parser.add_argument(
        "--steering-damping",
        type=float,
        default=None,
        help="Live PhysX damping for the two steer axes. Diagnostic only.",
    )
    parser.add_argument(
        "--drive-module-signs",
        type=str,
        default=None,
        help=(
            "Comma-separated per-module multiplier on the drive wheel "
            "velocity targets, in DRIVE_MODULES order, e.g. '1,-1'. For a "
            "diagonal 2-module base whose modules are mirror images, a "
            "mirrored module's steering zero can point the opposite way, so "
            "both wheels commanded 'forward' drive against each other. "
            "compute_drive_targets has no per-module sign table and cannot "
            "detect this. Unset leaves the mixer's output untouched."
        ),
    )
    parser.add_argument(
        "--no-heading-hold",
        action="store_true",
        help=(
            "Make compensate_yaw_rate a pass-through on the "
            "--cmd-vel-control path, so a published wz of 0.0 really is 0.0. "
            "Without this, heading hold injects up to +/-0.8 rad/s of yaw "
            "correction whenever the caller is not commanding rotation, "
            "which means there has never been a true zero-steering-slew "
            "test on this base."
        ),
    )
    parser.add_argument(
        "--articulation-root",
        choices=("base", "outermost"),
        default="base",
        help=(
            "Which prim keeps UsdPhysics.ArticulationRootAPI. 'base' is this "
            "file's current behaviour: the rigid body named `base`, which "
            "makes PhysX build a FIXED-base articulation anchored to the "
            "world. 'outermost' keeps root_prims[0], the ancestor Xform, "
            "which builds a FLOATING-base one -- and is what "
            "run_episode._fix_single_articulation_root actually selects on "
            "this stage, since it looks up `base`/`base_link` as DIRECT "
            "children and they are nested deeper. probe_base_drive.py, the "
            "one script that drives this base at 0.495 m/s, uses that "
            "version. Check `ROOTKIND is_fixed_base` in the log before "
            "changing this."
        ),
    )
    parser.add_argument(
        "--rocker-stiffness",
        type=float,
        default=625.0,
        help=(
            "Live PhysX stiffness for rocker_arm_joint. Default 625.0 "
            "undoes the degree-to-radian import bug (625 * 180/pi = "
            "35809.86 -- see free_caster_roll_joints' docstring for the "
            "same bug on the casters). GPU-measured 2026-08-13: the raw "
            "35809.86 swings the rocker's own reaction torque to "
            "-200..-310 N.m against millimeter-scale rocker motion -- the "
            "mechanism behind the previously-open 'wheel back-driven "
            "through the articulation, not the ground' stall in "
            "BASE_DRIVE_ROOT_CAUSE_2026-08-13.md. Corrected, reaction "
            "torque stayed single-digit and slip_ratio ran ~0.02-0.24 "
            "instead of ~0.7-0.9 on the same repro. Pass 0.0 instead only "
            "to let the rocker articulate freely, if BASESLIP shows the "
            "drive wheels turning without pulling."
        ),
    )
    parser.add_argument(
        "--rocker-damping",
        type=float,
        default=0.003,
        help="Live PhysX damping for rocker_arm_joint. Default 0.003 is "
             "the matching degree-to-radian correction (0.003 * 180/pi = "
             "0.17189, the asset's raw authored value). Pair with "
             "--rocker-stiffness.",
    )
    parser.add_argument(
        "--caster-roll-damping",
        type=float,
        default=None,
        help=(
            "Live PhysX damping for the two free-castering ROLL joints "
            "(default 0.0 -- they are unpowered rollers and must spin "
            "freely). The asset authors kd=1.0 in USD's DEGREE units, which "
            "PhysX imports as 57.29578 per RADIAN, and nothing on this path "
            "ever read that back: a 0.05 m caster rolling at 0.3 m/s then "
            "meets ~344 N.m of braking and parks against its own effort "
            "limit. Raise this only if the rollers ring, and prefer fixing "
            "their inertials instead."
        ),
    )
    parser.add_argument(
        "--publish-gripper-pad-tf",
        choices=("none", "right", "left"),
        default="none",
        help=(
            "Publish the live world pose of that arm's two Robotiq "
            "`*_inner_finger` pad links onto /tf, alongside --publish-object-tf. "
            "This is the ONLY way to observe where the real fingers are: the "
            "URDF that move_group and robot_state_publisher load "
            "(assets/derived/mobile_fr3_duo_v0_2.urdf) still carries a FRANKA "
            "hand, so its `<side>_fr3v2_hand_tcp` frame is a fiction that no "
            "physical link tracks. One arm at a time on purpose -- "
            "ROS2PublishTransformTree names each frame after the prim, and both "
            "Robotiq subtrees name their fingers `left_inner_finger` / "
            "`right_inner_finger` (Robotiq's own left/right, no arm prefix), so "
            "publishing both arms at once would emit two frames per name."
        ),
    )
    parser.add_argument(
        "--publish-object-tf",
        nargs="+",
        default=None,
        metavar="OBJECT_NAME",
        help=(
            "H5-H9 grasp-and-lift gate: object names (resolved live via "
            "resolve_prim_path, e.g. 'spoon2', 'bowl2') to publish onto "
            "/tf via ROS2PublishTransformTree, one target prim per name. "
            "Without this, no process outside Isaac's own Python can "
            "observe an object's real world pose -- the ROS2 bridge only "
            "carries robot state (joint_states/odom/clock), so a "
            "MoveIt-driven grasp+lift test run from the rclpy sidecar "
            "container has no way to confirm the object actually moved, "
            "which is exactly the gap the earlier standalone grip "
            "diagnostic fell into ('closed on empty air' was "
            "indistinguishable from a real grasp without this). Off by "
            "default -- do not add topics/graphs the current task doesn't "
            "need."
        ),
    )
    return parser.parse_args(argv)


def should_enable_keyboard_control(args: argparse.Namespace) -> bool:
    if args.keyboard_control is not None:
        return bool(args.keyboard_control)
    return args.task == "task3"


def robot_actuator_cfg_specs(
    gripper: str | None = None,
) -> dict[str, dict[str, Any]]:
    """Base/arm/spine joint names are identical on both robots (confirmed:
    P0.2's Robotiq enumeration found unchanged arm/base joint names --
    "rename nothing" applies here too). Only the gripper group differs,
    since the compat robot's simple single-revolute gripper and the real
    Robotiq 2F-85 parallel linkage are structurally different mechanisms.
    `gripper="robotiq"` swaps in the real linkage, confirmed via a live
    GPU scan of authored PhysX drive attributes (not guessed): exactly
    ONE joint per side (`<side>_right_finger_joint`) has an authored
    angular DriveAPI (stiffness=3.0 authored in the USD); the other 10
    (`outer_knuckle_joint`, `*_inner_finger_joint`,
    `*_inner_finger_knuckle_joint`) have none -- pure passive linkage,
    same "drive one, let the rest follow the mechanism" shape as the
    compat robot's own gripper group, just with different real names.
    See plans/REV20_TASKQUEUE.md for the full enumeration this was
    derived from.
    """
    if gripper == "robotiq":
        gripper_actuators = {
            "grippers": {
                "joint_names_expr": [".*_right_finger_joint"],
                "stiffness": 200.0,
                "damping": 20.0,
                "effort_limit_sim": 50.0,
            },
            "passive_gripper_linkage": {
                "joint_names_expr": [
                    ".*_outer_knuckle_joint",
                    ".*_inner_finger_joint",
                    ".*_inner_finger_knuckle_joint",
                ],
                "stiffness": 0.0,
                "damping": 0.0,
            },
        }
    else:
        gripper_actuators = {
            "grippers": {
                # mobile_fr3_duo_v0_2.usd gripper joints:
                # <side>_gripper_joint drives each closed-loop linkage.
                # The remaining linkage joints must stay passive;
                # position-driving every joint fights the mechanism
                # constraints.
                "joint_names_expr": [
                    "left_gripper_joint",
                    "right_gripper_joint",
                ],
                "stiffness": 200.0,
                "damping": 20.0,
                "effort_limit_sim": 50.0,
            },
            "passive_gripper_linkage": {
                "joint_names_expr": [
                    ".*_left_2_joint",
                    ".*_right_1_joint",
                    ".*_right_2_joint",
                    ".*_support_joint",
                ],
                "stiffness": 0.0,
                "damping": 0.0,
            },
        }

    return {
        "steering_joints": {
            "joint_names_expr": ["tmrv0_2_joint_0", "tmrv0_2_joint_2"],
            "stiffness": 500.0,
            "damping": 50.0,
            "effort_limit_sim": 200.0,
        },
        "drive_joints": {
            "joint_names_expr": ["tmrv0_2_joint_1", "tmrv0_2_joint_3"],
            "stiffness": 0.0,
            # 5.0 gave the wheels only ~7% of their velocity target (base
            # crawled at 0.04 m/s for a 0.5 m/s command); 500.0 tracks the
            # target within 2 s. Measured on sim-dev-g4b 2026-07-17 with
            # scripts/task3/probe_base_drive.py --drive-damping 500.
            "damping": 500.0,
            "effort_limit_sim": 500.0,
            "velocity_limit_sim": 20.0,
        },
        "passive_base_joints": {
            "joint_names_expr": [".*caster.*", "rocker_arm_joint"],
            # NOT zeroed, and this is deliberate -- see
            # plans/TOOL_FRAME_ROOT_CAUSE_2026-08-12.md. "Passive" here does
            # not mean "no drive": the asset drives all four caster DOFs and
            # the rocker with ACCELERATION-type drives (caster steering and
            # rocker kp=625 kd=0.003; caster roll kd=1.0), and that is the
            # only thing keeping them conditioned. Their bodies are
            # placeholder-light -- `caster_*_link` is 0.001 kg with inertia
            # 1e-4 and `caster_*_steering_link` is 0.01 kg -- hanging off a
            # 147 kg `base_link`, a mass ratio of ~1.5e5:1. Zeroing the gains
            # leaves a 1 gram wheel free-floating in a ground contact that
            # carries part of 374 kg: measured live, the caster DOFs then hit
            # 18-85 rad/s with the chassis stationary and a ZERO command,
            # which is what was eating the drive wheels' whole traction
            # budget. `None` preserves whatever the asset authored rather
            # than substituting a number of ours, so this stays right if the
            # asset changes.
            "stiffness": None,
            "damping": None,
        },
        "spine": {
            "joint_names_expr": ["franka_spine_vertical_joint"],
            # The spine lifts both FR3 arms; 200 N saturated before moving.
            # Preserve the drive strength authored in the robot USD.
            "stiffness": 50000.0,
            "damping": 5000.0,
            "effort_limit_sim": 500000.0,
        },
        "arms": {
            "joint_names_expr": [".*fr3v2_joint[1-7]"],
            "stiffness": 5000.0,
            "damping": 500.0,
            "effort_limit_sim": 200.0,
        },
        **gripper_actuators,
    }


def resolve_usd_path(selection: Path | None, default_path: Path) -> Path:
    if selection is None:
        return default_path

    candidate = selection.expanduser()
    if candidate.is_absolute() or candidate.is_file():
        return candidate
    return asset_path(*candidate.parts)


def yaw_to_quat(yaw_degrees: float) -> tuple[float, float, float, float]:
    half_yaw = math.radians(yaw_degrees) * 0.5
    return (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw))


def euler_xyz_to_quat(
    rotation_degrees: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    roll, pitch, yaw = (math.radians(angle) for angle in rotation_degrees)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def multiply_quats(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    left_w, left_x, left_y, left_z = left
    right_w, right_x, right_y, right_z = right
    return (
        left_w * right_w
        - left_x * right_x
        - left_y * right_y
        - left_z * right_z,
        left_w * right_x
        + left_x * right_w
        + left_y * right_z
        - left_z * right_y,
        left_w * right_y
        - left_x * right_z
        + left_y * right_w
        + left_z * right_x,
        left_w * right_z
        + left_x * right_y
        - left_y * right_x
        + left_z * right_w,
    )


def axis_angle_to_quat(
    axis: str,
    angle_degrees: float,
) -> tuple[float, float, float, float]:
    half_angle = math.radians(angle_degrees) * 0.5
    real = math.cos(half_angle)
    imaginary = math.sin(half_angle)
    if axis == "x":
        return (real, imaginary, 0.0, 0.0)
    if axis == "y":
        return (real, 0.0, imaginary, 0.0)
    return (real, 0.0, 0.0, imaginary)


def usd_rotate_xyz_to_quat(
    rotation_degrees: tuple[float, float, float],
) -> tuple[float, float, float, float]:
    x_rotation = axis_angle_to_quat("x", rotation_degrees[0])
    y_rotation = axis_angle_to_quat("y", rotation_degrees[1])
    z_rotation = axis_angle_to_quat("z", rotation_degrees[2])
    return multiply_quats(multiply_quats(x_rotation, y_rotation), z_rotation)


def resolve_robot_position(
    args: argparse.Namespace,
) -> tuple[float, float, float]:
    preset_x, preset_y, preset_z = TASK_ROBOT_POSES[args.task]["position"]
    return (
        preset_x if args.robot_x is None else args.robot_x,
        preset_y if args.robot_y is None else args.robot_y,
        preset_z if args.robot_z is None else args.robot_z,
    )


def resolve_robot_yaw(args: argparse.Namespace) -> float:
    if args.robot_yaw is not None:
        return args.robot_yaw
    return TASK_ROBOT_POSES[args.task]["yaw"]


def normalize_head_placement_name(selection: str) -> str:
    normalized = selection.strip().upper()
    if normalized == "RANDOM":
        return "random"
    if normalized in TASK3_HEAD_PLACEMENTS:
        return normalized
    allowed = ", ".join((*TASK3_HEAD_PLACEMENTS, "random"))
    raise ValueError(f"Unknown head placement '{selection}'. Use: {allowed}")


def head_placement_arg(selection: str) -> str:
    try:
        return normalize_head_placement_name(selection)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def resolve_head_placement(
    selection: str,
) -> tuple[str, tuple[float, float, float], tuple[float, float, float, float]]:
    normalized = normalize_head_placement_name(selection)
    if normalized == "random":
        normalized = random.choice(tuple(TASK3_HEAD_PLACEMENTS))

    position, rotation_degrees = TASK3_HEAD_PLACEMENTS[normalized]
    return normalized, position, usd_rotate_xyz_to_quat(rotation_degrees)


def set_head_xform_orient(
    prim: Any,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> None:
    from pxr import Gf as pxr_gf
    from pxr import UsdGeom as pxr_usd_geom

    Gf: Any = pxr_gf
    UsdGeom: Any = pxr_usd_geom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    for rotate_attr_name in (
        "xformOp:rotateXYZ",
        "xformOp:rotateX",
        "xformOp:rotateY",
        "xformOp:rotateZ",
    ):
        rotate_attr = prim.GetAttribute(rotate_attr_name)
        if rotate_attr:
            rotate_attr.Block()

    translate_op = xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    orient_op = xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat)
    translate_op.Set(Gf.Vec3d(*position))
    orient_op.Set(
        Gf.Quatf(
            orientation[0],
            orientation[1],
            orientation[2],
            orientation[3],
        )
    )
    xform.SetXformOpOrder([translate_op, orient_op], True)


def configure_ros2_bridge_env(args: argparse.Namespace) -> None:
    if args.ros2_bridge == "disabled":
        return

    bridge_lib = ROS2_BRIDGE_ROOT / args.ros_distro / "lib"
    if not bridge_lib.is_dir():
        raise FileNotFoundError(
            f"ROS2 bridge library path not found: {bridge_lib}"
        )

    rmw_by_bridge = {
        "fastdds": "rmw_fastrtps_cpp",
        "cyclonedds": "rmw_cyclonedds_cpp",
    }
    os.environ["ROS_DISTRO"] = args.ros_distro
    os.environ["RMW_IMPLEMENTATION"] = rmw_by_bridge[args.ros2_bridge]
    os.environ.setdefault("ROS_LOG_DIR", "/isaac-sim/kit/logs/ros")
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_paths = [str(bridge_lib)]
    if existing_ld_path:
        ld_paths.append(existing_ld_path)
    os.environ["LD_LIBRARY_PATH"] = ":".join(ld_paths)

    if os.environ.get(ROS2_ENV_READY_VAR) != "1":
        env = os.environ.copy()
        env[ROS2_ENV_READY_VAR] = "1"
        os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def enable_ros2_bridge(app: Any, args: argparse.Namespace) -> None:
    if args.ros2_bridge == "disabled":
        return

    import omni.kit.app

    extension_manager = omni.kit.app.get_app().get_extension_manager()
    extension_manager.set_extension_enabled_immediate(
        "isaacsim.ros2.bridge",
        True,
    )
    for _ in range(10):
        app.update()
    print(
        f"ROS2 bridge: {args.ros_distro} / {os.environ['RMW_IMPLEMENTATION']}"
    )


def fix_single_articulation_root(
    stage: Any, robot_prim_path: str, prefer: str = "base"
) -> str:
    """`prefer="outermost"` reproduces run_episode's ACTUAL selection.

    This is not a cosmetic difference between two copies of one helper. The
    two functions disagree about which prim keeps the API, and that decides
    whether PhysX builds a FIXED-base or a FLOATING-base articulation:

      * THIS file walks the whole subtree with `Usd.PrimRange` and keeps the
        prim NAMED `base`, wherever it sits. `base` is a rigid body.
      * `run_episode._fix_single_articulation_root` looks its candidates up by
        FULL PATH -- `{robot_prim_path}/base` and `{robot_prim_path}/base_link`
        -- i.e. DIRECT CHILDREN only. This file's own docstring records that
        the robot is referenced in and those links are NOT direct children
        ("confirmed on real GPU, do not assume shallow nesting again"), so
        both lookups miss, and run_episode falls through to `root_prims[0]`:
        the outermost prim in traversal order, the ancestor Xform.

    Per this file's own note at the top of the original: the API on a RIGID
    BODY builds a fixed-base articulation anchored to the world, on an
    ancestor XFORM a floating-base one. So the two helpers do not merely
    differ, they produce opposite kinds of robot -- and `probe_base_drive.py`,
    the one script that reliably drives this base at 0.495 m/s, is the one
    using run_episode's version.

    The 2026-08-13 revert restored the name-`base` selection here on the
    grounds that the probe "uses run_episode's ORIGINAL
    _fix_single_articulation_root, which leaves the root on `base`". That
    premise is wrong: it leaves the root on `base` only when `base` is a
    direct child, which on this stage it is not.

    Default stays `"base"` so nothing changes unless asked. `robot.is_fixed_base`
    is printed either way, which settles it by measurement rather than by
    reading either docstring.

    Keeps exactly one UsdPhysics.ArticulationRootAPI under the robot.

    mobile_fr3_duo_v0_2.usd carries the API on both .../base and
    .../base_link (nested under `{robot_prim_path}/Asset/...` once
    referenced into the scene, NOT as direct children of
    `robot_prim_path` -- confirmed on real GPU, do not assume shallow
    nesting again) -- PhysX (and any OmniGraph node that queries the
    articulation, e.g. ROS2PublishJointState) refuses ambiguous roots.
    Same fix as scripts/task3/run_episode.py's own
    ``_fix_single_articulation_root`` (kept here too, not imported, since
    that module isn't set up to be imported cross-directory); if this drifts
    from that copy, fix both. Returns the kept prim's path so callers never
    have to re-guess the nesting depth.
    """
    from pxr import Usd, UsdPhysics

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        print(
            "WARNING: articulation-root patch: prim not found "
            f"{robot_prim_path}",
        )
        return robot_prim_path
    root_prims = [
        prim
        for prim in Usd.PrimRange(robot_prim)
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    if not root_prims:
        print(
            f"WARNING: articulation-root patch: no ArticulationRootAPI prim "
            f"found under {robot_prim_path}",
        )
        return robot_prim_path
    keep = None
    if prefer == "base":
        for prim in root_prims:
            if prim.GetName() == "base":
                keep = prim
                break
    if keep is None:
        keep = root_prims[0]
    print(
        f"ROOTCHOICE prefer={prefer} kept={keep.GetPath()} "
        f"candidates={[str(p.GetPath()) for p in root_prims]}",
        flush=True,
    )
    # A FLOATING base, not a fixed one. PhysX decides this from what the
    # ArticulationRootAPI sits on: applied to a RIGID BODY it builds a
    # fixed-base articulation anchored to the world; applied to an ancestor
    # Xform it builds a floating-base one. `base` and `base_link` are both
    # rigid bodies, so keeping the API on either pinned this robot to the
    # world -- which is why the arms and gripper always worked while the
    # chassis never moved a centimetre.
    #
    # That was measured, not assumed. With the wheels switched to position
    # control and 400 N.m of drive available, a 6.0 rad step turned the wheel
    # 0.78 rad and stalled: a wheel with traction cannot roll if the body it
    # carries cannot move. The passive casters beside it spun freely, the
    # laser was clear, and travel was ~0.025 m in EVERY direction tried.
    #
    # Upstream reaches the same arrangement from the other side:
    # `ArticulationCfg(prim_path="{ENV_REGEX_NS}/Robot")` on the
    # `Robotiq_DEMO` branch roots the articulation at the Robot Xform.
    # REVERTED 2026-08-13. This used to relocate the ArticulationRootAPI onto
    # the ancestor Xform, on the theory that keeping it on the `base` RIGID
    # BODY built a world-anchored fixed-base articulation. The zero-gravity
    # test that seemed to confirm it (robot drifting freely) proved only that
    # the body could move, not that the articulation was intact.
    #
    # The evidence that overturns it: `scripts/task3/probe_base_drive.py`
    # drives this base at 0.495 m/s with zero slip, and it uses run_episode's
    # ORIGINAL `_fix_single_articulation_root`, which leaves the root on
    # `base`. Every scene run carrying the relocation has wheels that spin at
    # their velocity limit and that NOTHING brakes -- they stay at -18.6 rad/s
    # after the command goes to zero, which actuator damping of 500 would kill
    # in microseconds. Wheels no actuator can touch are wheels that dropped
    # out of the articulation, which is what re-rooting can do, and the arm
    # kept working because it stayed inside it.
    for prim in root_prims:
        if prim != keep:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
    print(f"Articulation root kept: {keep.GetPath()}")
    return str(keep.GetPath())


def apply_arm_joint_drive_gains(
    stage: Any,
    robot_prim_path: str,
    stiffness: float | None = None,
    damping: float | None = None,
    max_force: float | None = None,
) -> list[str]:
    """`build_stage`'s path drives the robot straight through PhysX (the
    OmniGraph `IsaacArticulationController` chain in
    `publish_ros2_joint_command`, no IsaacLab `Articulation`), so it never
    picks up `robot_actuator_cfg_specs()["arms"]`'s already-validated
    stiffness/damping override -- that dict only takes effect via
    `ImplicitActuatorCfg` on the `InteractiveScene` paths
    (keyboard-control, cmd-vel-control). The robot USD's own authored
    PhysX drive for `.*fr3v2_joint[1-7]` is `stiffness=400, damping=40,
    maxForce=87` (GPU-confirmed via direct `UsdPhysics.DriveAPI` read,
    2026-08-11) -- an order of magnitude weaker than the 5000/500 this
    project already established and validated for these exact joints
    (BLOCKER 1 in `plans/BLOCKED_FOR_OPUS.md`: "the same 'lock the arm'
    values used elsewhere in the repo"). H5-H9 execution trials measured
    a real, GPU-confirmed consequence: `right_fr3v2_joint1` (the shoulder,
    carrying the whole arm's gravity/inertial load) settled 0.147-0.157 rad
    off target under the weak gain, a 14.7cm end-effector Cartesian error
    (measured via `/compute_fk`) -- steady-state P-drive droop scales with
    load/stiffness, and joint1 carries far more load than the wrist
    joints, which tracked fine at the same weak gain. Applying the SAME
    already-validated value here, not inventing a new one.
    """
    from pxr import Usd, UsdPhysics

    arm_spec = robot_actuator_cfg_specs()["arms"]
    joint_pattern = re.compile(arm_spec["joint_names_expr"][0])
    stiffness = arm_spec["stiffness"] if stiffness is None else stiffness
    damping = arm_spec["damping"] if damping is None else damping
    max_force = arm_spec["effort_limit_sim"] if max_force is None else max_force

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        print(f"WARNING: arm drive gains: prim not found {robot_prim_path}")
        return []

    updated_names: list[str] = []
    for prim in Usd.PrimRange(robot_prim):
        if not joint_pattern.fullmatch(prim.GetName()):
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            continue
        drive.GetStiffnessAttr().Set(stiffness)
        drive.GetDampingAttr().Set(damping)
        drive.GetMaxForceAttr().Set(max_force)
        updated_names.append(prim.GetName())
    print(
        f"Arm joint drive gains applied: {len(updated_names)} joints matching "
        f"{joint_pattern.pattern!r} -> stiffness={stiffness}, "
        f"damping={damping}, maxForce={max_force}"
    )
    return updated_names


def apply_gripper_joint_drive_gains(
    stage: Any,
    robot_prim_path: str,
    stiffness: float | None = None,
    damping: float | None = None,
    max_force: float | None = None,
) -> list[str]:
    """Same problem as `apply_arm_joint_drive_gains`, same fix, for the one
    driven Robotiq knuckle per side.

    The Robotiq asset authors that joint as
    `stiffness=3.0, damping=0.0002, maxForce=26.0` (direct
    `UsdPhysics.DriveAPI` read of
    `task1_isaacsim/assets/Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd`).
    Damping of 2e-4 against a real stiffness is effectively an UNDAMPED
    SPRING: the finger oscillates instead of converging, so it can never
    settle on an object and never holds one. GPU-measured 2026-08-12: every
    gripper move -- open or close, any target -- failed to settle within 45 s,
    and successive samples of the same command read 0.44, 0.80, -0.89 and
    -0.94 rad. Those readings were repeatedly misdiagnosed this session as an
    inverted sign convention, as out-of-range clamping, and as a weak drive;
    they are one oscillating joint sampled at different phases.

    Values come from `robot_actuator_cfg_specs(gripper="robotiq")["grippers"]`
    -- the stiffness/damping/effort this repo already validated for these
    exact joints on the `InteractiveScene` paths, which the `build_stage`
    path silently never applies. Not a new number.
    """
    from pxr import Usd, UsdPhysics

    spec = robot_actuator_cfg_specs(gripper="robotiq")["grippers"]
    joint_pattern = re.compile(spec["joint_names_expr"][0])
    stiffness = spec["stiffness"] if stiffness is None else stiffness
    damping = spec["damping"] if damping is None else damping
    max_force = spec["effort_limit_sim"] if max_force is None else max_force

    robot_prim = stage.GetPrimAtPath(robot_prim_path)
    if not robot_prim.IsValid():
        print(f"WARNING: gripper drive gains: prim not found {robot_prim_path}")
        return []

    updated_names: list[str] = []
    for prim in Usd.PrimRange(robot_prim):
        if not joint_pattern.fullmatch(prim.GetName()):
            continue
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            continue
        drive.GetStiffnessAttr().Set(stiffness)
        drive.GetDampingAttr().Set(damping)
        drive.GetMaxForceAttr().Set(max_force)
        updated_names.append(prim.GetName())
    print(
        f"Gripper joint drive gains applied: {len(updated_names)} joints "
        f"matching {joint_pattern.pattern!r} -> stiffness={stiffness}, "
        f"damping={damping}, maxForce={max_force}"
    )
    return updated_names


def apply_grasp_friction(
    stage: Any,
    asset_root_path: str,
    object_names: list[str],
    friction: float = 1.2,
    restitution: float = 0.0,
) -> tuple[int, int]:
    """Bind a high-friction physics material to the Robotiq finger pads and to
    the graspable objects.

    NOTHING in this scene authors friction for grasping. Direct USD reads
    (2026-08-12): the robot asset has no `PhysicsMaterialAPI` anywhere, and
    neither `assets/simple_tray.usd` nor `assets/cup.usd` authors friction or
    mass -- so every grasp contact runs on Isaac's default material, roughly
    0.5/0.5. `apply_physics_material()` has existed in this file the whole
    time but is called exactly once, for Stage 3's coffee beans.

    This matters because the failure mode is RETENTION, not closure: the
    fingers demonstrably reach the object (the driven knuckle stops early at
    a real contact value instead of running to its limit), so the grasp is
    failing after contact, which is what a friction coefficient governs. A
    parallel-jaw grip resists slip by mu * normal_force on two pads; at
    mu=0.5 with the modest normal force this compliant linkage can produce,
    a smooth rigid object slides out.

    1.2 is a normal rubber-pad-on-plastic value and is a MATERIAL property,
    not a scene fit -- it does not encode any object's position and stays
    correct wherever things spawn. The real 2F-85 has rubber pads, so this
    is closer to the hardware than the 0.5 default is.

    Returns (pads_bound, objects_bound) so the caller can see it took effect.
    """
    from pxr import Usd, UsdGeom, UsdShade

    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = create_preview_material(
        stage,
        "/World/Looks/GraspFriction",
        diffuse_color=(0.15, 0.15, 0.15),
        metallic=0.0,
        roughness=0.9,
    )
    apply_physics_material(material, friction=friction, restitution=restitution)

    def _bind(prim: Any) -> bool:
        try:
            binding = UsdShade.MaterialBindingAPI.Apply(prim)
            binding.Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
                materialPurpose="physics",
            )
            return True
        except Exception as exc:  # noqa: BLE001 -- report, never mask
            print(f"WARNING: grasp friction bind failed on {prim.GetPath()}: {exc}")
            return False

    pads = 0
    robot_prim = stage.GetPrimAtPath(asset_root_path)
    if robot_prim and robot_prim.IsValid():
        # The pads are the `<side>_inner_finger` LINK prims -- `Xform`s, not
        # `Gprim`s. An earlier version required `IsA(UsdGeom.Gprim)` and bound
        # 0 pads while silently reporting success on the objects, so the
        # gripper half of the fix never applied. Binding on the link with
        # `strongerThanDescendants` covers its collision geometry below it.
        pad_pattern = re.compile(r".*inner_finger$", re.IGNORECASE)
        for prim in Usd.PrimRange(robot_prim):
            if pad_pattern.match(prim.GetName()):
                if _bind(prim):
                    pads += 1

    objects = 0
    for name in object_names:
        for prim in Usd.PrimRange(stage.GetPseudoRoot()):
            if prim.GetName() == name:
                if _bind(prim):
                    objects += 1
                break

    print(
        f"Grasp friction applied (staticFriction=dynamicFriction={friction}, "
        f"restitution={restitution}): {pads} gripper pad prims, "
        f"{objects} object prims {object_names}"
    )
    return pads, objects


def verify_and_fix_articulation_gains(
    app: Any,
    robot_prim_path: str,
    joint_names: list[str],
    stiffness: float,
    damping: float,
    max_force: float,
    fix: bool = True,
) -> None:
    """Opus's diagnosis, CONFIRMED (2026-08-11). `apply_arm_joint_drive_gains`'s
    `UsdPhysics.DriveAPI` writes never reached PhysX at all: the live
    tensor-backed gains read back as `kps=286478.9, kds=28647.9`
    regardless of every USD value ever written (400, 5000, 10000) --
    confirming the earlier "400->5000 helped, 5000->10000/maxForce
    didn't" narrative was chasing GPU physics noise around an unchanging
    true value (n=1 each, exactly what GOTCHAS' C9 warns against).

    CORRECTION TO AN EARLIER VERSION OF THIS DOCSTRING (Opus's pushback,
    2026-08-11): forcing the live gains DOWN to 5000/500 (a 57x DECREASE
    from 286478.9) made tracking much worse -- that is NOT evidence
    stiffness is a dead lever. It is the expected result of a P-drive
    steady-state-droop model (error ~ gravity_torque / kp) sampled in the
    only direction that model predicts should hurt. The untested
    direction -- kp ABOVE 286478.9 -- has never been tried through this
    (the only working) API. `fix=True` now tests that direction, not a
    return to the disproven 5000/500. Vary `max_force` in step with
    `stiffness` -- a high kp against a capped maxForce saturates the
    drive and looks exactly like "stiffness plateaued" again, the same
    false signal the pre-fix era's maxForce test produced (that test was
    also on the disconnected USD path and carries no information either
    way).

    Reads the LIVE, PhysX-backed gains/max-efforts via `isaacsim.core.
    prims.Articulation.get_gains()`/`get_max_efforts()`/`dof_names` (the
    tensor view, not the USD attribute) unconditionally, for evidence --
    `set_gains()`/`set_max_efforts()` are gated on `fix`. Must run AFTER
    the timeline is playing (the physics tensor view does not exist
    before the first physics step).
    """
    from isaacsim.core.prims import Articulation

    for _ in range(5):
        app.update()

    articulation = Articulation(prim_paths_expr=robot_prim_path)
    articulation.initialize()

    # Opus's other question: does the base tilt under the arm's own load,
    # giving joint1's (kinematically vertical) axis a real gravity
    # component even though a level base would give it none? Cheap to
    # read alongside the gains check, same live articulation.
    base_position, base_orientation = articulation.get_world_poses()
    print(
        f"Base link world pose: position={base_position}, "
        f"orientation(wxyz)={base_orientation}"
    )

    # articulation.get_gains(joint_names=...)'s internal name->index lookup
    # threw IndexError against this robot's DOF layout (GPU-observed,
    # 2026-08-11: "index 39 is out of bounds for axis 1 with size 36") --
    # a real bug in that convenience path, not something to route around
    # blindly. Using dof_names directly for our own index mapping instead.
    dof_names = list(articulation.dof_names)
    print(f"Articulation DOF count={len(dof_names)}, dof_names={dof_names}")
    indices = [dof_names.index(name) for name in joint_names if name in dof_names]
    missing = [name for name in joint_names if name not in dof_names]
    if missing:
        print(f"WARNING: requested joint names not in dof_names: {missing}")

    all_kps, all_kds = articulation.get_gains()
    all_max_efforts = articulation.get_max_efforts()
    before_kps = [float(all_kps[0][i]) for i in indices]
    before_kds = [float(all_kds[0][i]) for i in indices]
    before_max_efforts = [float(all_max_efforts[0][i]) for i in indices]
    print(
        f"Live articulation gains BEFORE correction (joints={[dof_names[i] for i in indices]}): "
        f"kps={before_kps}, kds={before_kds}, max_efforts={before_max_efforts}"
    )

    mismatch = any(abs(k - stiffness) > 1.0 for k in before_kps)
    if mismatch and not fix:
        print(
            "MISMATCH: live gains disagree with the USD write -- "
            "--force-live-gains not set, leaving PhysX's own values in place."
        )
    elif mismatch:
        print(
            "MISMATCH: live gains disagree with the USD write -- "
            "setting directly via Articulation.set_gains()/set_max_efforts()."
        )
        target_kps = list(all_kps[0])
        target_kds = list(all_kds[0])
        target_max_efforts = list(all_max_efforts[0])
        for i in indices:
            target_kps[i] = stiffness
            target_kds[i] = damping
            target_max_efforts[i] = max_force
        articulation.set_gains(kps=[target_kps], kds=[target_kds])
        articulation.set_max_efforts(values=[target_max_efforts])
        after_kps, after_kds = articulation.get_gains()
        after_max_efforts = articulation.get_max_efforts()
        print(
            f"Live articulation gains AFTER correction: "
            f"kps={[float(after_kps[0][i]) for i in indices]}, "
            f"kds={[float(after_kds[0][i]) for i in indices]}, "
            f"max_efforts={[float(after_max_efforts[0][i]) for i in indices]}"
        )
    else:
        print("Live gains already match the requested value -- no correction needed.")


def split_gripper_dofs(dof_names: list[str]) -> tuple[list[str], list[str]]:
    """Split the Robotiq DOFs into the one DRIVEN joint per side and the
    passive followers of the closed 4-bar linkage.

    Patterns mirror `robot_actuator_cfg_specs(gripper="robotiq")` exactly;
    the suffixes are distinct, `*_right_finger_joint` (driven) never also
    matching `*_inner_finger_joint` (passive).
    """
    driven = [n for n in dof_names if n.endswith("_right_finger_joint")]
    passive = [
        n for n in dof_names
        if n.endswith(("_outer_knuckle_joint", "_inner_finger_joint",
                       "_inner_finger_knuckle_joint"))
    ]
    return driven, passive


def verify_and_fix_gripper_gains(
    app: Any,
    robot_prim_path: str,
    stiffness: float,
    damping: float,
    max_force: float,
) -> None:
    """The gripper half of the gains fix that `verify_and_fix_articulation_gains`
    has only ever done for the arms.

    `apply_gripper_joint_drive_gains` writes `UsdPhysics.DriveAPI` and prints
    "stiffness=20.0" from the USD attribute it just wrote. That write is the
    SAME one already proven never to reach PhysX for the arm joints -- the
    live tensor-backed value read back as 286478.9 no matter what the USD
    said. Nobody had ever read the gripper's live gains, so the printed
    confirmation was reporting the USD file back to itself.

    Measured consequence (2026-08-12): sweeping the driven joint across its
    full authored range in 13 steps moved it not at all -- commanded 0.8 down
    to 0.0, measured pinned at -0.396 rad every time, real pad separation
    constant to within 0.2 mm. A closed 4-bar linkage whose ten PASSIVE
    joints are each being position-servoed at kp=286478 cannot move: the
    followers hold their own targets far harder than the single driven joint
    can pull. Passive means passive, so their drives are zeroed here.
    """
    from isaacsim.core.prims import Articulation

    for _ in range(5):
        app.update()
    articulation = Articulation(prim_paths_expr=robot_prim_path)
    articulation.initialize()
    dof_names = list(articulation.dof_names)
    driven, passive = split_gripper_dofs(dof_names)
    if not driven:
        print("WARNING: no driven gripper DOF found -- gripper gains not set")
        return

    kps, kds = articulation.get_gains()
    max_efforts = articulation.get_max_efforts()
    idx = {n: dof_names.index(n) for n in driven + passive}
    print(
        "Live GRIPPER gains BEFORE correction: "
        + ", ".join(
            f"{n}: kp={float(kps[0][i]):.1f} kd={float(kds[0][i]):.1f} "
            f"maxF={float(max_efforts[0][i]):.1f}"
            for n, i in idx.items()
        )
    )

    # Write ONLY the gripper DOFs, via joint_indices. Passing full-width
    # arrays instead round-trips every other joint's gains through this call,
    # and the base's drive wheels do not survive that: they are VELOCITY
    # joints (authored stiffness=0, damping=100000), and after a full-width
    # rewrite a base drive that reliably covered 0.2318 m managed 0.0026 m.
    # Nothing else in this scene should be touched by a gripper fix.
    order = driven + passive
    joint_indices = [idx[n] for n in order]
    tgt_kps = [stiffness if n in driven else 0.0 for n in order]
    tgt_kds = [damping if n in driven else 0.0 for n in order]
    tgt_max = [max_force if n in driven else 0.0 for n in order]
    articulation.set_gains(kps=[tgt_kps], kds=[tgt_kds],
                           joint_indices=joint_indices)
    articulation.set_max_efforts(values=[tgt_max], joint_indices=joint_indices)

    kps, kds = articulation.get_gains()
    max_efforts = articulation.get_max_efforts()
    print(
        "Live GRIPPER gains AFTER correction: "
        + ", ".join(
            f"{n}: kp={float(kps[0][i]):.1f} kd={float(kds[0][i]):.1f} "
            f"maxF={float(max_efforts[0][i]):.1f}"
            for n, i in idx.items()
        )
    )


def add_room_ground_collider(
    stage: Any, room_prim_path: str, thickness: float = 1.0
) -> float | None:
    """Put a THICK ground collider under the room, at the room floor's own height.

    `assets/robot_room.usd`'s floor is `Rectangle491`: 25 m x 25 m and
    **0.5 mm thick** (world z spans -0.0034 to -0.0029). A wheel of radius
    0.05 m rolling on a half-millimetre collider is the textbook PhysX
    tunnelling case -- the contact is found and lost between substeps, and
    depenetration spins the wheel. That is what the base has been doing: the
    drive wheel sits pinned at -20.0 rad/s, its velocity limit, in the wrong
    direction, even at idle against a zero target with actuator damping 500.

    `scripts/task3/probe_base_drive.py` drives the identical robot at
    0.495 m/s with zero slip, and the only relevant difference is that it
    runs on a `GroundPlaneCfg` -- a proper infinite half-space instead of a
    paper-thin slab.

    The height is MEASURED from the room's own broadest prim rather than
    assumed, so this stays correct if the room asset moves or is replaced.
    Returns the floor top z it used, or None if it could not find a floor.
    """
    from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

    room_prim = stage.GetPrimAtPath(room_prim_path)
    if not (room_prim and room_prim.IsValid()):
        print(f"WARNING: ground collider: no room prim at {room_prim_path}")
        return None

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
    )
    # Collect EVERY floor-like candidate and whether it actually collides.
    # The original version ranked purely on width and picked `Rectangle491`
    # (15.0 m wide, top -0.0029) -- which carries NO collider at all. The
    # room's only collidable floor under the robot spawn is `Rectangle014`
    # (top -0.0024, 1.0 mm thick). So this function was placing its thick
    # slab 0.5 mm BELOW the real collidable floor, where the wheels could
    # never reach it: the tunnelling fix was inert, and the wheels have been
    # riding the 1.0 mm slab this whole time. Measured 2026-08-13 with
    # scripts/task3/probe_base_collider_extents.py --room.
    candidates = []  # (has_collider, width, top_z, path)
    for prim in Usd.PrimRange(room_prim):
        if not prim.IsA(UsdGeom.Boundable):
            continue
        try:
            rng = cache.ComputeWorldBound(prim).ComputeAlignedRange()
        except Exception:  # noqa: BLE001 -- prims without a bound are not floors
            continue
        if rng.IsEmpty():
            continue
        lo, hi = rng.GetMin(), rng.GetMax()
        width = min(hi[0] - lo[0], hi[1] - lo[1])
        # Broad in BOTH horizontal axes and thin vertically: that is a floor,
        # not a wall (walls are broad in one axis and 3 m tall).
        if width < 5.0 or (hi[2] - lo[2]) > 0.5:
            continue
        candidates.append(
            (
                bool(prim.HasAPI(UsdPhysics.CollisionAPI)),
                float(width),
                float(hi[2]),
                prim,
            )
        )
    for has_collider, width, cand_top, prim in candidates:
        print(
            f"Ground collider candidate: {prim.GetPath()} width={width:.1f} m "
            f"top_z={cand_top:.4f} collider={'YES' if has_collider else 'no'}"
        )

    # A floor the robot cannot touch is not the floor, so collidable prims win.
    # Among those the HIGHEST surface is the one the wheels actually land on
    # (a rug beats the slab under it) -- but "highest" alone would happily
    # select a collidable CEILING: this room has three broad thin prims at
    # z=3.05-3.20 that are only excluded today because they carry no collider.
    # Restrict to the lower half of the room's own vertical extent, which is
    # derived from the asset rather than assumed.
    collidable = [c for c in candidates if c[0]]
    room_rng = cache.ComputeWorldBound(room_prim).ComputeAlignedRange()
    mid_z = 0.5 * (float(room_rng.GetMin()[2]) + float(room_rng.GetMax()[2]))
    ranked = [c for c in collidable if c[2] <= mid_z] or collidable or candidates
    if not ranked:
        print("WARNING: ground collider: no floor-like prim found")
        return None
    if not collidable:
        print("WARNING: ground collider: no COLLIDABLE floor-like prim found; "
              "falling back to the broadest non-collidable one")
    _, width, top_z, _ = max(ranked, key=lambda c: (c[2], c[1]))
    path = f"{room_prim_path}/GroundCollider"
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    half = thickness / 2.0
    xform = UsdGeom.Xformable(cube.GetPrim())
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, top_z - half))
    xform.AddScaleOp().Set(Gf.Vec3f(200.0, 200.0, float(thickness)))
    UsdPhysics.CollisionAPI.Apply(cube.GetPrim())
    PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
    UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
    print(f"Ground collider added at z={top_z:.4f} (room floor top, measured "
          f"from a {width:.1f} m wide prim), {thickness} m thick")

    # Retire the paper-thin floor colliders this slab replaces. Leaving them
    # enabled at the SAME height means the wheel can still land on a 1.0 mm
    # slab and tunnel through it between substeps -- which is the entire
    # failure this function exists to prevent, and adding a thick collider
    # underneath does not stop it while the thin one is still the first thing
    # the wheel meets. The thick slab spans the same 200 x 200 m at the same
    # top z, so nothing loses support.
    retired = 0
    for has_collider, _w, cand_top, prim in candidates:
        if not has_collider or cand_top > top_z:
            continue
        UsdPhysics.CollisionAPI(prim).CreateCollisionEnabledAttr(False)
        retired += 1
        print(f"Thin floor collider disabled: {prim.GetPath()} "
              f"(top_z={cand_top:.4f}), superseded by GroundCollider")
    if not retired:
        print("Thin floor colliders: none to retire")
    return top_z


def apply_floor_friction(
    stage: Any, room_prim_path: str, friction: float, restitution: float = 0.0
) -> int:
    """Bind a friction material to the room, so the drive wheels have grip.

    THE reason the base never moved. `scripts/task3/probe_base_drive.py`
    drives this exact robot at 0.495 m/s with its wheels tracking 9.9 of a
    10.0 rad/s target -- perfect rolling, 9.9 rad/s x 0.05 m radius = 0.495
    m/s, zero slip. The probe runs on a plain `GroundPlaneCfg`. In the room
    scene the same wheels spin at their velocity limit (-19.6 and +3.7 rad/s
    were logged while the commanded target was 0.0) and the chassis does not
    translate at all: 100% slip.

    Nothing in `assets/robot_room.usd` authors a physics material, so its
    floor runs on whatever default the collider gets, and the wheels have
    nothing to push against. Every drive-side explanation chased before this
    -- gains, joint friction, velocity limits, the ROS command path, the
    articulation root, spawn height -- was looking at the wrong half of the
    contact.

    Bound with `strongerThanDescendants` on the room root so it covers the
    floor mesh however that asset happens to nest it, the same way
    `apply_grasp_friction` covers the finger pads.
    """
    from pxr import Usd, UsdPhysics, UsdShade

    room_prim = stage.GetPrimAtPath(room_prim_path)
    if not (room_prim and room_prim.IsValid()):
        print(f"WARNING: floor friction: no room prim at {room_prim_path}")
        return 0

    material_path = f"{room_prim_path}/FloorPhysicsMaterial"
    material = UsdShade.Material.Define(stage, material_path)
    physics_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_api.CreateStaticFrictionAttr(friction)
    physics_api.CreateDynamicFrictionAttr(friction)
    physics_api.CreateRestitutionAttr(restitution)

    bound = 0
    try:
        binding = UsdShade.MaterialBindingAPI.Apply(room_prim)
        binding.Bind(
            material,
            bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            materialPurpose="physics",
        )
        bound = 1
    except Exception as exc:  # noqa: BLE001 -- report, never mask
        print(f"WARNING: floor friction bind failed: {exc}")
    print(f"Floor friction applied: staticFriction=dynamicFriction={friction} "
          f"on {room_prim_path} ({bound} binding)")
    return bound


def clear_base_joint_friction(stage: Any, asset_root_path: str) -> int:
    """Zero `physxJoint:jointFriction` on the base's steering and drive joints.

    THE reason the base never moved. Authored on this asset:

        tmrv0_2_joint_1 / _3  (drive)     jointFriction = 1.0
        tmrv0_2_joint_0 / _2  (steering)  jointFriction = 5.0
        caster_*_joint, caster_*_steering_joint, rocker_arm_joint  = 0.0

    The only joints carrying joint friction are the powered ones, and they are
    exactly the joints that would not turn; the zero-friction casters beside
    them spun freely the whole time. PhysX joint friction resists as a
    coefficient times the joint's own constraint force, so a wheel carrying
    the robot's weight through its axle sees a resisting torque far larger
    than any drive we can command -- while the same wheel unloaded turns
    slowly rather than not at all. That is precisely what was measured: in
    ZERO GRAVITY with no contact, 6.0 rad/s commanded with 200 N.m of drive
    produced 1.33 rad in 15 s, and under gravity a +6.0 rad position step
    with 400 N.m available moved the wheel 0.78 rad before stalling.

    This is a USD write applied BEFORE the timeline plays, which is where
    joint friction is read. It is not the drive-gain path that was proven not
    to reach PhysX -- that one is overridden at articulation init, and joint
    friction has no such override.
    """
    from pxr import Usd

    robot_prim = stage.GetPrimAtPath(asset_root_path)
    if not (robot_prim and robot_prim.IsValid()):
        print(f"WARNING: joint friction: no robot prim at {asset_root_path}")
        return 0
    cleared = 0
    for prim in Usd.PrimRange(robot_prim):
        if not prim.GetName().startswith("tmrv0_2_joint_"):
            continue
        attr = prim.GetAttribute("physxJoint:jointFriction")
        if not attr:
            continue
        before = attr.Get()
        if before in (None, 0.0):
            continue
        attr.Set(0.0)
        cleared += 1
        print(f"Base joint friction cleared: {prim.GetName()} {before} -> 0.0")
    if not cleared:
        print("Base joint friction: nothing to clear")
    return cleared


PLACEHOLDER_INERTIA_DIAG = (1.0e-4, 1.0e-4, 1.0e-4)


def derive_placeholder_link_inertials(
    stage: Any, asset_root_path: str
) -> int:
    """Let PhysX compute mass/inertia for links exported with a placeholder.

    The base's two caster wheels are authored `physics:mass = 0.001` (ONE
    GRAM) with `physics:diagonalInertia = (1e-4, 1e-4, 1e-4)`, and their
    steering links 0.01 kg with the same inertia triple. That triple is the
    URDF exporter's placeholder, not a measurement -- the same asset gives
    the drive wheel beside them 2.5 kg / 0.0019, and PhysX's own live figure
    for `base_link` is 147 kg. A 1 g wheel in a ground contact that carries
    part of a 374 kg robot is a ~1.5e5:1 mass ratio inside one articulation,
    which no solver can condition: measured live 2026-08-13, the caster DOFs
    ring at 13-85 rad/s with the chassis stationary and a ZERO command, the
    drive wheels' torque spikes to their +/-500 N.m limit against a zero
    target, and the chassis wanders 0.35 m and yaws 22 deg in 29 sim seconds
    with nothing commanding it.

    The fix is the asset's OWN convention, not a number of ours: `base_link`
    is authored `mass = 0.0`, `diagonalInertia = (0, 0, 0)` and PhysX derives
    147 kg from its collision geometry. Authoring the same thing on the
    placeholder links makes PhysX derive theirs the same way, so the value
    tracks the geometry and stays right if the asset changes. Only links that
    actually own collision geometry are touched -- a pure frame has nothing
    to derive from.
    """
    from pxr import Gf, Usd, UsdPhysics

    robot_prim = stage.GetPrimAtPath(asset_root_path)
    if not (robot_prim and robot_prim.IsValid()):
        print(f"WARNING: link inertials: no robot prim at {asset_root_path}")
        return 0

    fixed = 0
    blocked = 0
    seen = 0
    for prim in Usd.PrimRange(robot_prim, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        seen += 1
        if prim.IsInstanceProxy():
            # Say so instead of falling through to "none found", which reads
            # as "the asset is fine" and is exactly the kind of quietly-wrong
            # log line that has cost this code path three sessions already.
            blocked += 1
            continue
        inertia_attr = prim.GetAttribute("physics:diagonalInertia")
        if not (inertia_attr and inertia_attr.HasAuthoredValue()):
            continue
        diag = inertia_attr.Get()
        if diag is None:
            continue
        mass_attr_check = prim.GetAttribute("physics:mass")
        mass_val = float(mass_attr_check.Get()) if (
            mass_attr_check and mass_attr_check.HasAuthoredValue()
        ) else None
        # Loosened 2026-08-13: exact-tuple equality against
        # PLACEHOLDER_INERTIA_DIAG was measured against a DIFFERENT source
        # USD (the default Franka-hand mobile_fr3_duo_v0_2.usd) than the
        # Robotiq asset this launch actually loads
        # (Robotiq_2f_85_with_d405_mobile_fr3_duo_v0_2.usd) -- the exporter's
        # placeholder values are the same CONCEPT (near-zero, order 1e-4) but
        # not necessarily the same exact floats in a different export. GPU-
        # measured on the Robotiq asset: this exact-match left `fixed=0` out
        # of 86 rigid bodies while the caster-ringing symptom the function's
        # own docstring describes (wheel_tau spiking to +/-500 N.m against a
        # zero target, chassis yawing 25 deg in a few seconds with nothing
        # commanding it) was reproduced live. Switched to the same threshold
        # the docstring already states in prose (placeholder mass is
        # 0.001-0.01 kg vs. the real drive wheel's 2.5 kg) instead of an
        # exact-float match against one specific asset's export.
        is_placeholder_scale = (
            mass_val is not None and 0.0 < mass_val < 0.1
            and all(abs(float(v)) < 1.0e-2 for v in diag)
        )
        if not is_placeholder_scale:
            continue
        has_collider = any(
            child.HasAPI(UsdPhysics.CollisionAPI)
            for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
        )
        if not has_collider:
            continue
        mass_attr = prim.GetAttribute("physics:mass")
        before = float(mass_attr.Get()) if mass_attr else None
        if mass_attr:
            mass_attr.Set(0.0)
        inertia_attr.Set(Gf.Vec3f(0.0, 0.0, 0.0))
        fixed += 1
        print(
            f"Placeholder inertial cleared: {prim.GetName()} "
            f"mass={before} inertia={tuple(diag)} -> derived by PhysX",
            flush=True,
        )
    if blocked:
        print(
            f"WARNING: placeholder inertials: {blocked} rigid bodies are "
            "INSTANCE PROXIES and cannot be authored here -- this function is "
            "a NO-OP on this launch path. The caster links really do carry a "
            "1 g / 1e-4 placeholder inertial (measured, see "
            "plans/TOOL_FRAME_ROOT_CAUSE_2026-08-12.md); fixing it needs a "
            "write inside make_headless_robot_usd, or a live "
            "root_physx_view.set_masses/set_inertias after sim.reset().",
            flush=True,
        )
    elif not fixed:
        # `seen` is the point: "none found" out of ZERO inspected bodies means
        # the traversal never reached the robot's rigid bodies at all (they
        # live behind an instanceable reference), NOT that the asset is clean.
        # The caster links really do carry a 1 g / 1e-4 placeholder inertial.
        print(
            f"Placeholder inertials: none authorable "
            f"({seen} rigid bodies inspected)"
            + ("" if seen else " -- ZERO inspected, so this says nothing "
                               "about the asset; see "
                               "plans/TOOL_FRAME_ROOT_CAUSE_2026-08-12.md"),
            flush=True,
        )
    return fixed


def disable_articulation_sleeping(
    stage: Any, articulation_root_path: str
) -> bool:
    """Stop PhysX from putting the robot's articulation to sleep.

    PhysX puts an articulation whose mass-normalised kinetic energy stays
    below `sleepThreshold` to sleep and then stops simulating it. A sleeping
    articulation is NOT woken by writing a joint drive target -- and a joint
    drive target is the only thing `--cmd-vel-control` ever writes. So the
    base goes to sleep the moment it finishes settling and then ignores
    /cmd_vel completely, forever.

    Measured 2026-08-13, and this is the signature to recognise: with
    `wheel_tgt=[5.997, 6.003]` commanded and the steering tracking its target
    exactly, `wheel_vel=[-0.016, -0.049]` and `base_xy` held at
    [-3.109, -1.7904] BIT-IDENTICALLY for 30 s while sim time advanced 3.1 s.
    Every DOF frozen to five decimals -- including the purely passive casters
    and the rocker, which cannot be "commanded to zero" by anything -- while
    the clock runs is only produced by the body being excluded from
    simulation, never by a mechanical jam or a contact problem.

    This was MASKED until now: with the base joints' authored friction still
    live (see `clear_base_joint_friction`, which this launch path never
    called) the robot stick-slipped continuously and never sat still long
    enough to fall asleep. Fixing the friction is what let it settle, which
    is what let it sleep. That is why the two fixes only work together, and
    why fixing friction alone measured WORSE than no fix at all.

    Written before `timeline.play()`/`sim.reset()`, where PhysX reads it,
    exactly like `clear_base_joint_friction`. Applying `PhysxArticulationAPI`
    to a prim that already carries it is a no-op, and this touches only the
    sleep/stabilization thresholds -- it does NOT move `ArticulationRootAPI`,
    which is a hard rule.
    """
    from pxr import PhysxSchema

    prim = stage.GetPrimAtPath(articulation_root_path)
    if not (prim and prim.IsValid()):
        print(
            "WARNING: sleep threshold: no prim at "
            f"{articulation_root_path}"
        )
        return False
    api = PhysxSchema.PhysxArticulationAPI.Apply(prim)
    attr = api.GetSleepThresholdAttr()
    before = attr.Get() if attr else None
    api.CreateSleepThresholdAttr().Set(0.0)
    # Deliberately NOT touching stabilizationThreshold. Zeroing it as well
    # was tried and reverted: stabilization is what keeps a low-energy
    # articulation's contacts quiet, and this base's wheel contacts are
    # already marginal (see the ride-height finding), so disabling it trades
    # one problem for another. Sleeping is the bug; stabilization is not.
    print(
        f"Articulation sleepThreshold {before} -> 0.0 at "
        f"{articulation_root_path}",
        flush=True,
    )
    return True


def set_base_drive_target_velocity(
    stage: Any, asset_root_path: str, rad_per_s: float
) -> int:
    """Write `drive:angular:physics:targetVelocity` straight onto the two drive
    wheels, before play.

    This is a TEST, and a decisive one. Everything commanding the wheels so far
    has gone through /joint_command -> ROS2SubscribeJointState ->
    IsaacArticulationController, and nothing has ever confirmed that chain
    actually writes a VELOCITY target -- the arm, which does work through it,
    is position-controlled. Both wheels author `targetVelocity = 0.0`, so if
    the velocity never arrives, the authored zero plus the drive's damping is
    a brake, and that alone would explain a base that will not move in any
    direction under any gain, friction or spawn height.

    Setting it here bypasses ROS entirely. If the robot drives, the physics is
    sound and the ROS velocity channel is the bug. If it still does not, the
    block is in the articulation and ROS is exonerated.
    """
    from pxr import Usd

    robot_prim = stage.GetPrimAtPath(asset_root_path)
    if not (robot_prim and robot_prim.IsValid()):
        return 0
    count = 0
    for prim in Usd.PrimRange(robot_prim):
        if prim.GetName() not in ("tmrv0_2_joint_1", "tmrv0_2_joint_3"):
            continue
        attr = prim.GetAttribute("drive:angular:physics:targetVelocity")
        if not attr:
            continue
        print(f"Base drive targetVelocity: {prim.GetName()} "
              f"{attr.Get()} -> {rad_per_s}")
        attr.Set(rad_per_s)
        count += 1
    return count


def apply_drive_wheel_authority(
    robot: Any,
    drive_indices: list[int],
    *,
    floor_friction: float,
    damping: float | None = None,
    effort_limit: float | None = None,
    armature: float | None = None,
    sim_dt: float = 0.005,
) -> None:
    """Give the two drive wheels a velocity servo they can actually satisfy.

    Two separate defects, both live on the --cmd-vel-control path only, both
    fixed here by RUNTIME TENSOR WRITES rather than by config:

    1. **The actuator config's damping never reaches these joints.**
       `robot_actuator_cfg_specs()["drive_joints"]` asks for damping 500, but
       `TmrBaseAdapter.DRIVE_DAMPING` (task3_autonomy/skills.py) already
       records that "the same value in robot_actuator_cfg_specs did NOT reach
       the sim (the two live runs crawled identically before and after that
       config change)" and writes it through `write_joint_damping_to_sim`
       instead. Every script in this repo that has ever moved this base --
       probe_base_drive.py (0.495 m/s, zero slip), verify_navigate.py,
       run_stage1_setup.py -- goes through that adapter. THIS launch path
       does not use the adapter, calls `compute_drive_targets` directly, and
       so has been running on the USD-authored `damping=100000` the whole
       time. That is the same authored value the three comments in this file
       already name; nothing on this path ever overwrote it.

    2. **The effort limit is far outside the friction cone.** 500 N.m through
       a 0.05 m wheel is 10 kN of rim force. The contact under that wheel can
       transmit `mu * N`, a few hundred newtons. A velocity servo authorised
       to demand 20x what the tyre can pass does not accelerate the chassis:
       it breaks traction on the first tick, the wheel free-spins toward its
       velocity limit, the servo reverses, and the pair limit-cycles. The
       recorded signature is exactly that -- torque pinned at +/-500 with the
       wheel velocity spiking and collapsing, and, decisively, +/-500 N.m
       appearing on ticks where the commanded target is ZERO and the wheel is
       drifting at 1.9 rad/s. A servo that saturates against a zero target is
       unstable on its own terms; no amount of floor friction repairs it,
       and more friction only couples the hammering harder into the chassis
       (see the ringing and tip-over reports).

    The cap is DERIVED, not fitted: half the robot's live mass rides on the
    two drive contacts (2 of 4 ground contacts -- the other two are casters),
    so each wheel carries `M*g/4`, can pass `mu * M*g/4` of traction, and is
    given 80% of that so the drive stays strictly inside the friction cone.
    Change the floor friction and this moves with it. For scale, driving
    147 kg at 0.3 m/s^2 against 5% rolling resistance needs ~3 N.m per wheel,
    so the derived cap keeps several times the authority actually required.
    """
    import torch

    from tmr_base_control import WHEEL_RADIUS_M

    def _live(attr: str) -> list[float] | None:
        tensor = getattr(robot.data, attr, None)
        if tensor is None:
            return None
        return [round(float(tensor[0][i]), 4) for i in drive_indices]

    if damping is None:
        damping = float(robot_actuator_cfg_specs()["drive_joints"]["damping"])

    if armature is None:
        # THE DRIVE JOINTS HAVE NO ROTOR INERTIA. `BASEGAINS drive` reads
        # armature=[0.0, 0.0] live, so the solver sees damping 500 acting on
        # nothing but the wheel's own ~0.003 kg.m^2 (2.5 kg, r=0.05, as a
        # disc). The resulting gain-to-inertia ratio is
        # `damping * dt / I` = 500 * 0.005 / 0.003 = 800. A ratio above ~1 is
        # a limit cycle by construction: one tick of drive torque changes the
        # wheel velocity by more than the error that produced it, so the servo
        # overshoots, reverses, and oscillates forever instead of settling.
        # Measured, that is exactly what the wheels do -- 0.5 to 2.1 rad/s of
        # drift with a zero target, torque swinging over hundreds of N.m, and
        # a chassis that jitters +/-0.03 m per sample while netting 0.06 m in
        # 40 s.
        #
        # It also explains why capping the effort limit alone did not help:
        # even at the traction-feasible 36.74 N.m, one tick still swings the
        # wheel by 58 rad/s. The cap bounds how hard the oscillation hits the
        # floor, not whether it oscillates.
        #
        # A real wheel drive HAS rotor inertia -- a motor and gearbox
        # reflected through the reduction -- and this asset simply never
        # authored it. Armature is where PhysX takes that term, and adding it
        # is the standard remedy for exactly this signature on light wheels at
        # a 5 ms step. Sized at 2 * damping * dt, which puts the ratio at 0.5,
        # comfortably inside the stable region and derived from the two
        # numbers that set the problem rather than fitted to a run.
        armature = 2.0 * damping * sim_dt

    if effort_limit is None:
        # apply_floor_friction binds static=dynamic=--floor-friction. Unbound
        # (0.0) means PhysX's default material, whose friction is 0.5.
        mu = floor_friction if floor_friction > 0.0 else 0.5
        total_mass_kg = float(robot.data.default_mass[0].sum())
        normal_per_drive_wheel_n = total_mass_kg * 9.81 * 0.5 / 2.0
        traction_n = mu * normal_per_drive_wheel_n
        effort_limit = max(5.0, 0.8 * traction_n * WHEEL_RADIUS_M)

    # THERE ARE TWO EFFORT LIMITS, and only one of them is the one PhysX
    # enforces. IsaacLab 2.x splits them: `effort_limit` is the ACTUATOR
    # MODEL's clip, applied in Python inside the actuator's compute(), while
    # `effort_limit_sim` is the PhysX DOF max force. `robot_actuator_cfg_specs`
    # already spells its key `effort_limit_sim`, which is the tell that this
    # build makes the distinction. Writing one and reading it back proves
    # nothing about the other, and on 2026-08-13 that is exactly what
    # happened: a confirmed readback of 36.74 while `applied_torque` kept
    # spiking to -500, i.e. the cap was written somewhere the solver does not
    # consult. So write EVERY writer this build exposes, poke the actuator
    # objects' own tensors directly, and read back EVERY candidate field.
    # Uncertainty about which name is authoritative is not worth another GPU
    # round trip when doing all of them costs nothing.
    candidate_fields = (
        "joint_effort_limits_sim",
        "joint_effort_limits",
        "joint_max_effort",
    )

    def _live_all() -> dict[str, list[float]]:
        return {
            field: values
            for field in candidate_fields
            if (values := _live(field)) is not None
        }

    before_damping = _live("joint_damping")
    before_effort = _live_all()

    robot.write_joint_damping_to_sim(
        torch.full(
            (robot.num_instances, len(drive_indices)),
            damping,
            device=robot.device,
        ),
        joint_ids=drive_indices,
    )

    effort_tensor = torch.full(
        (robot.num_instances, len(drive_indices)),
        effort_limit,
        device=robot.device,
    )
    writers_used = []
    for name in (
        "write_joint_effort_limit_to_sim",
        "write_joint_effort_limits_to_sim",
        "write_joint_max_effort_to_sim",
    ):
        writer = getattr(robot, name, None)
        if writer is None:
            continue
        try:
            writer(effort_tensor, joint_ids=drive_indices)
            writers_used.append(name)
        except Exception as exc:  # noqa: BLE001 -- name exists, signature may differ
            print(f"BASEAUTHORITY writer {name} rejected: {exc}", flush=True)

    # The actuator model's own clip. `ImplicitActuator.compute()` clamps to
    # `self.effort_limit` before anything reaches the solver, and no
    # write_*_to_sim call updates that Python-side tensor, so a drive whose
    # model limit is still 500 stays uncapped no matter what the sim-side
    # field says.
    actuators_patched = []
    for actuator_name, actuator in getattr(robot, "actuators", {}).items():
        joint_ids = getattr(actuator, "joint_indices", None)
        if joint_ids is None:
            continue
        if isinstance(joint_ids, slice):
            owned = list(range(robot.num_joints))[joint_ids]
        else:
            owned = [int(v) for v in joint_ids]
        local = [
            position
            for position, joint_id in enumerate(owned)
            if joint_id in set(drive_indices)
        ]
        if not local:
            continue
        for attr in ("effort_limit", "_effort_limit"):
            tensor = getattr(actuator, attr, None)
            if tensor is None or not hasattr(tensor, "__setitem__"):
                continue
            try:
                tensor[:, local] = effort_limit
                actuators_patched.append(f"{actuator_name}.{attr}")
            except Exception as exc:  # noqa: BLE001 -- shape varies by model
                print(
                    f"BASEAUTHORITY actuator {actuator_name}.{attr} "
                    f"rejected: {exc}",
                    flush=True,
                )

    before_armature = _live("joint_armature")
    armature_writer = next(
        (
            name
            for name in ("write_joint_armature_to_sim",)
            if hasattr(robot, name)
        ),
        None,
    )
    if armature_writer is None:
        print(
            "WARNING: BASEAUTHORITY no armature writer on this Articulation; "
            "the drive servo stays at zero rotor inertia",
            flush=True,
        )
    else:
        getattr(robot, armature_writer)(
            torch.full(
                (robot.num_instances, len(drive_indices)),
                armature,
                device=robot.device,
            ),
            joint_ids=drive_indices,
        )

    print(
        f"BASEAUTHORITY drive damping {before_damping} -> "
        f"{_live('joint_damping')} (requested {damping})",
        flush=True,
    )
    print(
        f"BASEAUTHORITY drive armature {before_armature} -> "
        f"{_live('joint_armature')} (requested {armature:.4f} kg.m^2 via "
        f"{armature_writer}; damping*dt/armature = "
        f"{damping * sim_dt / armature:.2f})",
        flush=True,
    )
    print(
        f"BASEAUTHORITY drive effort_limit {before_effort} -> {_live_all()} "
        f"(requested {effort_limit:.2f} N.m; writers={writers_used or None}; "
        f"actuators={actuators_patched or None})",
        flush=True,
    )


def apply_steering_authority(
    robot: Any,
    steering_indices: list[int],
    *,
    stiffness: float | None = None,
    damping: float | None = None,
    effort_limit: float | None = None,
    label: str = "STEERAUTHORITY",
) -> None:
    """Optional live override of the two steer axes' position-servo terms.

    Staged for the second measured anomaly of 2026-08-13: `steer_tgt`
    walked to -1.221 rad while `steer_pos` stayed at -0.002 over ~10 sim
    seconds, with the servo live at stiffness=500 / damping=50 /
    effort_limit=200. A 1.2 rad error demands 600 N.m and is clipped to
    200, so the axis was commanding full effort and not moving.

    The leading explanation is that it was not a steering fault at all:
    the two caster rolls were braked at 57.3 N.m.s/rad (see
    `free_caster_roll_joints`), so the steer axis was trying to rotate a
    loaded wheel about a contact patch held rigid by two anchored rollers.
    If freeing the casters also frees the steering, nothing here is
    needed and these stay unset.

    If it does NOT, the next term to test is the 200 N.m clip, since
    scrub torque on a steered wheel carrying a share of 147 kg can exceed
    it outright. Sweeping `--steering-effort-limit` answers that in one
    run instead of a day of theories. All three default to None, which
    leaves whatever `robot_actuator_cfg_specs()["steering_joints"]`
    delivered completely untouched -- this function is inert unless a
    flag is passed.
    """
    import torch

    requested = {
        "stiffness": stiffness,
        "damping": damping,
        "effort_limit": effort_limit,
    }
    if all(value is None for value in requested.values()):
        return

    writers = {
        "stiffness": ("write_joint_stiffness_to_sim",),
        "damping": ("write_joint_damping_to_sim",),
        "effort_limit": (
            "write_joint_effort_limit_to_sim",
            "write_joint_effort_limits_to_sim",
            "write_joint_max_effort_to_sim",
        ),
    }
    readbacks = {
        "stiffness": "joint_stiffness",
        "damping": "joint_damping",
        "effort_limit": "joint_effort_limits",
    }

    def _live(attr: str) -> list[float] | None:
        tensor = getattr(robot.data, attr, None)
        if tensor is None:
            return None
        return [round(float(tensor[0][i]), 4) for i in steering_indices]

    for term, value in requested.items():
        if value is None:
            continue
        before = _live(readbacks[term])
        writer = next(
            (name for name in writers[term] if hasattr(robot, name)), None
        )
        if writer is None:
            print(
                f"WARNING: {label} no writer for {term} on this "
                "Articulation; left unchanged",
                flush=True,
            )
            continue
        getattr(robot, writer)(
            torch.full(
                (robot.num_instances, len(steering_indices)),
                value,
                device=robot.device,
            ),
            joint_ids=steering_indices,
        )
        print(
            f"{label} {term} {before} -> {_live(readbacks[term])} "
            f"(requested {value} via {writer})",
            flush=True,
        )


def free_caster_roll_joints(
    robot: Any,
    *,
    damping: float | None = None,
) -> None:
    """Release the brake PhysX is holding on the two free-castering rollers.

    Measured live, 2026-08-13, `BASEGAINS passive`:

        names   [rocker, caster steering x2, caster roll x2]
        stiff   [35809.8633, 0.0, 0.0, 0.0,     0.0]
        damping [    0.1719, 0.0, 0.0, 57.2958, 57.2958]
        effort  [     500.0, 0.0, 0.0, 500.0,   500.0]

    Every one of those numbers is an authored value multiplied by
    57.29578 = 180/pi. The asset authors angular drive gains in USD's
    DEGREE units (kp=625 kd=0.003 on the rocker, kd=1.0 on the caster
    rolls), and PhysX converts them to per-RADIAN units on import:
    625 * 57.29578 = 35809.86, 0.003 * 57.29578 = 0.17189,
    1.0 * 57.29578 = 57.29578. The conversion is correct. The authored
    numbers are what is wrong, and `passive_base_joints` passes them
    through untouched (`stiffness: None, damping: None`) on the stated
    principle of preserving what the asset authored -- which silently
    preserved a unit-scale mistake nobody had ever read back.

    A damping of 57.3 N.m.s/rad on a FREE-SPINNING roller is not
    conditioning, it is a brake, and it is a large one. Rolling at 0.3 m/s
    turns a 0.05 m caster at ~6 rad/s, against which that damping demands
    ~344 N.m -- so the joint runs into its own 500 N.m effort limit and
    parks there. Two braked casters carrying roughly half of a 147 kg
    chassis is a drag the two drive wheels cannot overcome at any effort
    limit, which is precisely the failure that survived the traction cap:
    the wheels are no longer breaking traction, they are pushing a robot
    whose other two ground contacts are locked.

    The reason the original comment gives for preserving these gains --
    that zeroing them left 1-gram placeholder caster links ringing at
    18-85 rad/s and eating the drive wheels' traction budget -- was
    correct at the time and is now obsolete: the placeholder inertials
    were fixed earlier today (`derive_placeholder_link_inertials`, widened
    to actually match this asset), so these links now carry
    geometry-derived mass and no longer need an artificial brake to stay
    conditioned. Both halves of that trade have to move together, and only
    one of them did.

    Steering DOFs are deliberately NOT touched: a caster that cannot
    swivel is a skid, but a caster that cannot roll is an anchor, and only
    the roll axis is braked here.
    """
    import torch

    roll_ids = [
        i
        for i, name in enumerate(robot.joint_names)
        if name.startswith("caster_") and "steering" not in name
    ]
    if not roll_ids:
        print(
            "WARNING: CASTERFREE no caster roll joints matched; "
            f"joint names sampled: {robot.joint_names[:8]}",
            flush=True,
        )
        return

    if damping is None:
        damping = 0.0

    def _live(attr: str) -> list[float] | None:
        tensor = getattr(robot.data, attr, None)
        if tensor is None:
            return None
        return [round(float(tensor[0][i]), 4) for i in roll_ids]

    before = _live("joint_damping")
    robot.write_joint_damping_to_sim(
        torch.full(
            (robot.num_instances, len(roll_ids)), damping, device=robot.device
        ),
        joint_ids=roll_ids,
    )
    print(
        f"CASTERFREE {[robot.joint_names[i] for i in roll_ids]} "
        f"damping {before} -> {_live('joint_damping')} "
        f"(requested {damping})",
        flush=True,
    )


def verify_and_fix_base_drive_gains(
    app: Any, robot_prim_path: str, max_force: float | None,
    skip_writes: bool = False, test_spin: float | None = None,
) -> None:
    """Report, and optionally raise, the live max effort on the base's drive
    wheels.

    The wheels are velocity joints (authored stiffness=0, damping=100000), and
    they were measured turning at 0.41 rad/s against a 4.0 rad/s command, with
    the two sides differing eightfold (4.13 rad vs 0.53 rad over the same 10 s)
    -- the signature of a torque-saturated drive, not a disconnected one. The
    base moved ~0.03 m and stopped in BOTH directions with nothing in front of
    it on /scan, which rules out an obstacle.

    Reported unconditionally because the live value has never once been read;
    `--base-max-force` is what changes it.
    """
    from isaacsim.core.prims import Articulation

    for _ in range(5):
        app.update()
    articulation = Articulation(prim_paths_expr=robot_prim_path)
    articulation.initialize()
    dof_names = list(articulation.dof_names)
    wheels = [n for n in dof_names if n in ("tmrv0_2_joint_1", "tmrv0_2_joint_3")]
    steer = [n for n in dof_names if n in ("tmrv0_2_joint_0", "tmrv0_2_joint_2")]
    # The unpowered rollers and the rocker. These carry the robot's weight and
    # must SPIN FREELY; a position drive on them locks the chassis to the
    # floor exactly the way the position drives on the Robotiq's passive
    # 4-bar links locked the gripper. Same defect, same place to look.
    # BOTH the roll and the swivel. A caster that cannot swivel is a skid:
    # the two powered modules push, the locked casters scrub sideways, and
    # the chassis goes nowhere -- which is exactly what was measured, ~0.025 m
    # of travel in EITHER direction with a clear /scan and the wheels turning.
    casters = [n for n in dof_names if n.startswith("caster_")]
    rocker = [n for n in dof_names if n == "rocker_arm_joint"]
    idx = {n: dof_names.index(n) for n in wheels + steer + casters + rocker}
    if not wheels:
        print("WARNING: no base drive wheels found in dof_names")
        return

    # PhysX's OWN masses for the moving base bodies. Nothing in the USD
    # authors `physics:mass`, so these are computed from collision geometry
    # and density -- and a body that came out with zero or near-zero mass and
    # inertia cannot be accelerated by any drive at any gain, which is what a
    # wheel that will not spin even when PhysX is told to spin it directly
    # would look like.
    try:
        body_names = list(articulation.body_names)
        masses = articulation.get_body_masses()
        inertias = articulation.get_body_inertias()
        total_mass = float(sum(float(masses[0][i]) for i in range(len(body_names))))
        heavy = sorted(range(len(body_names)),
                       key=lambda i: -float(masses[0][i]))[:12]
        print(f"Live ROBOT total mass={total_mass:.3f} kg over "
              f"{len(body_names)} bodies; heaviest: "
              + ", ".join(f"{body_names[i]}={float(masses[0][i]):.3f}"
                          for i in heavy))
        for i, bn in enumerate(body_names):
            if not any(k in bn for k in ("argo_drive", "caster", "rocker",
                                         "base_link")):
                continue
            inertia = inertias[0][i]
            diag = [float(inertia[0]), float(inertia[4]), float(inertia[8])] \
                if len(inertia) >= 9 else list(map(float, inertia))
            print(f"Live BASE body {bn}: mass={float(masses[0][i]):.5f} "
                  f"inertia_diag={[round(v, 6) for v in diag]}")
    except Exception as exc:  # noqa: BLE001 -- report, never mask
        print(f"NOTE: body masses unreadable: {type(exc).__name__}: {exc}")

    kps, kds = articulation.get_gains()
    efforts = articulation.get_max_efforts()
    try:
        vlim = articulation.get_max_joint_velocities()
        print("Live BASE joint velocity limits: " + ", ".join(
            f"{n}={float(vlim[0][i]):.3f}" for n, i in idx.items()))
    except Exception as exc:  # noqa: BLE001 -- report, never mask
        print(f"NOTE: max joint velocities unreadable: {type(exc).__name__}: {exc}")
    print("Live BASE gains BEFORE: " + ", ".join(
        f"{n}: kp={float(kps[0][i]):.1f} kd={float(kds[0][i]):.1f} "
        f"maxF={float(efforts[0][i]):.1f}" for n, i in idx.items()))

    # Restore the ORGANIZERS' OWN base actuator spec. Source:
    # `scripts/scenes/scene_robot_keyboard.py` on the upstream
    # `EBiM-Benchmark/benchmark` `Robotiq_DEMO` branch, which configures this
    # exact robot:
    #
    #     steering_joints      stiffness=500.0 damping=50.0  effort_limit=200
    #     drive_joints         stiffness=0.0   damping=5.0   effort_limit=200
    #     passive_base_joints  stiffness=0.0   damping=0.0   (.*caster.*,
    #                                                         rocker_arm_joint)
    #
    # The live values on our asset are off by six orders of magnitude: the
    # drive wheels carry damping=5729578 against upstream's 5.0. On a velocity
    # joint that is not "stiff", it is rigid -- any velocity error demands an
    # unbounded torque, the drive saturates, and the wheel simply does not
    # turn. Measured: 0.001-0.03 rad/s achieved against commands of 0.5-2.0,
    # and the base covering ~0.025 m in any direction with a clear /scan.
    #
    # An earlier revert of the caster/rocker part of this was wrong. It was
    # made on the observation that the chassis rode 0.18 m high, but that run
    # also carried a bad `--robot-z 0.12`; upstream spawns at z=0.0 and lists
    # both casters AND the rocker as passive with zero gains.
    # `--base-max-force`, when given, doubles as a POSITION-mode switch for
    # the wheels. Velocity commands to them have never once moved the base,
    # and a position gain distinguishes the two remaining explanations: if a
    # position step turns the wheel, the joint is free and the velocity
    # channel is what fails; if it does not, the wheel is mechanically jammed
    # and no controller change can help.
    wheel_kp = 0.0 if max_force is None else 500.0
    wheel_kd = 5.0 if max_force is None else 50.0
    # A spin target through the TENSOR API, which is the only path proven to
    # reach PhysX on this stack. `--base-test-spin` originally wrote
    # `drive:angular:physics:targetVelocity` in USD and the wheels did not
    # move -- the same defect already documented here for the arm and gripper
    # gains, where a USD DriveAPI write reads back fine from USD and never
    # reaches the solver. This re-runs that test through the API that works.
    if test_spin is not None:
        ji = [idx[n] for n in wheels if n in idx]
        if ji:
            import numpy as _np

            targets = _np.zeros((1, len(dof_names)), dtype=_np.float32)
            for i in ji:
                targets[0][i] = test_spin
            articulation.set_joint_velocity_targets(
                targets[:, ji], joint_indices=ji
            )
            print(f"Base test spin via tensor API: {test_spin} rad/s on "
                  f"{[dof_names[i] for i in ji]}")

    if skip_writes:
        print("Base drive gains: REPORT ONLY (--no-base-gain-fix)")
        return
    for names, kp, kd, eff in (
        (wheels, wheel_kp, wheel_kd, 200.0 if max_force is None else max_force),
        (steer, 500.0, 50.0, 200.0),
        (casters, 0.0, 0.0, None),
        # The rocker is NOT passive on this robot, whatever upstream's
        # `passive_base_joints` says for theirs. Measured link heights with it
        # free: drive wheels at z=0.055 and 0.054 (radius 0.05, so their
        # contact points are ~5 mm in the air) while the casters sit at
        # z=0.034 and 0.028 and carry the whole robot. Wheels off the ground
        # spin to their velocity limit -- +-20 rad/s was measured -- and
        # deliver no traction at all, which is the entire reason the base has
        # never moved more than ~0.03 m. The rocker is the suspension that
        # presses the drive wheels down, so it needs a drive that can hold a
        # commanded angle.
        (rocker, 500.0, 50.0, 200.0),
    ):
        names = [n for n in names if n in idx]
        if not names:
            continue
        ji = [idx[n] for n in names]
        articulation.set_gains(kps=[[kp] * len(ji)], kds=[[kd] * len(ji)],
                               joint_indices=ji)
        if eff is not None:
            articulation.set_max_efforts(values=[[eff] * len(ji)],
                                         joint_indices=ji)
    # Upstream also sets `velocity_limit_sim=20.0` on the drive joints. A
    # velocity-driven joint whose velocity LIMIT is near zero cannot turn no
    # matter how much torque the drive is willing to apply, and would look
    # exactly like the jam measured here: 0.02-0.34 rad in 20 s against a
    # 6.0 rad/s command, while the passive casters beside it spun a free
    # radian.
    ji = [idx[n] for n in wheels if n in idx]
    if ji:
        try:
            articulation.set_max_joint_velocities(
                values=[[20.0] * len(ji)], joint_indices=ji)
        except Exception as exc:  # noqa: BLE001 -- report, never mask
            print(f"NOTE: could not set max joint velocities: "
                  f"{type(exc).__name__}: {exc}")
    kps, kds = articulation.get_gains()
    efforts = articulation.get_max_efforts()
    print("Live BASE gains AFTER: " + ", ".join(
        f"{n}: kp={float(kps[0][i]):.1f} kd={float(kds[0][i]):.1f} "
        f"maxF={float(efforts[0][i]):.1f}" for n, i in idx.items()))


_EFFORT_LOG_SUBSCRIPTION = None  # module-level: must outlive main()'s return


def start_periodic_effort_logging(
    robot_prim_path: str, joint_names: list[str], interval_s: float = 2.0
) -> None:
    """Opus's decisive measurement (2026-08-11): the whole joint1 thread has
    measured positions and limits, never the actual applied torque. My kp
    sweep (1e6/2e3, 3e6/5e3) was saturated by ~40x at every point (kp*err
    vs max_force) -- flat-in-kp is exactly what a saturated drive predicts,
    not evidence against a torque mechanism; the earlier "falsified"
    conclusion doesn't hold. This logs `get_measured_joint_efforts()` (the
    real, physics-solved torque) and `get_applied_actions()` (the drive's
    own stored setpoint, to catch a possible command-path bug where PhysX's
    target differs from what we think we commanded -- joint1's axis is
    kinematically vertical, gravity moment arm ~zero, confirmed twice by
    the level-base and dead-spine checks, so a torque reading here that
    ISN'T near max_effort AND isn't near zero would be the "measurement
    frame is wrong" case) continuously, since torque needs to be read AT a
    settled pose an external ROS2 command (from move_group, a separate
    process) will produce well after this script's own `main()` returns --
    a one-shot check can't be timed against that from outside.
    """
    global _EFFORT_LOG_SUBSCRIPTION
    import time as _time

    import omni.kit.app
    from isaacsim.core.prims import Articulation

    articulation = Articulation(prim_paths_expr=robot_prim_path)
    articulation.initialize()
    dof_names = list(articulation.dof_names)
    indices = [dof_names.index(name) for name in joint_names if name in dof_names]
    names_in_order = [dof_names[i] for i in indices]

    state = {"last_log": 0.0}

    def _on_update(_event: Any) -> None:
        now = _time.monotonic()
        if now - state["last_log"] < interval_s:
            return
        state["last_log"] = now
        efforts = articulation.get_measured_joint_efforts()
        actions = articulation.get_applied_actions()
        positions = articulation.get_joint_positions()
        print(
            "EFFORT_LOG "
            f"joints={names_in_order} "
            f"measured_effort={[float(efforts[0][i]) for i in indices]} "
            f"applied_action_position={[float(actions.joint_positions[0][i]) for i in indices]} "
            f"current_position={[float(positions[0][i]) for i in indices]}",
            flush=True,
        )

    _EFFORT_LOG_SUBSCRIPTION = (
        omni.kit.app.get_app()
        .get_update_event_stream()
        .create_subscription_to_pop(_on_update, name="effort_log")
    )
    print(
        f"Periodic effort logging started for {names_in_order} "
        f"every {interval_s}s"
    )


def publish_ros2_clock(args: argparse.Namespace) -> None:
    """REV20 P0.4: enabling isaacsim.ros2.bridge alone does not publish
    /clock -- confirmed empirically (a fresh `ros2 topic list` over host
    networking showed only /parameter_events and /rosout after the bridge
    extension loaded and the timeline was playing). An explicit
    ROS2PublishClock node is required. `OnPlaybackTick` already exposes
    `outputs:time` (global playback time in seconds, taken directly from
    the execution context) -- wire that straight into the publisher rather
    than adding a separate IsaacReadSimulationTime node (first attempt used
    that node's `outputs:simulationTime` and produced no visible /clock
    topic; simplifying to rule out a stale/uncomputed pure-data-node
    dependency in the "execution" evaluator before investigating further).
    """
    if args.ros2_bridge == "disabled":
        return

    import omni.graph.core as og

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2_ClockGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
            ],
            keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "PublishClock.inputs:execIn",
                ),
                (
                    "Context.outputs:context",
                    "PublishClock.inputs:context",
                ),
                (
                    "OnPlaybackTick.outputs:time",
                    "PublishClock.inputs:timeStamp",
                ),
            ],
        },
    )
    print("ROS2 clock publisher graph created: /ROS2_ClockGraph")


def publish_ros2_joint_states(
    args: argparse.Namespace, robot_prim_path: str = "/World/Robot"
) -> None:
    """REV20 P1: publish /joint_states so `robot_state_publisher` can build
    the arm/base TF tree from the URDF (P1's own stated preference over
    hand-publishing every link transform). Pin names cross-checked against
    the installed extension's OGN docs, same discipline as
    ``publish_ros2_clock``.
    """
    if args.ros2_bridge == "disabled":
        return

    import omni.graph.core as og

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2_JointStateGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                (
                    "PublishJointState",
                    "isaacsim.ros2.bridge.ROS2PublishJointState",
                ),
            ],
            keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "PublishJointState.inputs:execIn",
                ),
                (
                    "Context.outputs:context",
                    "PublishJointState.inputs:context",
                ),
                (
                    "OnPlaybackTick.outputs:time",
                    "PublishJointState.inputs:timeStamp",
                ),
            ],
            keys.SET_VALUES: [
                ("PublishJointState.inputs:targetPrim", robot_prim_path),
            ],
        },
    )
    print(
        "ROS2 joint_states publisher graph created: /ROS2_JointStateGraph "
        f"(targetPrim={robot_prim_path})"
    )


def publish_ros2_odometry(
    args: argparse.Namespace,
    chassis_prim_path: str = "/World/Robot",
    base_frame_id: str = "base_link",
) -> None:
    """REV20 P1: /odom AND the odom->base_link /tf edge. `IsaacComputeOdometry`
    reads the chassis prim (reuse the SAME resolved articulation root
    `fix_single_articulation_root` returns -- do not pass `/World/Robot`
    again and rediscover the ambiguous-root problem from
    `publish_ros2_joint_states`) and feeds position/orientation/velocity
    into `ROS2PublishOdometry` for the /odom topic. `ROS2PublishOdometry`
    alone does NOT publish /tf -- confirmed by inspecting real /tf traffic
    with robot_state_publisher + a map->odom static_transform_publisher
    both running: /tf only ever carried `robot_state_publisher`'s URDF-
    internal edges (base_link as parent of e.g. rocker_arm_link, never as
    a child), so `tf2_echo map base_link` reported two disjoint trees.
    `ROS2PublishRawTransformTree` (confirmed present via
    TestOgnROS2PublishRawTransformTree.py's own inputs:
    parentFrameId/childFrameId/translation/rotation) closes the gap by
    publishing the same ComputeOdometry position/orientation as the
    odom->base_link edge, connecting the static map->odom tree to
    robot_state_publisher's base_link-rooted tree.
    """
    if args.ros2_bridge == "disabled":
        return

    import omni.graph.core as og

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2_OdometryGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                (
                    "ComputeOdometry",
                    "isaacsim.core.nodes.IsaacComputeOdometry",
                ),
                (
                    "PublishOdometry",
                    "isaacsim.ros2.bridge.ROS2PublishOdometry",
                ),
                (
                    "PublishOdomTF",
                    "isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
                ),
            ],
            keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "ComputeOdometry.inputs:execIn",
                ),
                (
                    "ComputeOdometry.outputs:execOut",
                    "PublishOdometry.inputs:execIn",
                ),
                (
                    "Context.outputs:context",
                    "PublishOdometry.inputs:context",
                ),
                (
                    "OnPlaybackTick.outputs:time",
                    "PublishOdometry.inputs:timeStamp",
                ),
                (
                    "ComputeOdometry.outputs:position",
                    "PublishOdometry.inputs:position",
                ),
                (
                    "ComputeOdometry.outputs:orientation",
                    "PublishOdometry.inputs:orientation",
                ),
                (
                    "ComputeOdometry.outputs:linearVelocity",
                    "PublishOdometry.inputs:linearVelocity",
                ),
                (
                    "ComputeOdometry.outputs:angularVelocity",
                    "PublishOdometry.inputs:angularVelocity",
                ),
                (
                    "ComputeOdometry.outputs:execOut",
                    "PublishOdomTF.inputs:execIn",
                ),
                (
                    "Context.outputs:context",
                    "PublishOdomTF.inputs:context",
                ),
                (
                    "OnPlaybackTick.outputs:time",
                    "PublishOdomTF.inputs:timeStamp",
                ),
                (
                    "ComputeOdometry.outputs:position",
                    "PublishOdomTF.inputs:translation",
                ),
                (
                    "ComputeOdometry.outputs:orientation",
                    "PublishOdomTF.inputs:rotation",
                ),
            ],
            keys.SET_VALUES: [
                ("ComputeOdometry.inputs:chassisPrim", chassis_prim_path),
                ("PublishOdomTF.inputs:parentFrameId", "odom"),
                ("PublishOdomTF.inputs:childFrameId", base_frame_id),
            ],
        },
    )
    print(
        "ROS2 odometry publisher graph created: /ROS2_OdometryGraph "
        f"(chassisPrim={chassis_prim_path}, "
        f"odom->{base_frame_id} /tf edge added)"
    )


def publish_world_map_static_tf(
    args: argparse.Namespace,
    world_frame: str = "world",
    map_frame: str = "map",
) -> None:
    """Anchor the ROS2 TF tree to the sim's `world` frame.

    The scene publishes object frames in `world`; the chassis odom edge is
    `odom->base_link`. Without a `world->map` (or `world->odom`) edge those
    are two disjoint TF trees, so the rclpy sidecar cannot tf2-lookup an
    object's pose in the robot's odom/base_link frame -- it can only compare
    raw numbers and hope the frames happen to share an origin (the exact
    documented approximation that made `gate_h5_grasp_and_lift.py` report
    the nearest object as ~4.5 m away even when parked at the table:
    objects live in `world`, the robot's odom is anchored at its spawn).

    REV21 P2 CORRECTION 2026-08-12: this used to publish `world->map` as the
    robot's spawn pose, under the assumption (true when this was written)
    that nav2's `map_to_odom_static_tf` stayed a static identity -- making
    the combined `world->map->odom` chain land odom at the spawn pose,
    expressed in world coordinates. That assumption is no longer true:
    `task3_nav2_bringup.launch.py`'s `map_to_odom_static_tf` is now
    parameterized from the real spawn pose too (P1, `6d12931`), because
    Nav2's own occupancy map has its origin stated in `room_obstacles.json`'s
    absolute/world coordinates -- `map` must equal `world` for that origin
    to mean what the YAML sidecar says it means, not be a second
    spawn-anchored frame the way this function used to treat it. With BOTH
    functions applying the spawn offset, `world->odom` doubled it -- GPU-
    confirmed: `odom->head_Camera` read a physically implausible ~4.9 m
    offset, and `live_camera_perception.py`'s own `odom->head_Camera`
    lookup failed outright (ConnectivityException) on the corrupted tree,
    manifesting as the long-documented "no TF for head_Camera -> odom yet"
    flake (`plans/PROGRESS.md`'s own "low-severity... not fully root-caused"
    note). Publishing `world->map` as identity instead -- `map` IS `world`,
    not a second copy of the spawn offset -- so the one real spawn offset
    lives only in `map->odom`, matching what the occupancy map already
    assumes. Chain: `world(=map)->odom(spawn pose)->base_link` plus
    `world->object`, so tf2 resolves `odom->object`/`base_link->object`
    correctly. `robot_position`/`robot_yaw` params dropped -- no longer
    used, and keeping them would silently invite reintroducing this exact
    double-offset bug.
    """
    if args.ros2_bridge == "disabled":
        return

    import omni.graph.core as og

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2_WorldMapStaticTF", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                (
                    "PublishWorldMapTF",
                    "isaacsim.ros2.bridge.ROS2PublishRawTransformTree",
                ),
            ],
            keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "PublishWorldMapTF.inputs:execIn",
                ),
                (
                    "Context.outputs:context",
                    "PublishWorldMapTF.inputs:context",
                ),
                (
                    "OnPlaybackTick.outputs:time",
                    "PublishWorldMapTF.inputs:timeStamp",
                ),
            ],
            keys.SET_VALUES: [
                (
                    "PublishWorldMapTF.inputs:translation",
                    (0.0, 0.0, 0.0),
                ),
                (
                    "PublishWorldMapTF.inputs:rotation",
                    # omni.graph's quatd input is (IJKR) = (x, y, z, w).
                    (0.0, 0.0, 0.0, 1.0),
                ),
                ("PublishWorldMapTF.inputs:parentFrameId", world_frame),
                ("PublishWorldMapTF.inputs:childFrameId", map_frame),
            ],
        },
    )
    print(
        "ROS2 static world->map TF graph created: /ROS2_WorldMapStaticTF "
        f"(parent={world_frame}, child={map_frame}, identity)"
    )


def gripper_pad_prim_paths(stage, asset_root_path: str, side: str) -> list[str]:
    """Prim paths of one arm's two Robotiq finger-pad links.

    Both Robotiq subtrees name their own fingers `left_inner_finger` /
    `right_inner_finger` with no arm-side prefix, so a name lookup is
    ambiguous across arms -- the arm is identified by the enclosing
    `<side>_Robotiq_2F_85` scope instead, which is unambiguous.
    """
    if side in (None, "none"):
        return []

    from pxr import Usd

    robot_prim = stage.GetPrimAtPath(asset_root_path)
    if not (robot_prim and robot_prim.IsValid()):
        print(f"WARNING: gripper pad TF: no robot prim at {asset_root_path}")
        return []
    scope = f"{side}_Robotiq_2F_85"
    paths = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(robot_prim)
        if prim.GetName().endswith("_inner_finger")
        and f"/{scope}/" in str(prim.GetPath())
    ]
    print(f"Gripper pad TF ({side} arm): {len(paths)} pad prims {paths}")
    return paths


def named_prim_paths(stage, asset_root_path: str, names: list[str]) -> list[str]:
    """Prim paths under the robot for an explicit list of link names, so any
    link's REAL pose can be put on /tf and measured.

    The robot's own /tf comes from robot_state_publisher via a URDF that does
    not match this asset, so a link's published pose is not evidence about
    where the link physically is. This is the same escape hatch the finger
    pads needed.
    """
    if not names:
        return []

    from pxr import Usd

    robot_prim = stage.GetPrimAtPath(asset_root_path)
    if not (robot_prim and robot_prim.IsValid()):
        return []
    wanted = set(names)
    paths = [str(p.GetPath()) for p in Usd.PrimRange(robot_prim)
             if p.GetName() in wanted]
    print(f"Extra link TF: {len(paths)} prims {paths}")
    return paths


def publish_ros2_object_tf(
    args: argparse.Namespace, stage, object_names: list[str],
    asset_root_path: str = "",
) -> None:
    """H5-H9 grasp-and-lift gate: put a named object's own live world pose
    onto /tf, the same way the robot's joints/odom already are, so a
    grasp-and-lift proof run from the rclpy sidecar container can confirm
    the object actually moved -- not just that the gripper closed and the
    arm's own commanded joints reached their targets, which is exactly the
    ambiguity the standalone grip diagnostic could never resolve ("closed
    on empty air" vs. a real grasp looked identical from the arm's own
    state alone).

    `ROS2PublishTransformTree` (distinct from `ROS2PublishRawTransformTree`,
    used above for the odom edge) takes a `targetPrims` list directly and
    reads each prim's own live world transform every tick -- confirmed via
    `isaac-lab-2-3-2-workshop`'s installed
    `OgnROS2PublishTransformTree.rst` (2026-08-11): no odometry/FK
    computation needed, unlike the chassis-based odom publisher above.
    """
    pad_paths = gripper_pad_prim_paths(
        stage, asset_root_path, getattr(args, "publish_gripper_pad_tf", "none")
    )
    pad_paths += named_prim_paths(
        stage, asset_root_path, getattr(args, "publish_link_tf", None) or []
    )
    if args.ros2_bridge == "disabled" or not (object_names or pad_paths):
        return

    from integration_test import resolve_prim_path

    import omni.graph.core as og

    resolved_paths = [resolve_prim_path(stage, name) for name in object_names]
    resolved_paths.extend(pad_paths)

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2_ObjectTFGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                (
                    "PublishObjectTF",
                    "isaacsim.ros2.bridge.ROS2PublishTransformTree",
                ),
            ],
            keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "PublishObjectTF.inputs:execIn",
                ),
                (
                    "Context.outputs:context",
                    "PublishObjectTF.inputs:context",
                ),
                (
                    "OnPlaybackTick.outputs:time",
                    "PublishObjectTF.inputs:timeStamp",
                ),
            ],
            keys.SET_VALUES: [
                ("PublishObjectTF.inputs:targetPrims", resolved_paths),
            ],
        },
    )
    print(
        "ROS2 object TF publisher graph created: /ROS2_ObjectTFGraph "
        f"(targetPrims={resolved_paths})"
    )


def publish_ros2_joint_command(
    args: argparse.Namespace, robot_prim_path: str = "/World/Robot"
) -> None:
    """H5-H9: the execution half of the MoveIt gate needs a way to command
    the live robot's joints from ROS2. `/joint_command` is the organizers'
    own topic name for this (`DEMO/record.py` on the public
    `EBiM-Benchmark/benchmark` `Robotiq_DEMO` branch, see
    `plans/BLOCKED_FOR_OPUS.md` BLOCKER 4's "free intel" section) -- reusing
    it rather than inventing a new one keeps this compatible with any
    tooling built against the official topic surface.

    Targets the SAME resolved articulation root
    `fix_single_articulation_root` returns, same discipline as
    `publish_ros2_odometry`'s own docstring -- do NOT re-pass an ambiguous
    `robot_prim_path` here and rediscover that bug.
    """
    if args.ros2_bridge == "disabled":
        return

    import omni.graph.core as og

    keys = og.Controller.Keys
    og.Controller.edit(
        {"graph_path": "/ROS2_JointCommandGraph", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                (
                    "SubscribeJointCommand",
                    "isaacsim.ros2.bridge.ROS2SubscribeJointState",
                ),
                (
                    "ArticulationController",
                    "isaacsim.core.nodes.IsaacArticulationController",
                ),
            ],
            keys.CONNECT: [
                (
                    "OnPlaybackTick.outputs:tick",
                    "SubscribeJointCommand.inputs:execIn",
                ),
                (
                    "Context.outputs:context",
                    "SubscribeJointCommand.inputs:context",
                ),
                (
                    "SubscribeJointCommand.outputs:execOut",
                    "ArticulationController.inputs:execIn",
                ),
                (
                    "SubscribeJointCommand.outputs:jointNames",
                    "ArticulationController.inputs:jointNames",
                ),
                (
                    "SubscribeJointCommand.outputs:positionCommand",
                    "ArticulationController.inputs:positionCommand",
                ),
                (
                    "SubscribeJointCommand.outputs:velocityCommand",
                    "ArticulationController.inputs:velocityCommand",
                ),
                (
                    "SubscribeJointCommand.outputs:effortCommand",
                    "ArticulationController.inputs:effortCommand",
                ),
            ],
            keys.SET_VALUES: [
                ("SubscribeJointCommand.inputs:topicName", "/joint_command"),
                ("ArticulationController.inputs:targetPrim", robot_prim_path),
            ],
        },
    )
    print(
        "ROS2 joint_command subscriber graph created: /ROS2_JointCommandGraph "
        f"(targetPrim={robot_prim_path}, topic=/joint_command)"
    )


# Relative-to-asset-root camera prim paths, confirmed by a direct `pxr`
# scan of the standalone downloaded Robotiq asset file (P0.2 enumeration,
# see plans/REV20_TASKQUEUE.md): 3 real onboard cameras (a ZED Mini head
# stereo rig + 2 D405 wrist cameras), matching the official topic surface
# recovered from the organizers' own DEMO/record.py
# (plans/BLOCKED_FOR_OPUS.md): /rgb_head, /rgb_left, /rgb_right.
# NOT yet independently confirmed for the compat robot currently loaded
# by default (gripper=None) -- `fix_single_articulation_root` resolves
# its own articulation root to the same "<asset_root>/base" shape the
# Robotiq asset uses, so these relative paths are a reasoned inference
# from that shared pattern, not a re-scan of this specific file. Each
# camera is skipped (not a hard failure) if its prim does not exist on
# whichever robot is actually loaded -- see publish_ros2_cameras.
ROBOTIQ_CAMERA_RELATIVE_PATHS: dict[str, tuple[str, str]] = {
    "rgb_head": (
        "zedmini/zed_mini_camera_link/zed_mini_left_camera_frame/head_Camera",
        "head_camera",
    ),
    "rgb_left": (
        "left_d405_camera_with_mount/d405_camera_link/left_Camera",
        "left_wrist_camera",
    ),
    "rgb_right": (
        "right_d405_camera_with_mount/d405_camera_link/right_Camera",
        "right_wrist_camera",
    ),
}


def list_camera_prims_under(stage: Any, root_path: str) -> list[str]:
    """Diagnostic: real `UsdGeom.Camera`-typed prims under `root_path` on
    the LIVE referenced stage. Avoids re-deriving prim paths from a
    separately-scanned standalone USD file (which can differ once
    referenced/renamed into the scene) -- this walks the exact stage
    `publish_ros2_cameras` will target, the same live-scene method that
    already correctly found the compat robot's cameras missing at the
    Robotiq-asset-derived guesses (see plans/REV20_TASKQUEUE.md).
    """
    from pxr import Usd, UsdGeom

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return []
    paths = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root_prim)
        if prim.IsA(UsdGeom.Camera)
    ]
    print(f"CAMERA_PRIMS under {root_path}: {paths}")
    return paths


def list_gripper_joint_names(stage: Any, root_path: str) -> list[str]:
    """Diagnostic: real joint prim names under `root_path` whose path
    contains "knuckle" or "finger" -- the Robotiq 4-bar linkage naming
    signature confirmed by P0.2's standalone-file scan
    (plans/BLOCKED_FOR_OPUS.md). Used to unblock
    `robot_actuator_cfg_specs()` (this file, ~line 318) for the robot
    swap without needing to fix the separate IsaacLab actuator-config
    crash first -- this script's `--no-keyboard-control` path never
    constructs an `Articulation`/actuator config at all, so it can
    reference the Robotiq asset (`--robot-usd`) and enumerate its real
    joints safely.
    """
    from pxr import Usd, UsdPhysics

    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        return []
    names = []
    for prim in Usd.PrimRange(root_prim):
        path_str = str(prim.GetPath())
        lowered = path_str.lower()
        if "knuckle" not in lowered and "finger" not in lowered:
            continue
        if not prim.IsA(UsdPhysics.RevoluteJoint):
            continue
        has_drive = prim.HasAPI(UsdPhysics.DriveAPI, "angular")
        stiffness = None
        if has_drive:
            drive = UsdPhysics.DriveAPI(prim, "angular")
            stiffness = drive.GetStiffnessAttr().Get()
        names.append(path_str)
        print(
            f"  {path_str} has_angular_drive={has_drive} stiffness={stiffness}"
        )
    print(f"GRIPPER_JOINTS under {root_path} ({len(names)}): {names}")
    return names


# REV20 P1 /scan: which onboard cameras (by their publish_ros2_cameras
# topic key) also get a depth_pcl (sensor_msgs/PointCloud2) publisher on
# the SAME render product, feeding an external `pointcloud_to_laserscan`
# node -- per the P0.4 lidar finding (no RTX Lidar prim on either robot).
# The head camera (a forward-facing ZED Mini stereo rig, not a wrist
# camera) is the natural choice for base-navigation obstacle sensing.
DEPTH_PCL_CAMERA_TOPICS: frozenset[str] = frozenset({"rgb_head"})


def publish_ros2_cameras(
    args: argparse.Namespace,
    asset_root_path: str,
    camera_relative_paths: dict[str, tuple[str, str]] | None = None,
    depth_pcl_topics: frozenset[str] = DEPTH_PCL_CAMERA_TOPICS,
) -> None:
    """REV20 P1: RGB + camera_info for the onboard cameras, on the exact
    official topic names (`/rgb_head`, `/rgb_left`, `/rgb_right`) recovered
    from the organizers' own `DEMO/record.py` -- rename nothing, per
    `plans/BLOCKED_FOR_OPUS.md`. Reuses the same
    `IsaacCreateRenderProduct` + `ROS2CameraHelper` + `ROS2CameraInfoHelper`
    OmniGraph shape as `setup_deformable_camera`, but targets a REAL
    onboard camera prim (`RenderProduct.inputs:cameraPrim`) instead of
    defining a new floating camera. Each entry is checked with
    `stage.GetPrimAtPath(...).IsValid()` before wiring -- a missing camera
    on whichever robot is currently loaded is a skip with a printed
    warning, not a crash, since the relative paths above are confirmed
    for the Robotiq asset only (see the module-level comment).

    For each topic in `depth_pcl_topics`, also adds a `depth_pcl`
    `ROS2CameraHelper` (confirmed via the real installed extension source,
    `OgnROS2CameraHelper.py`: `depth_pcl` uses the identical input schema
    as `rgb`/`depth`, just a different writer -- `ROS2PublishPointCloud`
    instead of `ROS2PublishImage`) reusing the SAME `RenderProduct`, on
    topic `points_<frame_id>`. An external `pointcloud_to_laserscan` node
    (real ROS2 package, run in a client container -- not an Isaac Sim
    OmniGraph node) subscribes to this and republishes `/scan`.
    """
    if args.ros2_bridge == "disabled":
        return
    if camera_relative_paths is None:
        camera_relative_paths = ROBOTIQ_CAMERA_RELATIVE_PATHS

    import omni.graph.core as og
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    keys = og.Controller.Keys
    camera_tf_prim_paths: list[str] = []

    for topic_name, (relative_path, frame_id) in camera_relative_paths.items():
        camera_prim_path = f"{asset_root_path}/{relative_path}"
        if not stage.GetPrimAtPath(camera_prim_path).IsValid():
            print(
                f"ROS2 camera SKIPPED (prim not found): {camera_prim_path} "
                f"(topic would have been /{topic_name})"
            )
            continue
        camera_tf_prim_paths.append(camera_prim_path)

        node_namespace = ""
        graph_path = f"/ROS2_CameraGraphs/{topic_name}"
        create_nodes = [
            ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
            (
                "RenderProduct",
                "isaacsim.core.nodes.IsaacCreateRenderProduct",
            ),
            (
                "RunOnce",
                "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame",
            ),
            ("Context", "isaacsim.ros2.bridge.ROS2Context"),
            ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
            (
                "CameraInfoPublish",
                "isaacsim.ros2.bridge.ROS2CameraInfoHelper",
            ),
        ]
        connect = [
            ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
            ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
            ("RenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
            (
                "RenderProduct.outputs:renderProductPath",
                "RGBPublish.inputs:renderProductPath",
            ),
            ("Context.outputs:context", "RGBPublish.inputs:context"),
            (
                "RenderProduct.outputs:execOut",
                "CameraInfoPublish.inputs:execIn",
            ),
            (
                "RenderProduct.outputs:renderProductPath",
                "CameraInfoPublish.inputs:renderProductPath",
            ),
            (
                "Context.outputs:context",
                "CameraInfoPublish.inputs:context",
            ),
        ]
        set_values = [
            ("RenderProduct.inputs:cameraPrim", camera_prim_path),
            ("RenderProduct.inputs:height", 480),
            ("RenderProduct.inputs:width", 640),
            ("RGBPublish.inputs:type", "rgb"),
            ("RGBPublish.inputs:nodeNamespace", node_namespace),
            ("RGBPublish.inputs:topicName", topic_name),
            ("RGBPublish.inputs:frameId", frame_id),
            ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
            (
                "CameraInfoPublish.inputs:topicName",
                f"{topic_name}/camera_info",
            ),
            ("CameraInfoPublish.inputs:frameId", frame_id),
            ("CameraInfoPublish.inputs:nodeNamespace", node_namespace),
            ("CameraInfoPublish.inputs:resetSimulationTimeOnStop", True),
        ]

        # Raw z-depth image (sensor_msgs/Image, 32FC1, distance_to_image_plane)
        # on every onboard camera -- BLOCK A perception needs a per-pixel depth
        # reading to back-project a detected box into 3-D (task3_pipeline/
        # sim_camera_perception.py's back_project()), which a PointCloud2
        # cannot serve directly. Reuses the same RenderProduct as RGB/depth_pcl;
        # same ROS2CameraHelper schema, just type="depth" per
        # OgnROS2CameraHelper.py (confirmed at setup_deformable_camera below).
        depth_topic_name = f"depth_{frame_id}"
        create_nodes.append(
            ("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper")
        )
        connect.extend(
            [
                ("RenderProduct.outputs:execOut", "DepthPublish.inputs:execIn"),
                (
                    "RenderProduct.outputs:renderProductPath",
                    "DepthPublish.inputs:renderProductPath",
                ),
                ("Context.outputs:context", "DepthPublish.inputs:context"),
            ]
        )
        set_values.extend(
            [
                ("DepthPublish.inputs:type", "depth"),
                ("DepthPublish.inputs:nodeNamespace", node_namespace),
                ("DepthPublish.inputs:topicName", depth_topic_name),
                ("DepthPublish.inputs:frameId", frame_id),
                ("DepthPublish.inputs:resetSimulationTimeOnStop", True),
            ]
        )

        pcl_topic_name = None
        if topic_name in depth_pcl_topics:
            pcl_topic_name = f"points_{frame_id}"
            create_nodes.append(
                ("DepthPCLPublish", "isaacsim.ros2.bridge.ROS2CameraHelper")
            )
            connect.extend(
                [
                    (
                        "RenderProduct.outputs:execOut",
                        "DepthPCLPublish.inputs:execIn",
                    ),
                    (
                        "RenderProduct.outputs:renderProductPath",
                        "DepthPCLPublish.inputs:renderProductPath",
                    ),
                    (
                        "Context.outputs:context",
                        "DepthPCLPublish.inputs:context",
                    ),
                ]
            )
            set_values.extend(
                [
                    ("DepthPCLPublish.inputs:type", "depth_pcl"),
                    ("DepthPCLPublish.inputs:nodeNamespace", node_namespace),
                    ("DepthPCLPublish.inputs:topicName", pcl_topic_name),
                    ("DepthPCLPublish.inputs:frameId", frame_id),
                    (
                        "DepthPCLPublish.inputs:resetSimulationTimeOnStop",
                        True,
                    ),
                ]
            )

        og.Controller.edit(
            {"graph_path": graph_path, "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: create_nodes,
                keys.CONNECT: connect,
                keys.SET_VALUES: set_values,
            },
        )
        if pcl_topic_name is not None:
            print(
                f"ROS2 depth_pcl publisher added to {graph_path} "
                f"(topic=/{pcl_topic_name}, frameId={frame_id})"
            )

        print(
            f"ROS2 depth publisher added to {graph_path} "
            f"(topic=/{depth_topic_name}, frameId={frame_id})"
        )
        print(
            f"ROS2 camera publisher graph created: {graph_path} "
            f"(cameraPrim={camera_prim_path}, topic=/{topic_name})"
        )

    if camera_tf_prim_paths:
        # Camera prims are not in the URDF `robot_state_publisher` uses (the
        # Robotiq+camera USD is a different asset than the Franka-Hand URDF
        # MoveIt loads -- see GOTCHAS), so /tf has no entry for any camera
        # link at all: a perception node cannot place a detected pixel in
        # `odom` without one. Same live-prim-transform pattern as
        # `publish_ros2_object_tf` above -- `ROS2PublishTransformTree` reads
        # each target prim's own world transform every tick, correct for
        # both the fixed head camera and the two wrist cameras that move
        # with the arm.
        og.Controller.edit(
            {"graph_path": "/ROS2_CameraTFGraph", "evaluator_name": "execution"},
            {
                keys.CREATE_NODES: [
                    ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                    ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                    (
                        "PublishCameraTF",
                        "isaacsim.ros2.bridge.ROS2PublishTransformTree",
                    ),
                ],
                keys.CONNECT: [
                    (
                        "OnPlaybackTick.outputs:tick",
                        "PublishCameraTF.inputs:execIn",
                    ),
                    (
                        "Context.outputs:context",
                        "PublishCameraTF.inputs:context",
                    ),
                    (
                        "OnPlaybackTick.outputs:time",
                        "PublishCameraTF.inputs:timeStamp",
                    ),
                ],
                keys.SET_VALUES: [
                    ("PublishCameraTF.inputs:targetPrims", camera_tf_prim_paths),
                ],
            },
        )
        print(
            "ROS2 camera TF publisher graph created: /ROS2_CameraTFGraph "
            f"(targetPrims={camera_tf_prim_paths})"
        )


def launch_isaac_sim(args: argparse.Namespace) -> None:
    if not ISAACSIM_LAUNCHER.is_file():
        raise FileNotFoundError(
            f"Isaac Sim launcher not found: {ISAACSIM_LAUNCHER}"
        )

    configure_ros2_bridge_env(args)

    command = [
        str(ISAACSIM_LAUNCHER),
        ISAACSIM_EXPERIENCES[args.experience],
    ]
    if args.headless:
        command.append("--no-window")
    command.extend(["--exec", str(Path(__file__).resolve()), "--inside-kit"])
    env = os.environ.copy()
    env[INSIDE_KIT_ENV_VAR] = "1"
    env[INNER_ARGV_ENV_VAR] = json.dumps(
        [arg for arg in sys.argv[1:] if arg != "--inside-kit"]
    )
    os.chdir("/isaac-sim")
    os.execvpe(command[0], command, env)


def set_xform(
    prim: Any,
    position: tuple[float, float, float],
    rotation: tuple[float, float, float, float],
) -> None:
    from pxr import Gf as pxr_gf
    from pxr import UsdGeom as pxr_usd_geom

    Gf: Any = pxr_gf
    UsdGeom: Any = pxr_usd_geom

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Vec3d(*position)
    )
    xform.AddOrientOp(UsdGeom.XformOp.PrecisionFloat).Set(
        Gf.Quatf(rotation[0], rotation[1], rotation[2], rotation[3])
    )


def reference_usd(
    stage: Any,
    prim_path: str,
    usd_path: Path,
    position: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rotation: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0),
    reset_asset_xform: bool = False,
) -> Any:
    from pxr import UsdGeom as pxr_usd_geom

    UsdGeom: Any = pxr_usd_geom

    parent_prim = UsdGeom.Xform.Define(stage, prim_path).GetPrim()
    set_xform(parent_prim, position, rotation)

    asset_prim = UsdGeom.Xform.Define(stage, f"{prim_path}/Asset").GetPrim()
    asset_prim.GetReferences().AddReference(str(usd_path.resolve()))
    if reset_asset_xform:
        set_xform(asset_prim, (0.0, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
    return asset_prim


def override_gravity_scale(stage: Any, scale: float) -> None:
    """Diagnostic only (`--gravity-scale`): create an explicit
    `/PhysicsScene` with `gravityMagnitude = 9.81 * scale`. With no
    authored scene, Isaac Sim applies its own implicit default (standard
    Earth gravity) -- this makes that override explicit and controllable,
    to test whether a residual joint tracking error is gravity-load droop
    (scale=0.0 removes the load; a droop mechanism should then vanish) or
    something else (a non-gravity mechanism will not respond to this).
    """
    from pxr import Gf, UsdPhysics

    scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(9.81 * scale)
    print(f"Gravity scale override applied: {scale} (magnitude={9.81 * scale})")


def remove_embedded_physics_scenes(stage: Any, root_prim: Any) -> list[str]:
    from pxr import Usd as pxr_usd
    from pxr import UsdPhysics as pxr_usd_physics

    Usd: Any = pxr_usd
    UsdPhysics: Any = pxr_usd_physics

    paths_to_remove = [
        str(prim.GetPath())
        for prim in Usd.PrimRange(root_prim)
        if prim.IsA(UsdPhysics.Scene)
    ]
    for prim_path in paths_to_remove:
        stage.OverridePrim(prim_path).SetActive(False)
    return paths_to_remove


def move_task3_head(
    stage: Any,
    room_asset_path: str,
    position: tuple[float, float, float],
    orientation: tuple[float, float, float, float],
) -> str:
    candidate_paths = (
        f"{room_asset_path}/head",
        f"{room_asset_path}/root/head",
        "/root/head",
    )
    for prim_path in candidate_paths:
        prim = stage.GetPrimAtPath(prim_path)
        if prim and prim.IsValid():
            set_head_xform_orient(prim, position, orientation)
            return prim_path

    raise RuntimeError(
        "Could not find task3 head prim. Tried: " + ", ".join(candidate_paths)
    )


def create_preview_material(
    stage: Any,
    path: str,
    diffuse_color: tuple[float, float, float],
    metallic: float = 0.0,
    roughness: float = 0.5,
) -> Any:
    from pxr import Gf as pxr_gf
    from pxr import Sdf as pxr_sdf
    from pxr import UsdShade as pxr_usd_shade

    Gf: Any = pxr_gf
    Sdf: Any = pxr_sdf
    UsdShade: Any = pxr_usd_shade

    material = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
        Gf.Vec3f(*diffuse_color)
    )
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(metallic)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness)
    surface_output = shader.CreateOutput("surface", Sdf.ValueTypeNames.Token)
    material.CreateSurfaceOutput().ConnectToSource(surface_output)
    return material


def apply_physics_material(
    material: Any,
    friction: float,
    restitution: float,
) -> None:
    from pxr import UsdPhysics as pxr_usd_physics

    UsdPhysics: Any = pxr_usd_physics

    physics_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    physics_api.CreateStaticFrictionAttr(friction)
    physics_api.CreateDynamicFrictionAttr(friction)
    physics_api.CreateRestitutionAttr(restitution)


def usd_world_bounds(
    path: Path,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    from pxr import Usd as pxr_usd
    from pxr import UsdGeom as pxr_usd_geom

    Usd: Any = pxr_usd
    UsdGeom: Any = pxr_usd_geom

    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open USD stage: {path}")

    purposes = [
        UsdGeom.Tokens.default_,
        UsdGeom.Tokens.render,
        UsdGeom.Tokens.proxy,
    ]
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), purposes)
    bound_range = bbox_cache.ComputeWorldBound(
        stage.GetPseudoRoot()
    ).ComputeAlignedRange()
    bound_min = bound_range.GetMin()
    bound_max = bound_range.GetMax()
    return tuple(bound_min), tuple(bound_max)


def bean_spawn_positions(
    count: int,
    bowl_position: tuple[float, float, float],
) -> list[tuple[float, float, float]]:
    bowl_min_local, bowl_max_local = usd_world_bounds(BOWL_USD)
    container_min = tuple(
        bowl_min_local[index] + bowl_position[index] for index in range(3)
    )
    container_max = tuple(
        bowl_max_local[index] + bowl_position[index] for index in range(3)
    )
    container_center_xy = (
        0.5 * (container_min[0] + container_max[0]),
        0.5 * (container_min[1] + container_max[1]),
    )
    container_inner_radius = 0.5 * min(
        container_max[0] - container_min[0],
        container_max[1] - container_min[1],
    )
    bean_radius = BEAN_PHYSICS["radius"]
    bean_half_height = BEAN_PHYSICS["half_height"]
    bean_length = 2.0 * (bean_half_height + bean_radius)
    radial_margin = max(1.25 * bean_radius, 0.60 * bean_half_height)
    usable_radius = max(
        bean_radius,
        container_inner_radius
        - BEAN_PHYSICS["spawn_wall_thickness"]
        - radial_margin,
    )
    layer_height = max(2.4 * bean_radius, 0.9 * bean_length)
    spawn_bottom_z = bowl_position[2] + BEAN_PHYSICS["spawn_height"]
    ring_spacing = BEAN_PHYSICS["spawn_spacing_scale"] * max(
        2.8 * bean_radius,
        0.92 * bean_length,
    )
    angular_spacing = BEAN_PHYSICS["spawn_spacing_scale"] * max(
        2.6 * bean_radius,
        0.8 * bean_length,
    )

    positions = []
    layer_index = 0
    while len(positions) < count:
        z = spawn_bottom_z + layer_index * layer_height
        ring_phase = 0.5 * math.pi * (layer_index % 4)

        positions.append((container_center_xy[0], container_center_xy[1], z))
        if len(positions) >= count:
            break

        ring_radius = ring_spacing
        while ring_radius <= usable_radius and len(positions) < count:
            circumference = 2.0 * math.pi * ring_radius
            count_on_ring = max(6, int(circumference / angular_spacing))
            angle_step = 2.0 * math.pi / count_on_ring
            for ring_index in range(count_on_ring):
                angle = ring_phase + ring_index * angle_step
                radial_jitter = random.uniform(
                    -0.08 * ring_spacing,
                    0.08 * ring_spacing,
                )
                theta_jitter = random.uniform(-0.08, 0.08) * angle_step
                current_radius = min(
                    usable_radius,
                    max(bean_radius, ring_radius + radial_jitter),
                )
                x = current_radius * math.cos(angle + theta_jitter)
                y = current_radius * math.sin(angle + theta_jitter)
                if x * x + y * y > usable_radius * usable_radius:
                    continue
                positions.append(
                    (
                        container_center_xy[0] + x,
                        container_center_xy[1] + y,
                        z
                        + random.uniform(
                            -0.08 * bean_radius,
                            0.08 * bean_radius,
                        ),
                    )
                )
                if len(positions) >= count:
                    break
            ring_radius += ring_spacing
        layer_index += 1
    return positions[:count]


def add_coffee_beans(
    stage: Any,
    count: int,
    color: tuple[float, float, float],
    density: float,
    bowl_position: tuple[float, float, float],
    dynamic: bool = True,
) -> None:
    if count <= 0:
        return

    from pxr import UsdGeom as pxr_usd_geom
    from pxr import UsdPhysics as pxr_usd_physics
    from pxr import UsdShade as pxr_usd_shade

    UsdGeom: Any = pxr_usd_geom
    UsdPhysics: Any = pxr_usd_physics
    UsdShade: Any = pxr_usd_shade

    UsdGeom.Scope.Define(stage, "/World/Scene")
    UsdGeom.Scope.Define(stage, "/World/Scene/CoffeeBeans")
    UsdGeom.Scope.Define(stage, "/World/Looks")
    material = create_preview_material(
        stage,
        "/World/Looks/CoffeeBean",
        diffuse_color=color,
        metallic=0.0,
        roughness=0.8,
    )
    apply_physics_material(
        material,
        friction=BEAN_PHYSICS["friction"],
        restitution=BEAN_PHYSICS["restitution"],
    )

    radius = BEAN_PHYSICS["radius"]
    half_height = BEAN_PHYSICS["half_height"]

    positions = bean_spawn_positions(count, bowl_position)
    for index, position in enumerate(positions):
        bean_prim_path = f"/World/Scene/CoffeeBeans/Bean_{index:04d}"
        bean = UsdGeom.Capsule.Define(stage, bean_prim_path)
        bean.CreateRadiusAttr(radius)
        bean.CreateHeightAttr(2.0 * half_height)
        bean.CreateAxisAttr("X")
        bean_prim = bean.GetPrim()

        yaw = random.uniform(0.0, 2.0 * math.pi)
        set_xform(bean_prim, position, yaw_to_quat(math.degrees(yaw)))

        UsdPhysics.CollisionAPI.Apply(bean_prim)
        if dynamic:
            UsdPhysics.RigidBodyAPI.Apply(bean_prim)
            mass_api = UsdPhysics.MassAPI.Apply(bean_prim)
            mass_api.CreateDensityAttr(density)
        UsdShade.MaterialBindingAPI.Apply(bean_prim).Bind(material)


def load_deformable_assets(
    stage: Any,
) -> None:
    root_position = TASK2_TABLE_POSITION
    asset_root_path = "/World/Scene/task_objects"

    for asset_key, asset_config in TASK2_OBJECT_SPAWN_CONFIG.items():
        if asset_key in ("boards",):
            for i, board_spawn in enumerate(asset_config["spawns"]):
                reference_usd(
                    stage,
                    f"{asset_root_path}/board_{i}",
                    asset_path(asset_config["asset_path"]),
                    position=tuple(
                        root_position[index] + board_spawn["position"][index]
                        for index in range(3)
                    ),
                    rotation=board_spawn["rotation"],
                )
            continue
        reference_usd(
            stage,
            f"{asset_root_path}/{asset_key}",
            asset_path(asset_config["asset_path"]),
            position=tuple(
                root_position[index] + asset_config["position"][index]
                for index in range(3)
            ),
            rotation=asset_config["rotation"],
        )


def setup_deformable_camera(
    stage: Any,
) -> None:
    import omni.graph.core as og
    from pxr import Gf as pxr_gf
    from pxr import UsdGeom as pxr_usd_geom

    Gf: Any = pxr_gf
    UsdGeom: Any = pxr_usd_geom

    # Creating a Camera Prim
    camera_prim_path = "/World/Scene/eval_camera"
    camera_prim = UsdGeom.Camera.Define(stage, camera_prim_path)
    xform_api = UsdGeom.XformCommonAPI(camera_prim)
    xform_api.SetTranslate(Gf.Vec3d(*TASK2_CAMERA_POSITION))
    xform_api.SetRotate((0, 0, 0), UsdGeom.XformCommonAPI.RotationOrderXYZ)
    camera_prim.GetFocalLengthAttr().Set(20)
    camera_prim.GetFocusDistanceAttr().Set(400)
    camera_prim.GetProjectionAttr().Set("perspective")
    # camera_prim.GetHorizontalApertureAttr().Set(21)
    # camera_prim.GetVerticalApertureAttr().Set(16)

    # ROS2 helper
    ROS_TOPIC_NAMESPACE = "/isaac/eval_camera"
    ROS_TOPIC_FRAMEID = "eval_camera"

    keys = og.Controller.Keys
    (ros_camera_graph, _, _, _) = og.Controller.edit(
        {
            "graph_path": "/ROS2_CameraGraphs/eval_camera",
            "evaluator_name": "execution",
        },
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                (
                    "CameraInfoPublish",
                    "isaacsim.ros2.bridge.ROS2CameraInfoHelper",
                ),
                (
                    "RenderProduct",
                    "isaacsim.core.nodes.IsaacCreateRenderProduct",
                ),
                (
                    "RunOnce",
                    "isaacsim.core.nodes.OgnIsaacRunOneSimulationFrame",
                ),
                ("Context", "isaacsim.ros2.bridge.ROS2Context"),
                ("RGBPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("DepthPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                ("SemanticPublish", "isaacsim.ros2.bridge.ROS2CameraHelper"),
                (
                    "Bbox2dTightPublish",
                    "isaacsim.ros2.bridge.ROS2CameraHelper",
                ),
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "RunOnce.inputs:execIn"),
                ("RunOnce.outputs:step", "RenderProduct.inputs:execIn"),
                (
                    "RenderProduct.outputs:execOut",
                    "CameraInfoPublish.inputs:execIn",
                ),
                (
                    "RenderProduct.outputs:renderProductPath",
                    "CameraInfoPublish.inputs:renderProductPath",
                ),
                (
                    "Context.outputs:context",
                    "CameraInfoPublish.inputs:context",
                ),
                ("RenderProduct.outputs:execOut", "RGBPublish.inputs:execIn"),
                (
                    "RenderProduct.outputs:renderProductPath",
                    "RGBPublish.inputs:renderProductPath",
                ),
                (
                    "RenderProduct.outputs:execOut",
                    "DepthPublish.inputs:execIn",
                ),
                (
                    "RenderProduct.outputs:renderProductPath",
                    "DepthPublish.inputs:renderProductPath",
                ),
                (
                    "RenderProduct.outputs:execOut",
                    "SemanticPublish.inputs:execIn",
                ),
                (
                    "RenderProduct.outputs:renderProductPath",
                    "SemanticPublish.inputs:renderProductPath",
                ),
                ("Context.outputs:context", "SemanticPublish.inputs:context"),
                (
                    "RenderProduct.outputs:execOut",
                    "Bbox2dTightPublish.inputs:execIn",
                ),
                (
                    "RenderProduct.outputs:renderProductPath",
                    "Bbox2dTightPublish.inputs:renderProductPath",
                ),
                (
                    "Context.outputs:context",
                    "Bbox2dTightPublish.inputs:context",
                ),
            ],
            keys.SET_VALUES: [
                # Render Product
                ("RenderProduct.inputs:cameraPrim", camera_prim_path),
                ("RenderProduct.inputs:height", 720),
                ("RenderProduct.inputs:width", 1280),
                # Publisher: Camera Info
                ("CameraInfoPublish.inputs:topicName", "camera_info"),
                ("CameraInfoPublish.inputs:frameId", ROS_TOPIC_FRAMEID),
                (
                    "CameraInfoPublish.inputs:nodeNamespace",
                    ROS_TOPIC_NAMESPACE,
                ),
                ("CameraInfoPublish.inputs:resetSimulationTimeOnStop", True),
                # Publisher: RGB
                ("RGBPublish.inputs:type", "rgb"),
                ("RGBPublish.inputs:nodeNamespace", ROS_TOPIC_NAMESPACE),
                ("RGBPublish.inputs:topicName", "image_raw"),
                ("RGBPublish.inputs:frameId", ROS_TOPIC_FRAMEID),
                ("RGBPublish.inputs:resetSimulationTimeOnStop", True),
                # Publisher: Depth
                ("DepthPublish.inputs:type", "depth"),
                ("DepthPublish.inputs:nodeNamespace", ROS_TOPIC_NAMESPACE),
                ("DepthPublish.inputs:topicName", "depth"),
                ("DepthPublish.inputs:frameId", ROS_TOPIC_FRAMEID),
                ("DepthPublish.inputs:resetSimulationTimeOnStop", True),
                # Publisher: Semantic Segmentation
                ("SemanticPublish.inputs:topicName", "semantic_segmentation"),
                ("SemanticPublish.inputs:type", "semantic_segmentation"),
                ("SemanticPublish.inputs:frameId", ROS_TOPIC_FRAMEID),
                ("SemanticPublish.inputs:nodeNamespace", ROS_TOPIC_NAMESPACE),
                ("SemanticPublish.inputs:enableSemanticLabels", True),
                ("SemanticPublish.inputs:resetSimulationTimeOnStop", True),
                # Publisher: 2D Bounding Box Tight
                ("Bbox2dTightPublish.inputs:topicName", "bbox_2d_tight"),
                ("Bbox2dTightPublish.inputs:type", "bbox_2d_tight"),
                ("Bbox2dTightPublish.inputs:resetSimulationTimeOnStop", True),
                ("Bbox2dTightPublish.inputs:frameId", ROS_TOPIC_FRAMEID),
                (
                    "Bbox2dTightPublish.inputs:nodeNamespace",
                    ROS_TOPIC_NAMESPACE,
                ),
                ("Bbox2dTightPublish.inputs:enableSemanticLabels", True),
            ],
        },
    )


def set_initial_perspective_view(app: Any) -> None:
    if not INITIAL_VIEW_POSE:
        return

    position, rotation = INITIAL_VIEW_POSE
    rotation_quat = euler_xyz_to_quat(rotation)
    camera_path = "/OmniverseKit_Persp"

    try:
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.viewport.utility.camera_state import ViewportCameraState
        from pxr import Gf as pxr_gf

        Gf: Any = pxr_gf
        viewport = get_active_viewport()
        if viewport is not None:
            viewport.camera_path = camera_path
            try:
                camera_state = ViewportCameraState(camera_path, viewport)
            except TypeError:
                camera_state = ViewportCameraState(camera_path)
            camera_state.set_position_world(Gf.Vec3d(*position), True)
            camera_state.set_rotation_world(Gf.Quatd(*rotation_quat), True)
            app.update()
            return
    except Exception as exc:
        print(f"Viewport pose API unavailable: {exc}")


def configure_robot_room_stage(
    app: Any,
    stage: Any,
    room_path: Path,
    task: str,
    head_placement: str,
    *,
    robot_path: Path | None = None,
    robot_position: tuple[float, float, float] | None = None,
    robot_rotation: tuple[float, float, float, float] | None = None,
    robot_yaw: float | None = None,
    dynamic_beans: bool = True,
) -> Any:
    from pxr import UsdGeom as pxr_usd_geom
    from pxr import UsdLux as pxr_usd_lux

    UsdGeom: Any = pxr_usd_geom
    UsdLux: Any = pxr_usd_lux

    stage.SetFramesPerSecond(60.0)
    stage.SetTimeCodesPerSecond(60.0)
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)

    world = UsdGeom.Xform.Define(stage, "/World")
    stage.SetDefaultPrim(world.GetPrim())
    UsdGeom.Scope.Define(stage, "/World/Environment")

    print(f"PROGRESS: referencing room USD {room_path}", flush=True)
    room_asset_prim = reference_usd(
        stage,
        "/World/Environment/RobotRoom",
        room_path,
        reset_asset_xform=True,
    )
    print("PROGRESS: room USD referenced", flush=True)
    removed_physics_scenes = remove_embedded_physics_scenes(
        stage,
        room_asset_prim,
    )
    if removed_physics_scenes:
        print(
            "Disabled embedded room physics scenes: "
            + ", ".join(removed_physics_scenes),
            flush=True,
        )
    if (
        robot_path is not None
        and robot_position is not None
        and robot_rotation is not None
    ):
        reference_usd(
            stage,
            "/World/Robot",
            robot_path,
            robot_position,
            robot_rotation,
        )

    resolved_head_placement = None
    head_prim_path = None
    if task == "task1":
        pass
    elif task == "task2":
        load_deformable_assets(stage)
        setup_deformable_camera(stage)
    elif task == "task3":
        (
            resolved_head_placement,
            head_position,
            head_orientation,
        ) = resolve_head_placement(head_placement)
        head_prim_path = move_task3_head(
            stage,
            str(room_asset_prim.GetPath()),
            head_position,
            head_orientation,
        )
        add_coffee_beans(
            stage,
            count=DEFAULT_BEAN_COUNT,
            color=DEFAULT_BEAN_COLOR,
            density=DEFAULT_BEAN_DENSITY,
            bowl_position=TASK3_BOWL_POSITION,
            dynamic=dynamic_beans,
        )

    dome = UsdLux.DomeLight.Define(stage, "/World/Light")
    dome.CreateIntensityAttr(3000.0)

    for _ in range(10):
        app.update()

    set_initial_perspective_view(app)

    print("=" * 80)
    print("Robot room loaded in Isaac Sim")
    print("=" * 80)
    print(f"Room USD: {room_path}")
    if robot_path is not None:
        print(f"Robot USD: {robot_path}")
    if robot_position is not None:
        print(
            "Robot start: "
            f"({robot_position[0]:.3f}, {robot_position[1]:.3f}, "
            f"{robot_position[2]:.3f})"
        )
    if robot_yaw is not None:
        print(f"Robot yaw: {robot_yaw:.1f} deg")
    bean_mode = "dynamic" if dynamic_beans else "static"
    bean_count = DEFAULT_BEAN_COUNT if task == "task3" else 0
    print(f"Coffee beans: {bean_count} ({bean_mode})")
    if resolved_head_placement and head_prim_path:
        print(f"Head placement: {resolved_head_placement}")
        print(f"Head prim: {head_prim_path}")
    return stage


def build_stage(
    app: Any,
    room_path: Path,
    robot_path: Path,
    task: str,
    robot_position: tuple[float, float, float],
    robot_rotation: tuple[float, float, float, float],
    robot_yaw: float,
    head_placement: str,
) -> Any:
    import omni.usd

    context = omni.usd.get_context()
    context.new_stage()
    for _ in range(10):
        app.update()

    stage = context.get_stage()
    if stage is None:
        raise RuntimeError("Could not create an Isaac Sim stage.")

    return configure_robot_room_stage(
        app,
        stage,
        room_path=room_path,
        task=task,
        head_placement=head_placement,
        robot_path=robot_path,
        robot_position=robot_position,
        robot_rotation=robot_rotation,
        robot_yaw=robot_yaw,
    )


def make_robot_actuator_cfgs(
    implicit_actuator_cfg: Any, gripper: str | None = None
) -> dict[str, Any]:
    return {
        name: implicit_actuator_cfg(**spec)
        for name, spec in robot_actuator_cfg_specs(gripper=gripper).items()
    }


def make_control_scene_cfg(
    *,
    num_envs: int,
    robot_path: Path,
    robot_position: tuple[float, float, float],
    robot_rotation: tuple[float, float, float, float],
    gripper: str | None = None,
) -> Any:
    import isaaclab.sim as sim_utils
    from isaaclab.actuators import ImplicitActuatorCfg
    from isaaclab.assets import ArticulationCfg
    from isaaclab.scene import InteractiveSceneCfg

    scene_cfg = InteractiveSceneCfg(num_envs=num_envs, env_spacing=10.0)
    scene_cfg.robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(usd_path=str(robot_path)),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=robot_position,
            rot=robot_rotation,
            joint_pos=INITIAL_ROBOT_JOINT_POS,
        ),
        actuators=make_robot_actuator_cfgs(
            ImplicitActuatorCfg, gripper=gripper
        ),
    )
    return scene_cfg


def disable_robot_external_wrenches(robot: Any) -> None:
    """Keep Isaac Lab from applying unused external link wrenches."""
    for composer_name in (
        "instantaneous_wrench_composer",
        "permanent_wrench_composer",
    ):
        composer = getattr(robot, composer_name, None)
        if composer is not None:
            composer.reset()


def require_single_teleop_environment(num_envs: int) -> None:
    """Lula's Core articulation wrapper controls one Task 3 robot."""
    if num_envs != 1:
        raise RuntimeError(
            "Keyboard dual-arm IK requires exactly one environment; "
            f"received num_envs={num_envs}."
        )


MOTION_GENERATION_EXTENSION = "isaacsim.robot_motion.motion_generation"


def enable_motion_generation_extension(extension_manager: Any) -> None:
    """Enable the Lula extension before importing its Python package."""
    if extension_manager.is_extension_enabled(MOTION_GENERATION_EXTENSION):
        return
    enabled = extension_manager.set_extension_enabled_immediate(
        MOTION_GENERATION_EXTENSION, True
    )
    if enabled is False:
        raise RuntimeError(
            "Could not enable Isaac Sim motion-generation extension: "
            f"{MOTION_GENERATION_EXTENSION}"
        )


def measured_position_targets(robot: Any) -> Any:
    """Snapshot measured joints once as persistent position targets."""
    return robot.data.joint_pos.detach().clone()


def reset_robot_to_default_state(robot: Any, env_origins: Any) -> None:
    """Write the configured Isaac Lab initial state into PhysX."""
    root_state = robot.data.default_root_state.clone()
    root_state[:, :3] += env_origins
    joint_positions = robot.data.default_joint_pos.clone()
    joint_velocities = robot.data.default_joint_vel.clone()

    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(root_state[:, 7:])
    robot.write_joint_state_to_sim(joint_positions, joint_velocities)
    robot.set_joint_position_target(joint_positions)
    robot.set_joint_velocity_target(joint_velocities)


def robot_root_world_pose(
    robot: Any,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Read the first environment's root pose in Isaac Lab wxyz order."""
    position = robot.data.root_pos_w[0].detach().cpu().tolist()
    orientation = robot.data.root_quat_w[0].detach().cpu().tolist()
    return tuple(position), tuple(orientation)


def clamp_direct_joint_command(command: Any, robot: Any, groups: Any) -> Any:
    """Clamp present direct targets to the articulation's soft limits."""
    if (
        command.left_joint_positions is None
        and command.right_joint_positions is None
    ):
        return command
    limits = getattr(robot.data, "soft_joint_pos_limits", None)
    if limits is None:
        raise RuntimeError(
            "Direct arm commands require robot.data.soft_joint_pos_limits"
        )
    required_ids = groups.left_arm + groups.right_arm
    if (
        limits.ndim != 3
        or limits.shape[0] < 1
        or limits.shape[2] != 2
        or not required_ids
        or limits.shape[1] <= max(required_ids)
    ):
        raise RuntimeError(
            "soft_joint_pos_limits must have shape (envs, joints, 2)"
        )
    from teleop_targets import clamp_arm_joint_positions

    updates = {}
    for side, values, joint_ids in (
        ("left", command.left_joint_positions, groups.left_arm),
        ("right", command.right_joint_positions, groups.right_arm),
    ):
        if values is None:
            continue
        lower = limits[0, list(joint_ids), 0].detach().cpu().tolist()
        upper = limits[0, list(joint_ids), 1].detach().cpu().tolist()
        updates[f"{side}_joint_positions"] = clamp_arm_joint_positions(
            values, lower, upper
        )
    return replace(command, **updates)


def configure_keyboard_control_stage(
    configure: Any,
    app: Any,
    stage: Any,
    **kwargs: Any,
) -> Any:
    """Configure room props only; InteractiveScene owns the sole robot."""
    return configure(app, stage, robot_path=None, **kwargs)


def run_with_app_cleanup(app: Any, callback: Any) -> Any:
    # `SimulationApp.close()` hard-exits the process, so a bare try/finally
    # here destroys any propagating exception before Python can print it --
    # a silent, clean-looking failure (see generate_occupancy_map.py history).
    # Print the traceback before close() gets the chance.
    try:
        return callback()
    except BaseException:
        import traceback

        print("RUN_WITH_APP_CLEANUP_FAILED -- traceback follows", flush=True)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        raise
    finally:
        app.close()


class PynputKeyboardTeleop:
    def __init__(self, keyboard_module: Any) -> None:
        self._keyboard = keyboard_module
        self.pressed: set[str] = set()
        self.stop_requested = False
        self._listener: Any | None = None

    def start(self) -> None:
        # The organizers' own Isaac Sim keyboard demo (EBiM-Benchmark/
        # benchmark, Robotiq_DEMO branch, scripts/scenes/keyboard_control.py)
        # uses suppress=True so Kit's own window/viewport hotkeys never see
        # a keystroke -- without it, a bare "f" both toggles our gripper and
        # fires Kit's Rotate-tool hotkey, and Alt+F both runs our command
        # and switches the viewport camera to the "Front" preset (both
        # observed live). But suppress=True also blocks Kit from ever
        # seeing the Alt key, which breaks every Alt+drag camera gesture
        # (orbit, pan) app-wide, not just the one conflicting hotkey --
        # confirmed live: only scroll-zoom kept working, Alt+orbit/pan did
        # not. That is a worse trade for teleoperation, where seeing the
        # robot matters more than avoiding a harmless duplicate hotkey, so
        # suppress stays False here despite the organizers' choice. pynput
        # is still preferred over KitKeyboardTeleop below because its
        # listener doesn't require window focus either way.
        self._listener = self._keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
            suppress=False,
        )
        self._listener.daemon = True
        self._listener.start()

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()

    def _on_press(self, key: Any) -> bool | None:
        self._update_pressed(key, add=True)
        if key == self._keyboard.Key.esc:
            self.stop_requested = True
            return False
        return None

    def _on_release(self, key: Any) -> bool | None:
        self._update_pressed(key, add=False)
        if key == self._keyboard.Key.esc:
            self.stop_requested = True
            return False
        return None

    def _update_pressed(self, key: Any, *, add: bool) -> None:
        key_name = None
        if hasattr(key, "char") and key.char:
            key_name = key.char.lower()
        elif hasattr(key, "name"):
            key_name = key.name

        if key_name is None:
            return
        key_name = normalize_keyboard_event_input(key_name)
        if add:
            self.pressed.add(key_name)
        else:
            self.pressed.discard(key_name)


class KitKeyboardTeleop:
    def __init__(self, carb_input: Any, appwindow: Any) -> None:
        self._carb_input = carb_input
        self._keyboard = appwindow.get_default_app_window().get_keyboard()
        self._input = carb_input.acquire_input_interface()
        self._subscription: Any | None = None
        self.pressed: set[str] = set()
        self.stop_requested = False

    def start(self) -> None:
        self._subscription = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            self._on_keyboard_event,
        )

    def stop(self) -> None:
        if self._subscription is None:
            return
        unsubscribe = getattr(
            self._input,
            "unsubscribe_to_keyboard_events",
            None,
        )
        if unsubscribe is not None:
            unsubscribe(self._keyboard, self._subscription)
        self._subscription = None

    def _on_keyboard_event(self, event: Any, *_args: Any) -> bool:
        key_name = normalize_keyboard_event_input(event.input)
        if key_name is None:
            return True

        event_type = event.type
        if event_type in (
            self._carb_input.KeyboardEventType.KEY_PRESS,
            self._carb_input.KeyboardEventType.KEY_REPEAT,
        ):
            self.pressed.add(key_name)
            if key_name == "esc":
                self.stop_requested = True
        elif event_type == self._carb_input.KeyboardEventType.KEY_RELEASE:
            self.pressed.discard(key_name)
            if key_name == "esc":
                self.stop_requested = True
        return True


def normalize_keyboard_event_input(key_input: Any) -> str | None:
    raw_name = getattr(key_input, "name", None)
    if raw_name is None:
        raw_name = str(key_input).rsplit(".", maxsplit=1)[-1]
    key_name = str(raw_name).lower()
    aliases = {
        "escape": "esc",
        "left_arrow": "left",
        "right_arrow": "right",
        "arrow_left": "left",
        "arrow_right": "right",
        "key_1": "1",
        "key_2": "2",
        "key_3": "3",
        "left_shift": "shift",
        "right_shift": "shift",
        "shift_l": "shift",
        "shift_r": "shift",
    }
    return aliases.get(key_name, key_name)


def create_keyboard_teleop() -> Any:
    # Prefer pynput (suppress=True) over Kit's own carb.input subscription --
    # this is the organizers' own proven approach (EBiM-Benchmark/benchmark,
    # Robotiq_DEMO branch, scripts/scenes/keyboard_control.py) and avoids
    # every key also being seen by Kit's built-in viewport/tool hotkeys.
    # carb.input remains the fallback for environments without pynput
    # installed.
    try:
        from pynput import keyboard

        return PynputKeyboardTeleop(keyboard)
    except ImportError:
        pass

    import carb.input
    import omni.appwindow

    return KitKeyboardTeleop(carb.input, omni.appwindow)


def print_keyboard_control_help(control_help: str) -> None:
    print("\n" + "=" * 80)
    print("Keyboard robot control enabled (direct dual-arm + Shift base map)")
    print("=" * 80)
    print(control_help)
    print("  ESC     stop keyboard listener and exit")
    print("  Ctrl+C  exit")
    print(
        "Tip: the listener is global, so the viewport does not need focus.\n"
    )


def run_keyboard_control(
    args: argparse.Namespace,
    *,
    room_path: Path,
    robot_path: Path,
    robot_position: tuple[float, float, float],
    robot_rotation: tuple[float, float, float, float],
    robot_yaw: float,
) -> None:
    configure_ros2_bridge_env(args)
    require_single_teleop_environment(args.num_envs)
    try:
        from isaaclab.app import AppLauncher
    except ImportError as exc:
        raise RuntimeError(
            "Keyboard robot control requires the Isaac Lab runtime. "
            "Run this in the isaac-lab Docker profile, or pass "
            "--no-keyboard-control to use the passive Isaac Sim viewer."
        ) from exc
    if args.livestream:
        if not args.public_ip:
            raise ValueError("--livestream requires --public-ip or PUBLIC_IP")
        os.environ["PUBLIC_IP"] = args.public_ip
    app_launcher = AppLauncher(
        {
            "headless": args.headless,
            "enable_cameras": bool(args.livestream),
            # Isaac Sim 5.1 public WebRTC mode is 1.  Mode 2 is private/NVCF
            # networking and leaves an Internet client connected to signaling
            # but without a usable public media endpoint.
            "livestream": 1 if args.livestream else -1,
        }
    )
    simulation_app = app_launcher.app
    run_with_app_cleanup(
        simulation_app,
        lambda: _run_keyboard_control_app(
            args,
            simulation_app=simulation_app,
            room_path=room_path,
            robot_path=robot_path,
            robot_position=robot_position,
            robot_rotation=robot_rotation,
            robot_yaw=robot_yaw,
        ),
    )


def _run_keyboard_control_app(
    args: argparse.Namespace,
    *,
    simulation_app: Any,
    room_path: Path,
    robot_path: Path,
    robot_position: tuple[float, float, float],
    robot_rotation: tuple[float, float, float, float],
    robot_yaw: float,
) -> None:
    from dual_arm_lula import (
        LEFT_ARM_JOINTS,
        RIGHT_ARM_JOINTS,
        create_raw_dual_arm_lula,
    )
    from integration_test import resolve_prim_path
    from keyboard_arm_teleop import KeyboardTeleopMapper, control_help
    from run_episode import (
        _fix_single_articulation_root,
        make_headless_robot_usd,
        prepare_rigid_body_view_path,
    )
    from teleop_commands import safe_command
    from teleop_recording import TeleopEpisodeRecorder
    from teleop_targets import (
        CartesianTargetTracker,
        DirectJointTargetLatch,
        Pose,
        TargetLimits,
        TeleopTargets,
        compose_position_targets,
        discover_joint_groups,
        pose_base_to_world,
        pose_world_to_base,
        position_target_subset,
    )
    from tmr_base_control import (
        compensate_yaw_rate,
        compute_drive_targets,
        find_drive_joint_ids,
        get_root_yaw,
    )

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    enable_ros2_bridge(simulation_app, args)
    publish_ros2_clock(args)

    # TODO: Improve the interactive real-time factor (and perceived arm speed)
    # by profiling the synchronous dual-arm IK loop and decimating IK/control
    # or rendering updates without changing the commanded physical velocities.
    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005,
        device=args.device,
        gravity=(0.0, 0.0, -9.81),
    )
    print("Creating SimulationContext...", flush=True)
    sim = SimulationContext(sim_cfg)
    print("SimulationContext ready.", flush=True)
    print("Configuring robot room stage...", flush=True)
    configure_keyboard_control_stage(
        configure_robot_room_stage,
        simulation_app,
        sim.stage,
        room_path=room_path,
        task=args.task,
        head_placement=args.head_placement,
        robot_position=robot_position,
        robot_yaw=robot_yaw,
        dynamic_beans=args.dynamic_beans,
    )
    print("Robot room stage configured.", flush=True)
    sim.set_camera_view(
        eye=[robot_position[0] + 3.5, robot_position[1] + 3.5, 2.5],
        target=[robot_position[0], robot_position[1], 0.5],
    )

    print("Creating Isaac Lab InteractiveScene...", flush=True)
    scene_cfg = make_control_scene_cfg(
        num_envs=args.num_envs,
        robot_path=make_headless_robot_usd(robot_path),
        robot_position=robot_position,
        robot_rotation=robot_rotation,
    )
    scene = InteractiveScene(scene_cfg)
    for object_name in ("simple_tray", "bowl2", "spoon2", "plate2", "cup"):
        object_path = resolve_prim_path(sim.stage, object_name)
        prepare_rigid_body_view_path(sim.stage, object_path)
    _fix_single_articulation_root(sim.stage, "/World/envs/env_0/Robot")
    print("InteractiveScene ready.", flush=True)
    if args.skip_initial_reset:
        print("Skipping initial simulation and scene reset.", flush=True)
    else:
        print("Resetting simulation...", flush=True)
        sim.reset()
        print("Simulation reset complete.", flush=True)
        print("Resetting scene...", flush=True)
        scene.reset()
        print("Scene reset complete.", flush=True)

    robot = scene["robot"]
    if args.skip_initial_reset:
        print("Skipping configured robot initial state write.", flush=True)
    else:
        print("Writing configured robot initial state to PhysX...", flush=True)
        reset_robot_to_default_state(robot, scene.env_origins)
        scene.write_data_to_sim()
        print("Configured robot initial state ready.", flush=True)
    print(
        f"Robot joints ({len(robot.joint_names)}): {robot.joint_names}",
        flush=True,
    )
    stabilization_steps = max(0, args.stabilization_steps)
    stabilization_targets = robot.data.default_joint_pos.clone()
    for index in range(stabilization_steps):
        robot.set_joint_position_target(stabilization_targets)
        disable_robot_external_wrenches(robot)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.cfg.dt)
        if index == 0 or (index + 1) % 50 == 0:
            print(
                f"Stabilizing robot... {index + 1}/{stabilization_steps}",
                flush=True,
            )

    print("Finding drive joint ids...", flush=True)
    # Same floor friction as the other launch path. Without it the wheels
    # spin at their velocity limit and the chassis does not move at all --
    # robot_room.usd authors no physics material, so the drive wheels have
    # nothing to push against, while probe_base_drive.py rolls perfectly on a
    # plain GroundPlaneCfg.
    if getattr(args, "ground_collider", False):
        add_room_ground_collider(sim.stage, "/World/Environment/RobotRoom")
    if getattr(args, "floor_friction", 0.0) > 0.0:
        apply_floor_friction(
            sim.stage, "/World/Environment/RobotRoom", args.floor_friction
        )

    steering_indices, drive_indices = find_drive_joint_ids(robot.joint_names)
    print("Drive joint ids ready.", flush=True)
    joint_groups = discover_joint_groups(robot.joint_names)
    position_targets = measured_position_targets(robot)

    print("Enabling Isaac Sim motion-generation extension...", flush=True)
    import omni.kit.app

    enable_motion_generation_extension(
        omni.kit.app.get_app().get_extension_manager()
    )
    print("Isaac Sim motion-generation extension ready.", flush=True)
    print("Creating raw Lula NumPy joint-state bridge...", flush=True)
    dual_arm_ik = create_raw_dual_arm_lula(
        robot.joint_names,
        lambda: robot.data.joint_pos[0].detach().cpu().numpy(),
    )
    root_position, root_orientation = robot_root_world_pose(robot)
    initial_spine = float(position_targets[0, joint_groups.spine[0]].item())
    left_world_values, right_world_values = (
        dual_arm_ik.current_end_effector_poses(
            root_position,
            root_orientation,
            initial_spine,
        )
    )
    left_relative = pose_world_to_base(
        Pose(tuple(left_world_values[0]), tuple(left_world_values[1])),
        root_position,
        root_orientation,
    )
    right_relative = pose_world_to_base(
        Pose(tuple(right_world_values[0]), tuple(right_world_values[1])),
        root_position,
        root_orientation,
    )
    initial_left_gripper = float(
        position_targets[0, joint_groups.left_gripper[0]].item()
    )
    initial_right_gripper = float(
        position_targets[0, joint_groups.right_gripper[0]].item()
    )
    tracker = CartesianTargetTracker(
        TeleopTargets(
            left=left_relative,
            right=right_relative,
            left_gripper=initial_left_gripper,
            right_gripper=initial_right_gripper,
            spine=initial_spine,
        ),
        limits=TargetLimits(
            position_min=(-1.5, -1.5, -0.5),
            position_max=(1.5, 1.5, 2.5),
            gripper_min=0.0,
            gripper_max=1.0,
            spine_min=0.0,
            spine_max=0.85,
        ),
    )
    mapper = KeyboardTeleopMapper()
    direct_joint_latch = DirectJointTargetLatch()
    print("Dual-arm Lula controller ready.", flush=True)
    print("Reading root yaw...", flush=True)
    heading_hold_yaw = get_root_yaw(robot)
    print("Root yaw ready.", flush=True)
    print("Creating keyboard teleop backend...", flush=True)
    teleop = create_keyboard_teleop()
    print("Keyboard teleop backend ready.", flush=True)
    print(f"Active steering joints: {steering_indices}", flush=True)
    print(f"Active drive joints: {drive_indices}", flush=True)
    print_keyboard_control_help(control_help())

    count = 0
    listener_started = False
    recorder = None
    if args.record_teleop:
        episode_name = args.episode_name
        if episode_name is None:
            episode_name = time.strftime("tray_probe_%Y%m%d_%H%M%S")
        recorder = TeleopEpisodeRecorder(
            args.record_dir,
            episode_name,
            sample_every_steps=args.record_every_steps,
            metadata={
                "task": args.task,
                "head_placement": args.head_placement,
                "robot_position": list(robot_position),
                "robot_yaw": robot_yaw,
                "control_help": control_help(),
            },
        )
        print(f"Teleop recording: {recorder.output_path}", flush=True)
    session_started = time.monotonic()
    stop_reason = "operator_exit"
    try:
        teleop.start()
        listener_started = True
        print("Keyboard teleop listener started.", flush=True)
        while simulation_app.is_running() and not teleop.stop_requested:
            now = time.monotonic()
            if (
                args.max_seconds > 0.0
                and now - session_started >= args.max_seconds
            ):
                stop_reason = "max_seconds"
                print("Maximum teleop duration reached.", flush=True)
                break
            command = mapper.map_keys(
                set(teleop.pressed), timestamp=now, dt=sim.cfg.dt
            )
            command = safe_command(command, now=now, timeout=0.25)
            command = clamp_direct_joint_command(command, robot, joint_groups)

            vx, vy, wz_cmd = command.base_twist
            wz, heading_hold_yaw = compensate_yaw_rate(
                robot,
                vx,
                vy,
                wz_cmd,
                heading_hold_yaw,
                manual_rotation=abs(wz_cmd) > 1.0e-4,
            )

            targets = tracker.apply(command)
            root_position, root_orientation = robot_root_world_pose(robot)
            left_world = pose_base_to_world(
                targets.left, root_position, root_orientation
            )
            right_world = pose_base_to_world(
                targets.right, root_position, root_orientation
            )
            ik_result = dual_arm_ik.solve(
                left_world.position,
                right_world.position,
                left_world.orientation_wxyz,
                right_world.orientation_wxyz,
                spine_position=targets.spine,
                base_position=root_position,
                base_orientation_wxyz=root_orientation,
            )
            left_arm_targets, right_arm_targets = direct_joint_latch.select(
                command,
                ik_result,
                LEFT_ARM_JOINTS,
                RIGHT_ARM_JOINTS,
            )
            position_targets = compose_position_targets(
                position_targets,
                joint_groups,
                left_arm=left_arm_targets,
                right_arm=right_arm_targets,
                left_gripper=targets.left_gripper,
                right_gripper=targets.right_gripper,
                spine=targets.spine,
            )
            arm_position_targets, arm_position_joint_ids = (
                position_target_subset(position_targets, joint_groups)
            )
            robot.set_joint_position_target(
                arm_position_targets,
                joint_ids=arm_position_joint_ids,
            )

            steering_pos_targets, drive_vel_targets = compute_drive_targets(
                robot,
                steering_indices,
                vx,
                vy,
                wz,
                num_envs=args.num_envs,
                device=sim.device,
            )
            robot.set_joint_position_target(
                steering_pos_targets,
                joint_ids=steering_indices,
            )
            robot.set_joint_velocity_target(
                drive_vel_targets,
                joint_ids=drive_indices,
            )

            disable_robot_external_wrenches(robot)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.cfg.dt)

            if recorder is not None:
                recorder.record(
                    step=count,
                    sim_time=count * sim.cfg.dt,
                    keys=set(teleop.pressed),
                    command=command,
                    targets=targets,
                    root_position=root_position,
                    root_orientation=root_orientation,
                    left_world=left_world,
                    right_world=right_world,
                )

            count += 1
            if count % 400 == 0 and (vx != 0.0 or vy != 0.0 or wz != 0.0):
                print(
                    f"step={count} vx={vx:+.2f} vy={vy:+.2f} "
                    f"wz={wz:+.2f} keys={sorted(teleop.pressed)}"
                )
    except KeyboardInterrupt:
        stop_reason = "keyboard_interrupt"
        print("\nStopped by user.")
    finally:
        if listener_started:
            teleop.stop()
        if recorder is not None:
            recorder.close(reason=stop_reason)
            print(
                f"Teleop recording closed: {recorder.output_path}",
                flush=True,
            )


def run_ros2_cmd_vel_control(
    args: argparse.Namespace,
    *,
    room_path: Path,
    robot_path: Path,
    robot_position: tuple[float, float, float],
    robot_rotation: tuple[float, float, float, float],
    robot_yaw: float,
) -> None:
    """REV20 P1: `/cmd_vel` -> base drive, real `IsaacLab` `Articulation`
    control (NOT the `--no-keyboard-control` raw-stage ROS2 sensor path,
    which never constructs an `Articulation` and so has nothing to drive
    -- see `plans/REV20_TASKQUEUE.md`'s `/cmd_vel` scoping note). Deliberately
    a separate, minimal entry point rather than bolting a ROS2 subscriber
    onto `_run_keyboard_control_app` -- that function's dual-arm IK/
    teleop-recording machinery is unrelated and untested against this
    change; base-only driving here matches the P1 gate's own scope
    ("/cmd_vel -> TmrBaseAdapter.apply_twist", not arm control).
    """
    configure_ros2_bridge_env(args)
    require_single_teleop_environment(args.num_envs)
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app
    run_with_app_cleanup(
        simulation_app,
        lambda: _run_ros2_cmd_vel_control_app(
            args,
            simulation_app=simulation_app,
            room_path=room_path,
            robot_path=robot_path,
            robot_position=robot_position,
            robot_rotation=robot_rotation,
            robot_yaw=robot_yaw,
        ),
    )


def _run_ros2_cmd_vel_control_app(
    args: argparse.Namespace,
    *,
    simulation_app: Any,
    room_path: Path,
    robot_path: Path,
    robot_position: tuple[float, float, float],
    robot_rotation: tuple[float, float, float, float],
    robot_yaw: float,
) -> None:
    # NOTE: `make_headless_robot_usd` only. Do NOT also import
    # run_episode's `_fix_single_articulation_root` here: it is annotated
    # `-> None` and returns None on every path, so assigning its result to
    # `articulation_root_path` silently yielded `targetPrim=None` /
    # `chassisPrim=None` and Isaac then rejected both publisher graphs with
    # "The prim .../None is not valid. Please specify at least one valid
    # chassis prim" -- observed on GPU. The module-level
    # `fix_single_articulation_root` in THIS file is the P1-verified variant
    # that returns the kept prim's real resolved path, which is exactly why
    # its own docstring says it is kept here rather than imported.
    from run_episode import make_headless_robot_usd
    from tmr_base_control import (
        compensate_yaw_rate,
        compute_drive_targets,
        find_drive_joint_ids,
        get_root_yaw,
    )

    import isaaclab.sim as sim_utils
    from isaaclab.scene import InteractiveScene
    from isaaclab.sim import SimulationContext

    sim_cfg = sim_utils.SimulationCfg(
        dt=0.005, device=args.device, gravity=(0.0, 0.0, -9.81),
        physx=sim_utils.PhysxCfg(
            # Kit's own boot warning names this exact fix for exactly this
            # symptom ("if experiencing noisy velocities, consider enabling
            # this flag... together with more velocity iterations"). GPU-
            # measured 2026-08-13: with the defaults (False, 0 min velocity
            # iterations) the base's caster/steering DOFs ring at rest with
            # ZERO commands -- wheel_tau spiking to -303/-270 N.m against a
            # zero target, base_vel_b spiking to 0.5 m/s, chassis wandering
            # >0.5m and yawing >60deg in a few seconds -- even after the
            # placeholder-inertia fix above (that fix is real and reduced
            # peak torque from -500 to -300, but did not eliminate the
            # ringing, so it was solver conditioning too, not only mass
            # ratio).
            enable_external_forces_every_iteration=True,
            min_velocity_iteration_count=32,
            min_position_iteration_count=16,
            solve_articulation_contact_last=True,
        ),
    )
    sim = SimulationContext(sim_cfg)
    configure_keyboard_control_stage(
        configure_robot_room_stage,
        simulation_app,
        sim.stage,
        room_path=room_path,
        task=args.task,
        head_placement=args.head_placement,
        robot_position=robot_position,
        robot_yaw=robot_yaw,
        dynamic_beans=args.dynamic_beans,
    )
    scene_cfg = make_control_scene_cfg(
        num_envs=args.num_envs,
        robot_path=make_headless_robot_usd(robot_path),
        robot_position=robot_position,
        robot_rotation=robot_rotation,
        gripper=args.gripper,
    )
    scene = InteractiveScene(scene_cfg)
    articulation_root_path = fix_single_articulation_root(
        sim.stage, "/World/envs/env_0/Robot", prefer=args.articulation_root
    )
    # THE third of the "three stacked base bugs" that this launch path never
    # actually got. `clear_base_joint_friction` was only ever called on the
    # build_stage/--no-keyboard-control path -- the path that CANNOT drive the
    # base -- so every --cmd-vel-control run so far has been driving against
    # the asset's authored physxJoint:jointFriction (drive joints 1.0,
    # steering joints 5.0). plans/TOOL_FRAME_ROOT_CAUSE_2026-08-12.md recorded
    # this as "fixed and defaulted on on both launch paths"; the code said
    # otherwise, and the code wins.
    #
    # USD-time write, before sim.reset() builds the physics views: IsaacLab
    # seeds each actuator's `friction` from the value it READS BACK out of
    # PhysX at articulation init (Articulation._process_actuators_cfg passes
    # `friction=self._data.default_joint_friction_coeff[:, joint_ids]` and then
    # writes it straight back), so an authored value round-trips into the live
    # tensor unless it is cleared first. The live-tensor read-back below is
    # what actually confirms it landed.
    clear_base_joint_friction(sim.stage, "/World/envs/env_0/Robot")
    # ...and the other half of the same bug: with the friction gone the base
    # finally settles completely, and a settled articulation goes to SLEEP.
    # Joint drive targets do not wake it, so /cmd_vel is ignored forever after
    # the first idle period. Both fixes are required; either alone measures
    # worse than neither. See `disable_articulation_sleeping`.
    disable_articulation_sleeping(sim.stage, articulation_root_path)
    # ...and the third: the caster links carry the URDF exporter's
    # placeholder inertial (1 g wheel, inertia 1e-4) under a 147 kg
    # `base_link`, which is what the passive base DOFs have actually been
    # ringing on. USD-time write, same as the two above, because PhysX reads
    # inertials when it builds the articulation.
    derive_placeholder_link_inertials(sim.stage, "/World/envs/env_0/Robot")

    # MOVED ABOVE `sim.reset()` 2026-08-13. These two used to run AFTER it,
    # i.e. after PhysX had already parsed the stage and built its colliders.
    # `add_room_ground_collider` survived that (adding a prim raises a USD
    # notice PhysX acts on, and the measured base-z drop proves the slab
    # landed), but `apply_floor_friction` only rebinds a MATERIAL onto
    # colliders PhysX had already created with the default material -- a
    # change with no creation notice behind it. That is the same
    # ordering-and-never-read-back defect as `clear_base_joint_friction`
    # never being called on this path and the ground collider being measured
    # off a prim with no collider: three separate "the fix was recorded as
    # applied but the code applied it somewhere it could not take effect"
    # bugs on one code path. Both are USD-time writes, so both belong before
    # the reset, exactly like the joint-friction and sleep-threshold writes
    # above.
    if getattr(args, "ground_collider", False):
        add_room_ground_collider(sim.stage, "/World/Environment/RobotRoom")

    if getattr(args, "floor_friction", 0.0) > 0.0:
        apply_floor_friction(
            sim.stage, "/World/Environment/RobotRoom", args.floor_friction
        )

    sim.reset()
    scene.reset()
    robot = scene["robot"]
    reset_robot_to_default_state(robot, scene.env_origins)
    scene.write_data_to_sim()

    steering_indices, drive_indices = find_drive_joint_ids(robot.joint_names)
    print(f"Active steering joints: {steering_indices}", flush=True)
    print(f"Active drive joints: {drive_indices}", flush=True)

    # Read the LIVE, tensor-backed joint friction back rather than trusting
    # the USD write above -- this repo has been burned twice by USD writes
    # that printed their own value back while PhysX kept the old one. Scoped
    # with `joint_ids`, NEVER a full-width write: a full-width property write
    # round-trips every other joint's values through the same call and has
    # silently broken this exact base drive before (see the gains gotcha in
    # plans/TOOL_FRAME_ROOT_CAUSE_2026-08-12.md).
    base_joint_ids = list(steering_indices) + list(drive_indices)
    friction_before = [
        round(float(v), 4)
        for v in robot.data.joint_friction_coeff[0, base_joint_ids]
    ]
    if any(v > 0.0 for v in friction_before):
        robot.write_joint_friction_coefficient_to_sim(
            0.0, joint_ids=base_joint_ids
        )
    friction_after = [
        round(float(v), 4)
        for v in robot.data.joint_friction_coeff[0, base_joint_ids]
    ]
    print(
        f"BASEFRICTION live steering+drive {friction_before} -> "
        f"{friction_after}",
        flush=True,
    )

    # One-time dump of the LIVE drive-authority terms. A wheel that will not
    # reach its velocity target is limited by exactly one of these -- damping
    # (torque demanded), effort limit (torque allowed), armature (apparent
    # inertia), or nothing at all, in which case the limit is at the contact
    # and not in the joint. Printing them once beats guessing which.
    def _live(attr, ids):
        tensor = getattr(robot.data, attr, None)
        if tensor is None:
            return None
        return [round(float(tensor[0][i]), 4) for i in ids]

    # The passive DOFs are in this dump too. They were the blind spot: the
    # actuator group used to overwrite the asset's authored ACCELERATION-type
    # caster/rocker drives with zeros, and nothing ever printed what actually
    # landed, so "passive" was assumed rather than observed.
    def _limits(ids):
        for attr in ("joint_pos_limits", "joint_limits",
                     "soft_joint_pos_limits", "default_joint_pos_limits"):
            tensor = getattr(robot.data, attr, None)
            if tensor is None:
                continue
            try:
                return attr, [
                    [round(float(tensor[0][i][0]), 3),
                     round(float(tensor[0][i][1]), 3)]
                    for i in ids
                ]
            except Exception:  # noqa: BLE001 -- shape varies by build
                continue
        return None

    _passive_indices = [
        i for i, n in enumerate(robot.joint_names)
        if "caster" in n or n == "rocker_arm_joint"
    ]
    for label, ids in (
        ("steering", steering_indices),
        ("drive", drive_indices),
        ("passive", _passive_indices),
    ):
        print(
            f"BASEGAINS {label} "
            f"names={[robot.joint_names[i] for i in ids]} "
            f"stiffness={_live('joint_stiffness', ids)} "
            f"damping={_live('joint_damping', ids)} "
            f"effort_limit={_live('joint_effort_limits', ids)} "
            f"armature={_live('joint_armature', ids)} "
            # A DRIVE WHEEL MUST BE CONTINUOUS. If these come back finite,
            # the wheel can only rotate through that range and then stops
            # against a hard constraint -- which is a resisting torque with
            # no slip, at any effort limit, and no gain touches it. Every
            # position-mode measurement in this repo's history has the
            # shape of a joint hitting a limit: 6.0 rad commanded -> 0.78
            # achieved, 137 rad -> 0.19, 68.28 rad -> 0.048.
            f"pos_limits={_limits(ids)}",
            flush=True,
        )
    print(
        f"BASEMASS total={float(robot.data.default_mass[0].sum()):.2f} kg "
        f"over {robot.data.default_mass.shape[1]} bodies",
        flush=True,
    )

    # The single boolean that settles fixed-vs-floating base, instead of
    # inferring it from two helpers' docstrings. A fixed-base articulation
    # cannot translate no matter what the wheels do, and it is the one
    # explanation that predicts "commands clean, torque small, position rock
    # stable" without needing any traction argument at all.
    # One-time body inventory with resting heights. Without this we have been
    # guessing which link is which from name patterns, and that is how the
    # drive wheels stayed invisible in the ride-height readout.
    _rest_z = {
        n: round(float(robot.data.body_link_pos_w[0][i][2]), 4)
        for i, n in enumerate(robot.body_names)
        if float(robot.data.body_link_pos_w[0][i][2]) < 0.30
    }
    print(f"BODYNAMES n={len(robot.body_names)} all={list(robot.body_names)}",
          flush=True)
    print(f"BODYREST below_0.30m={_rest_z}", flush=True)

    print(
        f"ROOTKIND is_fixed_base={getattr(robot, 'is_fixed_base', 'UNKNOWN')} "
        f"root_prim={articulation_root_path} "
        f"root_body={robot.body_names[0] if robot.body_names else None}",
        flush=True,
    )

    apply_drive_wheel_authority(
        robot,
        drive_indices,
        floor_friction=float(getattr(args, "floor_friction", 0.0) or 0.0),
        damping=args.drive_damping,
        effort_limit=args.drive_effort_limit,
        armature=args.drive_armature,
        sim_dt=float(sim.cfg.dt),
    )
    free_caster_roll_joints(robot, damping=args.caster_roll_damping)
    apply_steering_authority(
        robot,
        steering_indices,
        stiffness=args.steering_stiffness,
        damping=args.steering_damping,
        effort_limit=args.steering_effort_limit,
    )
    # The rocker carried the same degree-unit gain bug as the casters
    # (see free_caster_roll_joints' docstring): raw asset stiffness
    # 35809.86 = 625 * 180/pi welds the suspension rocker rigid.
    # GPU-confirmed 2026-08-13 (outputs/codex_20260813/rocker_fix_test1.log,
    # rocker_fix_test2.log) as the mechanism behind the previously-open
    # "wheel back-driven through the articulation, not the ground" stall
    # in BASE_DRIVE_ROOT_CAUSE_2026-08-13.md -- corrected (625/0.003
    # defaults below), rocker reaction torque stayed single-digit and
    # slip_ratio ran ~0.02-0.24 instead of ~0.7-0.9 on the same repro,
    # with sustained real base travel where the raw value stalled. Now
    # applied by default.
    _rocker_ids = [
        i for i, n in enumerate(robot.joint_names) if n == "rocker_arm_joint"
    ]
    if _rocker_ids:
        apply_steering_authority(
            robot,
            _rocker_ids,
            stiffness=args.rocker_stiffness,
            damping=args.rocker_damping,
            label="ROCKERAUTHORITY",
        )

    # enable_ros2_bridge/publish_ros2_clock moved HERE, after the stage and
    # scene are fully populated -- calling them right after AppLauncher
    # (this path's original order, mirrored from _run_keyboard_control_app)
    # crashes `omni.graph.core` every time: "Unable to create prim for graph
    # at /ROS2_ClockGraph" / "Failed to wrap graph in node given {...}".
    # GPU-verified, reproduced 4x against a bare/near-empty stage (including
    # with an explicit /World default prim defined -- not just "no prim
    # exists yet"), and confirmed absent on the build_stage/
    # --no-keyboard-control path, which loads the full room+robot content
    # BEFORE calling enable_ros2_bridge/publish_ros2_clock. Root cause not
    # fully isolated beyond "OmniGraph's node creation needs a stage with
    # real content, not just a defined prim, before it will create a graph
    # node" -- not chased further since the reorder itself is cheap and safe,
    # and matches what the one launch path that has always worked does.
    #
    # STALE CLAIM, RE-OPENED 2026-08-12: the reorder above is real (stage
    # loaded, "Robot room loaded in Isaac Sim" and "Active steering/drive
    # joints" both printed first, confirmed in the log) but this exact call
    # still hit the identical OmniGraphError, reproduced 2/2 on this
    # environment. Testing a narrower hypothesis: `build_stage()` (the
    # working --no-keyboard-control path) renders real frames as part of its
    # own composition; `InteractiveScene`/`sim.reset()`/`scene.reset()` here
    # may populate physics tensors without ever driving an actual render
    # tick, and OmniGraph's node creation may need at least one. Untested
    # before now -- add the same settle pattern `generate_occupancy_map.py`
    # already uses for an analogous "PhysX needs real frames before it will
    # act" case, rather than guessing another reorder.
    for _ in range(20):
        simulation_app.update()
    enable_ros2_bridge(simulation_app, args)

    # /clock comes from rclpy on this path, NOT from the OmniGraph node.
    # `publish_ros2_clock` fails here with "Failed to wrap graph in node given
    # {'graph_path': '/ROS2_ClockGraph'}" and takes the whole app down with
    # it -- before any base code runs, which is what BLOCKER 5 has actually
    # been all along. Reordering it after the stage is populated and adding 20
    # render ticks were both tried and did not help.
    #
    # This is the same move already made for /joint_states a few lines below,
    # and for the same reason: in this app mode an IsaacLab SimulationContext
    # owns the stage and the physics tensors, and the OmniGraph ROS nodes do
    # not survive alongside it. The rclpy node this path already creates can
    # publish the topic directly, from `sim.current_time`, which is the true
    # simulation clock.
    #
    # Every remaining OmniGraph publisher is wrapped so that one failing does
    # not kill the run -- and so the log says exactly which ones work here.
    def _try_graph(label: str, fn) -> bool:
        try:
            fn()
            print(f"OmniGraph {label}: OK", flush=True)
            return True
        except Exception as exc:  # noqa: BLE001 -- report, never mask
            print(f"OmniGraph {label}: FAILED {type(exc).__name__}: {exc}",
                  flush=True)
            return False

    # NOT publish_ros2_joint_states (the OmniGraph ROS2PublishJointState
    # node) on this path. That node and the InteractiveScene/Articulation
    # constructed above both try to own the same physics tensor view; the
    # Articulation wins the GPU pipeline and the OmniGraph node is left
    # reading a CPU-side view of a GPU-resident tensor, which PhysX rejects
    # every tick: "Incompatible device of DOF position tensor ... expected
    # device 0, received device -1". Confirmed NOT Robotiq-specific (A/B'd
    # against the default robot too) -- it is a two-owners bug, not an asset
    # bug. `--no-keyboard-control` has no Articulation at all, which is why
    # ROS2PublishJointState works fine there. Here, publish straight from
    # the Articulation's own GPU-resident `robot.data` instead, in the same
    # rclpy node already used for /cmd_vel, matching robot_state_publisher's
    # expected /joint_states shape (name/position/velocity, no effort).
    if not args.no_extra_graphs:
        _try_graph("odometry",
                   lambda: publish_ros2_odometry(
                       args, chassis_prim_path=articulation_root_path))
    # Cameras (and the depth_pcl cloud that becomes /scan) are needed HERE, not
    # only on the --no-keyboard-control path, because GATE P2 requires both at
    # once: Nav2's costmaps consume /scan while Nav2 drives the base over
    # /cmd_vel. Without this the two requirements were mutually exclusive --
    # the sensor path could not be driven and the drivable path was blind.
    #
    # The asset root differs between the two paths and must not be copied from
    # the other one: InteractiveScene nests the robot at
    # /World/envs/env_0/Robot, whereas the --no-keyboard-control path
    # references it at /World/Robot/Asset. Cameras are looked up relative to
    # this root and are SKIPPED with a printed warning (not a crash) if absent,
    # so this stays safe on the compat robot, which genuinely lacks them.
    if not args.no_extra_graphs:
        _try_graph("cameras",
                   lambda: publish_ros2_cameras(
                       args, asset_root_path="/World/envs/env_0/Robot"))
    else:
        print("OmniGraph odometry+cameras SKIPPED (--no-extra-graphs)",
              flush=True)

    import rclpy
    from geometry_msgs.msg import Twist
    from rclpy.parameter import Parameter
    from sensor_msgs.msg import JointState

    rclpy.init(args=[])
    node = rclpy.create_node("cmd_vel_base_control")
    # Every other node in this stack (Nav2, robot_state_publisher, move_group)
    # runs use_sim_time: true against Isaac's /clock -- this node's own
    # timestamps must use the same clock or its /joint_states will drift
    # against robot_state_publisher's TF, which consumes it.
    node.set_parameters(
        [Parameter("use_sim_time", Parameter.Type.BOOL, True)]
    )
    latest_twist = {"vx": 0.0, "vy": 0.0, "wz": 0.0, "stamp": 0.0}

    def _on_cmd_vel(msg: Twist) -> None:
        latest_twist["vx"] = float(msg.linear.x)
        latest_twist["vy"] = float(msg.linear.y)
        latest_twist["wz"] = float(msg.angular.z)
        latest_twist["stamp"] = time.monotonic()

    node.create_subscription(Twist, "/cmd_vel", _on_cmd_vel, 10)
    print("Subscribed to /cmd_vel, driving the base only.", flush=True)

    # This launch path has NO /joint_command consumer of any kind for the
    # arm/gripper -- confirmed 2026-08-13, `ros2 topic info /joint_command
    # --verbose` shows zero subscriptions here, and a Jacobian measured by
    # perturbing each arm joint read exactly 0.0000 m/rad on every column.
    # scripts/task3/grasp_and_transport.py (and everything built on
    # calibrate_tool_frame.py's Calibrator) publishes arm targets on
    # /joint_command and has no other way to move the arm, so without this
    # every grasp attempt on this launch path silently commands an arm that
    # never moves. `build_stage`'s OmniGraph ArticulationController does
    # consume /joint_command, but its base drive is unreliable (asymmetric
    # wheel response, ~4% of a requested move) in a way this path's base
    # control is not. Reusing the SAME direct-API technique already proven
    # for the base here (no OmniGraph involved) gives both: base via
    # /cmd_vel, arm/gripper via /joint_command, one process, one launch mode.
    _joint_name_to_id = {n: i for i, n in enumerate(robot.joint_names)}
    _latest_joint_cmd: dict[str, float] = {}
    _latest_joint_vel_cmd: dict[str, float] = {}

    def _on_joint_command(msg: JointState) -> None:
        # BUG (2026-08-13): only msg.position was ever read here, so any
        # velocity-type /joint_command message (drive_base_joint_command.py's
        # wheel commands -- the drive wheels are authored stiffness=0.0 /
        # damping=100000.0, a pure velocity joint, so a position target on
        # them is a total no-op) was silently dropped. GPU-measured: a
        # requested 1.3 m straight drive moved the base 0.0157 m -- pure
        # settling noise, not real driving, for a script whose velocity
        # commands were simply never applied. Mirrors the base/steering
        # split drive_base_joint_command.py's own docstring already
        # documents for a different launch path; this path needed the same
        # fix independently since it has its own, separate subscriber.
        for n, p in zip(msg.name, msg.position):
            if n in _joint_name_to_id:
                _latest_joint_cmd[n] = float(p)
        for n, v in zip(msg.name, msg.velocity):
            if n in _joint_name_to_id:
                _latest_joint_vel_cmd[n] = float(v)
                _latest_joint_cmd.pop(n, None)

    node.create_subscription(JointState, "/joint_command", _on_joint_command, 10)
    print(f"Subscribed to /joint_command (direct API, not OmniGraph) for "
          f"{len(_joint_name_to_id)} known joints.", flush=True)

    from rosgraph_msgs.msg import Clock as ClockMsg

    # TF from rclpy, not OmniGraph. This mode is the ONLY one that can drive
    # the base -- the OmniGraph IsaacArticulationController delivers position
    # commands to the arm but moves the wheels for neither position nor
    # velocity (a 137 rad position ramp turned them 0.19 rad and the chassis
    # 0.0025 m) -- while its ROS2PublishTransformTree graphs all fail to
    # create here. So the grasp stack's inputs are published directly from the
    # articulation and the stage instead, exactly as /joint_states already is.
    #
    # Everything is emitted in ODOM, computed from the known spawn pose, so
    # this path does not depend on Nav2's map->odom static transform at all
    # and cannot silently disagree with it.
    from geometry_msgs.msg import TransformStamped
    from tf2_ros import TransformBroadcaster

    tf_broadcaster = TransformBroadcaster(node)
    _spawn_x, _spawn_y, _ = robot_position
    # robot_yaw is DEGREES (see yaw_to_quat, which converts it).
    _yaw_rad = math.radians(robot_yaw)
    _cos_y = math.cos(-_yaw_rad)
    _sin_y = math.sin(-_yaw_rad)
    # Rotation counterpart of _world_to_odom's position rotation. Without
    # this, an orientation passed straight through (as `robot.data.
    # root_quat_w` is below) is WORLD-frame while everything else broadcast
    # here is odom-frame -- position correct, rotation silently wrong by the
    # spawn yaw. Found 2026-08-13 driving the base on a heading controller
    # that reads this rotation: see plans/TOOL_FRAME_ROOT_CAUSE_2026-08-12.md.
    _spawn_quat_inv = yaw_to_quat(-robot_yaw)

    def _world_quat_to_odom(quat_wxyz):
        return multiply_quats(_spawn_quat_inv, tuple(float(v) for v in quat_wxyz))

    from pxr import UsdGeom as _UsdGeom

    # NOT robot.body_names.index(name): both `left_Robotiq_2F_85` and
    # `right_Robotiq_2F_85` contain a `left_inner_finger` AND a
    # `right_inner_finger` (Robotiq's own left/right, no arm prefix -- see
    # gripper_pad_prim_paths's docstring), so `.index()` on the flat body
    # list silently returns ONE arm's copy for both names regardless of
    # which arm that is. Found 2026-08-13: a `--side right` grasp servo's
    # Jacobian measurement showed pad_mid barely moving across an entire
    # shoulder-yaw sweep of the RIGHT arm -- because the published
    # "right_inner_finger" was the LEFT arm's stationary finger the whole
    # time. `gripper_pad_prim_paths` already does this correctly by prim
    # PATH, scoped to one side (`--publish-gripper-pad-tf <side>`); reusing
    # it here and reading via XformCache, the same technique already used
    # for objects below, sidesteps body-index ambiguity entirely.
    asset_root_path = "/World/envs/env_0/Robot"  # matches publish_ros2_cameras above
    _pad_side = getattr(args, "publish_gripper_pad_tf", "none")
    _pad_paths = gripper_pad_prim_paths(sim.stage, asset_root_path, _pad_side)
    # NOT XformCache for these -- found 2026-08-13, second bug in the same
    # spot. IsaacLab's Articulation/InteractiveScene takes over the GPU
    # tensor view for every body belonging to the robot, and PhysX's normal
    # write-back of that body's transform onto its USD prim never happens
    # while that ownership holds -- confirmed live: /joint_command drove
    # right_fr3v2_joint1 from 0.003 to -2.03 rad (matching /joint_states, the
    # tensor-backed reading, exactly), while the XformCache-read pad_mid
    # stayed bit-identical to five decimals across the entire aim sweep. The
    # arm was genuinely moving; the pad TF was reading a frozen rest pose.
    # Objects (below) are NOT part of the robot Articulation, so nothing
    # hijacks their normal PhysX->USD write-back -- XformCache is trusted
    # for them, untested but structurally a different case.
    #
    # Fix: read pads from robot.data.body_pos_w/body_quat_w (tensor-backed,
    # always live), like base_link already does. That alone reintroduces the
    # ORIGINAL bug this file is named after (body_names.index() is ambiguous
    # between the two arms' identically-named pads) -- resolved here by a
    # one-time calibration instead: XformCache is still accurate at THIS
    # single instant, before any staleness has had a chance to matter, so
    # comparing each name-ambiguous candidate body's tensor position against
    # the correctly PATH-resolved prim's XformCache position right now picks
    # the right one, once, with no hardcoded index or ordering assumption.
    pad_body_ids: list[tuple[str, int]] = []
    if _pad_paths:
        _cal_cache = _UsdGeom.XformCache()
        for _p in _pad_paths:
            _prim = sim.stage.GetPrimAtPath(_p)
            _name = _prim.GetName()
            _ref_pos = _cal_cache.GetLocalToWorldTransform(_prim).ExtractTranslation()
            _candidates = [i for i, n in enumerate(robot.body_names) if n == _name]
            if not _candidates:
                print(f"WARNING: pad calibration: no body named {_name!r}",
                      flush=True)
                continue
            _best_i, _best_d = _candidates[0], float("inf")
            for _i in _candidates:
                _bp = robot.data.body_pos_w[0][_i]
                _d = ((float(_bp[0]) - _ref_pos[0]) ** 2
                      + (float(_bp[1]) - _ref_pos[1]) ** 2
                      + (float(_bp[2]) - _ref_pos[2]) ** 2)
                if _d < _best_d:
                    _best_i, _best_d = _i, _d
            print(f"Pad calibration: {_name!r} -> body index {_best_i} "
                  f"(of {_candidates}, xform-vs-tensor dist {_best_d ** 0.5:.5f} m)",
                  flush=True)
            pad_body_ids.append((_name, _best_i))
    print(f"TF: publishing {len(pad_body_ids)} finger pads via live tensor "
          f"state (side={_pad_side})", flush=True)

    object_prims = []
    if args.publish_object_tf:
        _bbox_stage = sim.stage
        for _name in args.publish_object_tf:
            for _prim in _bbox_stage.Traverse():
                if _prim.GetName() == _name:
                    object_prims.append((_name, _prim))
                    break
        print(f"TF: publishing {len(object_prims)} object prims: "
              f"{[n for n, _ in object_prims]}", flush=True)
        _xform_cache = _UsdGeom.XformCache()

    def _world_to_odom(x: float, y: float, z: float):
        dx, dy = x - _spawn_x, y - _spawn_y
        return (dx * _cos_y - dy * _sin_y, dx * _sin_y + dy * _cos_y, z)

    def _send_tf(child: str, xyz, quat_wxyz, stamp) -> None:
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id = "odom"
        t.child_frame_id = child
        t.transform.translation.x = float(xyz[0])
        t.transform.translation.y = float(xyz[1])
        t.transform.translation.z = float(xyz[2])
        t.transform.rotation.w = float(quat_wxyz[0])
        t.transform.rotation.x = float(quat_wxyz[1])
        t.transform.rotation.y = float(quat_wxyz[2])
        t.transform.rotation.z = float(quat_wxyz[3])
        tf_broadcaster.sendTransform(t)

    clock_pub = node.create_publisher(ClockMsg, "/clock", 10)
    print("Publishing /clock from sim.current_time (rclpy, not OmniGraph).",
          flush=True)

    joint_state_pub = node.create_publisher(JointState, "/joint_states", 10)
    joint_state_names = list(robot.joint_names)
    print(
        f"Publishing /joint_states from robot.data ({len(joint_state_names)} "
        "joints), not the OmniGraph node -- see the two-owners note above.",
        flush=True,
    )

    heading_hold_yaw = get_root_yaw(robot)
    timeline = None
    if args.autoplay:
        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()

    # A stale /cmd_vel (no message for CMD_VEL_TIMEOUT_S) stops the base --
    # same safety default every real ROS2 diff-drive bringup uses, so a
    # dropped publisher does not leave the robot drifting indefinitely.
    CMD_VEL_TIMEOUT_S = 0.5
    _dbg_n = [0]
    # BASEDBG cadence. The default 200 ticks is ONE FULL SIM SECOND at
    # sim_cfg.dt=0.005, not one tick -- a distinction that matters, because a
    # 2026-08-13 escalation read two consecutive BASEDBG lines as consecutive
    # physics steps and concluded a 0.824 m "single-tick teleport". Set
    # EBIM_BASEDBG_EVERY=1 to watch a transient build up instead of sampling
    # its endpoints.
    _dbg_every = max(1, int(os.environ.get("EBIM_BASEDBG_EVERY", "200")))
    _base_owned_ids = set(steering_indices) | set(drive_indices)
    _drive_signs = None
    if args.drive_module_signs:
        import torch as _torch_signs

        _sign_values = [
            float(v) for v in str(args.drive_module_signs).split(",")
        ]
        if len(_sign_values) != len(drive_indices):
            raise SystemExit(
                f"--drive-module-signs needs {len(drive_indices)} "
                f"comma-separated values, got {len(_sign_values)}"
            )
        _drive_signs = _torch_signs.tensor(
            [_sign_values], device=sim.device
        )
        print(f"DRIVESIGNS applying per-module {_sign_values}", flush=True)
    try:
        while simulation_app.is_running():
            rclpy.spin_once(node, timeout_sec=0.0)
            stale = (
                time.monotonic() - latest_twist["stamp"] > CMD_VEL_TIMEOUT_S
            )
            vx = 0.0 if stale else latest_twist["vx"]
            vy = 0.0 if stale else latest_twist["vy"]
            wz_cmd = 0.0 if stale else latest_twist["wz"]

            wz, heading_hold_yaw = compensate_yaw_rate(
                robot,
                vx,
                vy,
                wz_cmd,
                heading_hold_yaw,
                # `compensate_yaw_rate` (scripts/common/tmr_base_control.py)
                # injects HEADING_HOLD_KP * yaw_error - HEADING_HOLD_KD *
                # yaw_rate, clamped to +/-0.8 rad/s, whenever the caller is not
                # commanding rotation itself. With wz_cmd = 0 that is ALWAYS
                # on, so a "pure vx, wz=0" test is not one: measured
                # 2026-08-13, a raw wz=0.0 command produced a growing
                # -0.061 -> -0.106 wz with steer targets drifting away from
                # zero in step with it. --no-heading-hold forces
                # manual_rotation=True, making compensate_yaw_rate a
                # pass-through so wz really is what was published.
                manual_rotation=(
                    abs(wz_cmd) > 1.0e-4 or args.no_heading_hold
                ),
            )
            steering_pos_targets, drive_vel_targets = compute_drive_targets(
                robot,
                steering_indices,
                vx,
                vy,
                wz,
                num_envs=args.num_envs,
                device=sim.device,
            )
            robot.set_joint_position_target(
                steering_pos_targets, joint_ids=steering_indices
            )
            # PER-MODULE DRIVE SIGN. `compute_drive_targets` derives each
            # wheel's sign from atan2 plus a direct-vs-flipped choice, which
            # is correct ONLY if a positive wheel velocity means "forward
            # along that module's steering direction" on BOTH modules. On a
            # diagonal 2-module base the modules are usually mirror images,
            # and a mirrored module's steering zero can point the opposite way
            # -- in which case both wheels commanded positive drive AGAINST
            # each other with the mixer none the wiser.
            #
            # Measured 2026-08-13: with steer_pos = [0.001, -0.006] (both
            # modules pointing along body +x) and wheel_tgt = [5.004, 4.996]
            # (both commanded forward), wheel_vel came back [5.0, -0.4] with
            # wheel_tau [2.0, 250.0]. The front wheel tracked its target
            # perfectly at near-zero torque while the rear sat NEGATIVE with
            # torque pinned at the cap. One wheel back-driven by the other is
            # exactly what opposed modules produce, and it is immune to
            # damping, armature, effort limit, casters, rocker, heading hold
            # and steering slew -- which is every knob that was tried.
            if _drive_signs is not None:
                drive_vel_targets = drive_vel_targets * _drive_signs
            robot.set_joint_velocity_target(
                drive_vel_targets, joint_ids=drive_indices
            )
            # ARBITRATION: while a /cmd_vel is FRESH, the mixer above owns the
            # four base joints and /joint_command may not write them. Before
            # this, `_base_owned_ids` was computed and then used only in a
            # debug print, so both channels wrote the same wheels in the same
            # tick, from two publishers at two different rates, last write
            # winning at random. That became reachable only today, when the
            # /joint_command handler was fixed to stop dropping velocity
            # commands -- i.e. the moment `drive_base_joint_command.py` and
            # `drive_base_cmd_vel.py` could both actually reach the wheels.
            # Two asynchronous velocity targets on one servo is a sign-flip
            # oscillator by construction.
            #
            # Scoped to the FRESH case on purpose: with /cmd_vel idle or
            # stopped, /joint_command keeps full base access, so
            # drive_base_joint_command.py still works exactly as before.
            _jc_blocked = _base_owned_ids if not stale else frozenset()
            if _latest_joint_cmd:
                import torch as _torch2

                _jc_names = [
                    n for n in _latest_joint_cmd
                    if _joint_name_to_id[n] not in _jc_blocked
                ]
                if _jc_names:
                    _jc_ids = [_joint_name_to_id[n] for n in _jc_names]
                    _jc_pos = _torch2.tensor(
                        [[_latest_joint_cmd[n] for n in _jc_names]],
                        device=sim.device,
                    )
                    robot.set_joint_position_target(
                        _jc_pos, joint_ids=_jc_ids
                    )
            if _latest_joint_vel_cmd:
                import torch as _torch3

                # Same arbitration, and this is the half that matters: the
                # drive wheels are stiffness=0 velocity joints, so it is the
                # VELOCITY channel that can fight the mixer for them.
                _jv_names = [
                    n for n in _latest_joint_vel_cmd
                    if _joint_name_to_id[n] not in _jc_blocked
                ]
                if _jv_names:
                    _jv_ids = [_joint_name_to_id[n] for n in _jv_names]
                    _jv_vel = _torch3.tensor(
                        [[_latest_joint_vel_cmd[n] for n in _jv_names]],
                        device=sim.device,
                    )
                    robot.set_joint_velocity_target(
                        _jv_vel, joint_ids=_jv_ids
                    )
            # REMOVED 2026-08-13: a per-tick `write_root_velocity_to_sim(
            # root_com_vel_w + 1e-3 about world Z)` "wake nudge". It was never
            # verified to do anything, and reading IsaacLab's own source shows
            # it cannot do what it claimed while doing something it did not
            # intend:
            #   * `Articulation.write_root_velocity_to_sim` just forwards to
            #     `write_root_com_velocity_to_sim`, and `data.root_com_vel_w`
            #     is `root_physx_view.get_root_velocities()` read straight
            #     back, so read-then-write is an IDENTITY write. The only
            #     thing it actually changed was the +1e-3.
            #   * That +1e-3 was added to the CURRENT velocity every physics
            #     tick, so it is not a one-off nudge -- it is a sustained
            #     0.2 rad/s^2 (1e-3 / dt) yaw acceleration injected for as
            #     long as any /cmd_vel is active, applied AFTER the solver so
            #     nothing damps it inside the tick. An unmodelled yaw torque
            #     fighting the heading controller is the last thing this base
            #     needed, and it is a live suspect for the 63.9 deg roll
            #     tip-over recorded the same day.
            #   * Sleeping is already prevented at the source by
            #     `disable_articulation_sleeping` (sleepThreshold = 0), and
            #     the frozen-base signature this was written for turned out to
            #     be the passive-caster instability instead -- see
            #     plans/TOOL_FRAME_ROOT_CAUSE_2026-08-12.md.
            # `probe_base_drive.py` drives this same base at 0.495 m/s with the
            # wheels tracking 9.9 of a 10.0 rad/s target, so the asset and the
            # physics are fine and the difference is in this path's setup.
            # Print commanded vs achieved once a second to say which half is
            # wrong here: a zero TARGET means the twist never arrived; a
            # nonzero target with zero wheel velocity means the physics.
            _dbg_n[0] += 1
            if _dbg_n[0] % _dbg_every == 0:
                print(
                    f"BASEDBG twist=({vx:.3f},{vy:.3f},{wz:.3f}) stale={stale} "
                    f"wheel_tgt={[round(float(v), 3) for v in drive_vel_targets[0]]} "
                    f"wheel_vel={[round(float(robot.data.joint_vel[0][i]), 3) for i in drive_indices]} "
                    # Steering commanded-vs-actual: with the base joints'
                    # authored friction live, these two diverge and stay
                    # diverged (the modules stick), the wheels then scrub
                    # sideways, and the chassis crawls while the wheels turn.
                    # That divergence is the signature to watch for.
                    f"steer_tgt={[round(float(v), 3) for v in steering_pos_targets[0]]} "
                    f"steer_pos={[round(float(robot.data.joint_pos[0][i]), 3) for i in steering_indices]} "
                    # Torque actually delivered to the wheels vs what the
                    # actuator asked for: if applied is pinned at the effort
                    # limit while the wheel still lags its target, the limit
                    # is in the joint; if applied is small, the limit is at
                    # the tyre/ground contact instead.
                    f"wheel_tau={[round(float(robot.data.applied_torque[0][i]), 1) for i in drive_indices]} "
                    f"base_vel_b={[round(float(v), 3) for v in robot.data.root_lin_vel_b[0][:2]]} "
                    f"base_xy={[round(float(v), 4) for v in robot.data.root_pos_w[0][:2]]} "
                    # Which base joints a /joint_command publisher has taken
                    # over. The /joint_command block below re-applies its
                    # stored targets AFTER the /cmd_vel block above, so any
                    # base joint listed here has its /cmd_vel target silently
                    # discarded every tick -- steer_tgt above is then what was
                    # COMPUTED, not what was applied.
                    f"jc_base={sorted(n for n in _latest_joint_cmd if _joint_name_to_id.get(n) in _base_owned_ids)}",
                    flush=True,
                )
                # BASESLIP: the one measurement that separates the two
                # remaining explanations for "commands are clean, torque is
                # small, base does not move".
                #
                #   slip ~ 1  -> the wheels are turning and the floor is not
                #               holding them. Traction/normal-load problem.
                #   slip ~ 0 with wheel_vel ~ 0 -> the wheels are not turning
                #               at all. Torque/authority problem.
                #
                # Everything printed above is compatible with BOTH, which is
                # why five rounds of gain changes could not distinguish them.
                # `rim` is what the wheel would travel at with no slip;
                # `base_speed` is what the chassis actually does.
                #
                # `wheel_z` and `caster_z` are here because the leading
                # explanation for a traction failure on THIS robot is load
                # distribution, not friction: `rocker_arm_joint` is live at
                # stiffness 35809.86 (625 authored in degree units), i.e. a
                # suspension rocker welded rigid. A rigid rocker over four
                # ground contacts is over-constrained, and it can hold the
                # drive wheels light while the casters carry the robot. Wheels
                # that carry no load cannot pull, at any effort limit, and no
                # gain fixes it. Comparing the drive-wheel body heights
                # against the caster body heights says whether that is what is
                # happening.
                def _rocker_state():
                    ids = [
                        i
                        for i, n in enumerate(robot.joint_names)
                        if n == "rocker_arm_joint"
                    ]
                    if not ids:
                        return None
                    i = ids[0]
                    pos = float(robot.data.joint_pos[0][i])
                    limits = getattr(robot.data, "joint_pos_limits", None)
                    lo = hi = None
                    if limits is not None:
                        try:
                            lo = float(limits[0][i][0])
                            hi = float(limits[0][i][1])
                        except Exception:  # noqa: BLE001 -- shape varies
                            lo = hi = None
                    at_limit = None
                    if lo is not None and hi is not None:
                        margin = min(abs(pos - lo), abs(hi - pos))
                        at_limit = margin < 0.01
                    return {
                        "pos": round(pos, 4),
                        "limits": [round(lo, 3), round(hi, 3)]
                        if lo is not None
                        else None,
                        "at_limit": at_limit,
                        "tau": round(
                            float(robot.data.applied_torque[0][i]), 1
                        ),
                    }

                def _live_effort_limit() -> list[float] | None:
                    tensor = getattr(
                        robot.data, "joint_effort_limits", None
                    )
                    if tensor is None:
                        return None
                    return [
                        round(float(tensor[0][i]), 2) for i in drive_indices
                    ]

                def _effort_cap_hit() -> list[bool] | None:
                    limits = _live_effort_limit()
                    if limits is None:
                        return None
                    return [
                        abs(float(robot.data.applied_torque[0][i]))
                        > 0.95 * limit
                        for i, limit in zip(drive_indices, limits)
                    ]

                # The demand ceiling. An implicit velocity servo can only ask
                # for `damping * (target - actual)`, so raising the effort
                # limit above that ceiling changes NOTHING -- at damping 20
                # and a 5 rad/s target the drive can never demand more than
                # ~100 N.m, which makes a 250 N.m cap untestable by
                # construction. Printing it stops us sweeping a limit that is
                # not the binding constraint.
                _demand_ceiling = [
                    round(
                        float(
                            robot.data.joint_damping[0][i]
                            * abs(
                                float(drive_vel_targets[0][k])
                                - float(robot.data.joint_vel[0][i])
                            )
                        ),
                        1,
                    )
                    for k, i in enumerate(drive_indices)
                ]
                _slip_bodies = getattr(robot.data, "body_link_pos_w", None)
                _rim = [
                    round(float(robot.data.joint_vel[0][i]) * 0.05, 4)
                    for i in drive_indices
                ]
                _base_speed = float(
                    (
                        robot.data.root_lin_vel_b[0][0] ** 2
                        + robot.data.root_lin_vel_b[0][1] ** 2
                    )
                    ** 0.5
                )
                _fastest_rim = max((abs(v) for v in _rim), default=0.0)
                _slip = (
                    round((_fastest_rim - _base_speed) / _fastest_rim, 3)
                    if _fastest_rim > 1e-4
                    else None
                )
                _z = {}
                if _slip_bodies is not None:
                    # base_link is in here deliberately. If the chassis is
                    # resting ON THE FLOOR rather than on its wheels, the
                    # drag is mu * the robot's whole weight instead of a
                    # rolling term, and NO effort limit inside the friction
                    # cone can overcome it. That is the one mechanism that
                    # predicts torque pinned at the cap with the wheels
                    # barely turning AND slip near zero.
                    # NAME-BASED FILTERING MISSED THE DRIVE WHEELS. The
                    # 2026-08-13 body_z dump contained base, base_link, the
                    # rocker and every caster link -- and no drive wheel at
                    # all, because those links are not named "wheel". The one
                    # ride height that decides whether the drive wheels are
                    # even carrying load was the one absent from the readout.
                    # Height-based selection cannot miss a body the way a name
                    # pattern can: everything within 0.25 m of the floor is
                    # part of the ground-contact problem, whatever it is
                    # called.
                    for _bi, _bn in enumerate(robot.body_names):
                        _bz = float(_slip_bodies[0][_bi][2])
                        if _bz < 0.25:
                            _z[_bn] = round(_bz, 4)
                print(
                    f"BASESLIP rim={_rim} base_speed={round(_base_speed, 4)} "
                    f"demand_ceiling={_demand_ceiling} "
                    # The rocker's LIVE angle against its measured limits of
                    # [-0.16, 0.18]. The 2026-08-13 asymmetric stall -- front
                    # wheel tracking at single-digit torque while the rear sat
                    # at zero velocity with torque pinned -- needs a
                    # per-module explanation, and the rocker is the one
                    # suspension element that does not act on both modules
                    # equally. A rocker parked against a hard limit reacts its
                    # module's drive torque into a rigid constraint, which
                    # blocks that wheel and only that wheel.
                    f"rocker={_rocker_state()} "
                    f"slip_ratio={_slip} "
                    # BUG, 2026-08-13: this compared against a HARDCODED 36.7
                    # instead of the joint's live effort limit, so once
                    # --drive-effort-limit was swept to 60 and 250 the flag
                    # only ever meant "torque exceeded 34.9 N.m" -- which is
                    # unremarkable at those caps -- while reading as
                    # "saturated". Two sweep points were interpreted through
                    # it. Compare against the live limit, and print the
                    # torque and the limit alongside so the flag can never
                    # again be the only thing we look at.
                    f"tau={[round(float(robot.data.applied_torque[0][i]), 1) for i in drive_indices]} "
                    f"tau_limit={_live_effort_limit()} "
                    f"effort_cap_hit={_effort_cap_hit()} "
                    f"body_z={_z}",
                    flush=True,
                )
            disable_robot_external_wrenches(robot)
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.cfg.dt)

            clock_msg = ClockMsg()
            _t = float(sim.current_time)
            clock_msg.clock.sec = int(_t)
            clock_msg.clock.nanosec = int((_t - int(_t)) * 1e9)
            clock_pub.publish(clock_msg)

            joint_state_msg = JointState()
            joint_state_msg.header.stamp = clock_msg.clock
            joint_state_msg.name = joint_state_names
            joint_state_msg.position = (
                robot.data.joint_pos[0].detach().cpu().tolist()
            )
            joint_state_msg.velocity = (
                robot.data.joint_vel[0].detach().cpu().tolist()
            )
            joint_state_pub.publish(joint_state_msg)

            # ~20 Hz is plenty for the pad servo and keeps the XformCache
            # recomputation off the physics step.
            if _dbg_n[0] % 10 == 0:
                _stamp = clock_msg.clock
                _root = robot.data.root_pos_w[0]
                _rq = robot.data.root_quat_w[0]
                _send_tf("base_link",
                         _world_to_odom(float(_root[0]), float(_root[1]),
                                        float(_root[2])),
                         _world_quat_to_odom(_rq), _stamp)
                for _pname, _bid in pad_body_ids:
                    _p = robot.data.body_pos_w[0][_bid]
                    _q = robot.data.body_quat_w[0][_bid]
                    _send_tf(_pname,
                             _world_to_odom(float(_p[0]), float(_p[1]),
                                            float(_p[2])),
                             _world_quat_to_odom(_q), _stamp)
                if object_prims:
                    _xform_cache.Clear()
                    for _oname, _oprim in object_prims:
                        _m = _xform_cache.GetLocalToWorldTransform(_oprim)
                        _tr = _m.ExtractTranslation()
                        _q = _m.ExtractRotationQuat()
                        _i = _q.GetImaginary()
                        _send_tf(_oname,
                                 _world_to_odom(_tr[0], _tr[1], _tr[2]),
                                 _world_quat_to_odom(
                                     [_q.GetReal(), _i[0], _i[1], _i[2]]),
                                 _stamp)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main() -> None:
    args = parse_args()
    room_path = resolve_usd_path(
        args.room_usd,
        asset_path("robot_room.usd"),
    )
    robot_path = resolve_usd_path(
        args.robot_usd,
        asset_path("mobile_fr3_duo_v0_2.usd"),
    )

    if not room_path.is_file():
        raise FileNotFoundError(f"Room USD not found: {room_path}")
    if not robot_path.is_file():
        raise FileNotFoundError(f"Robot USD not found: {robot_path}")

    robot_position = resolve_robot_position(args)
    robot_yaw = resolve_robot_yaw(args)
    robot_rotation = yaw_to_quat(robot_yaw)

    if args.cmd_vel_control:
        run_ros2_cmd_vel_control(
            args,
            room_path=room_path,
            robot_path=robot_path,
            robot_position=robot_position,
            robot_rotation=robot_rotation,
            robot_yaw=robot_yaw,
        )
        return

    if should_enable_keyboard_control(args):
        run_keyboard_control(
            args,
            room_path=room_path,
            robot_path=robot_path,
            robot_position=robot_position,
            robot_rotation=robot_rotation,
            robot_yaw=robot_yaw,
        )
        return

    if not args.inside_kit and os.environ.get(INSIDE_KIT_ENV_VAR) != "1":
        launch_isaac_sim(args)
        return

    import omni.kit.app

    from run_episode import make_headless_robot_usd

    app = omni.kit.app.get_app()
    try:
        build_stage(
            app,
            room_path=room_path,
            # `make_headless_robot_usd` deactivates the asset's own `Graph`
            # prim (both `Graph/ROS_JointStates` -- a SECOND /clock and
            # /joint_states publisher, root cause of the constant "Detected
            # jump back in time" TF warnings GPU-observed during H5-H9 --
            # and `Graph/Steer_joint_Controller`, whose script_node crashes
            # after sim.reset() in headless mode) BEFORE Kit ever composes
            # the reference, the only point that actually works (OmniGraph
            # instantiates its runtime nodes at composition time; a later
            # `prim.SetActive(False)` on an already-composed graph does NOT
            # stop already-created nodes from continuing to tick -- tried
            # that first, GPU-confirmed ineffective by publisher-count
            # still 3 on /joint_states after "deactivating" it, 2026-08-11).
            # This wrapper already existed and was already used by the
            # keyboard-control/cmd-vel-control paths; build_stage was the
            # one path that never went through it.
            robot_path=make_headless_robot_usd(robot_path),
            task=args.task,
            robot_position=robot_position,
            robot_rotation=robot_rotation,
            robot_yaw=robot_yaw,
            head_placement=args.head_placement,
        )
        import omni.usd

        live_stage = omni.usd.get_context().get_stage()
        articulation_root_path = fix_single_articulation_root(
            live_stage, "/World/Robot"
        )
        asset_root_path = articulation_root_path.rsplit("/", 1)[0]
        arm_joint_names = apply_arm_joint_drive_gains(
            live_stage,
            asset_root_path,
            stiffness=args.arm_stiffness,
            damping=args.arm_damping,
            max_force=args.arm_max_force,
        )
        apply_gripper_joint_drive_gains(
            live_stage,
            asset_root_path,
            stiffness=args.gripper_stiffness,
            damping=args.gripper_damping,
            max_force=args.gripper_max_force,
        )
        clear_base_joint_friction(live_stage, asset_root_path)
        # Ported from _run_ros2_cmd_vel_control_app 2026-08-13: with the
        # joint friction cleared, the base settles completely and PhysX
        # puts the articulation to sleep, and a joint drive target does not
        # wake a sleeping articulation -- see disable_articulation_sleeping's
        # own docstring and plans/TOOL_FRAME_ROOT_CAUSE_2026-08-12.md. This
        # path had every other one of that session's base-drive fixes
        # already (clear_base_joint_friction, add_room_ground_collider,
        # apply_floor_friction, correctly-scoped pad TF via
        # publish_ros2_object_tf) except this one.
        disable_articulation_sleeping(live_stage, articulation_root_path)
        if args.ground_collider:
            add_room_ground_collider(
                live_stage, "/World/Environment/RobotRoom"
            )
        if args.floor_friction > 0.0:
            apply_floor_friction(
                live_stage, "/World/Environment/RobotRoom", args.floor_friction
            )
        if args.base_test_spin:
            set_base_drive_target_velocity(
                live_stage, asset_root_path, args.base_test_spin
            )
        if args.grasp_friction > 0.0:
            apply_grasp_friction(
                live_stage,
                asset_root_path,
                list(args.publish_object_tf or []),
                friction=args.grasp_friction,
            )
        effective_arm_stiffness = (
            robot_actuator_cfg_specs()["arms"]["stiffness"]
            if args.arm_stiffness is None
            else args.arm_stiffness
        )
        effective_arm_damping = (
            robot_actuator_cfg_specs()["arms"]["damping"]
            if args.arm_damping is None
            else args.arm_damping
        )
        effective_arm_max_force = (
            robot_actuator_cfg_specs()["arms"]["effort_limit_sim"]
            if args.arm_max_force is None
            else args.arm_max_force
        )
        if args.gravity_scale is not None:
            override_gravity_scale(live_stage, args.gravity_scale)

        enable_ros2_bridge(app, args)
        publish_ros2_clock(args)
        publish_ros2_joint_states(args, robot_prim_path=articulation_root_path)
        publish_ros2_odometry(args, chassis_prim_path=articulation_root_path)
        publish_world_map_static_tf(args)
        if not args.no_joint_command_graph:
            publish_ros2_joint_command(
                args, robot_prim_path=articulation_root_path
            )
        else:
            print("ROS2 joint_command graph SKIPPED (--no-joint-command-graph)")
        publish_ros2_object_tf(
            args, live_stage, args.publish_object_tf,
            asset_root_path=asset_root_path,
        )
        list_camera_prims_under(live_stage, asset_root_path)
        list_gripper_joint_names(live_stage, asset_root_path)
        publish_ros2_cameras(args, asset_root_path=asset_root_path)

        import omni.timeline

        timeline = omni.timeline.get_timeline_interface()
        if args.autoplay:
            timeline.play()
            print("Timeline: playing")
            if arm_joint_names:
                verify_and_fix_articulation_gains(
                    app,
                    articulation_root_path,
                    arm_joint_names,
                    effective_arm_stiffness,
                    effective_arm_damping,
                    effective_arm_max_force,
                    fix=args.force_live_gains,
                )
                if args.log_arm_efforts:
                    start_periodic_effort_logging(
                        articulation_root_path, arm_joint_names
                    )
            # Not gated behind --force-live-gains: for the gripper there is no
            # "PhysX's own values might be better" case to preserve. The
            # authored passive drives make the linkage immovable, which is
            # measured, not a tuning preference.
            verify_and_fix_gripper_gains(
                app,
                articulation_root_path,
                args.gripper_stiffness,
                args.gripper_damping,
                args.gripper_max_force,
            )
            verify_and_fix_base_drive_gains(
                app, articulation_root_path, args.base_max_force,
                skip_writes=args.no_base_gain_fix,
                test_spin=args.base_test_spin,
            )
        else:
            timeline.stop()
            print(
                "Timeline: paused. Click Play in the Isaac Sim GUI to start."
            )
        print("Close the Isaac Sim GUI window to exit.")
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()
