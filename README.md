# EBiM Task 3 — Assisted Living & Feeding

Autonomous table setting and clearing on a dual-arm mobile manipulator.

> ## ▶ Running this on the testbed? Read **[OPERATING.md](OPERATING.md)**
>
> Every command, both scenarios, which machine to run it on, what the robot does
> at each step, and what every exit code means. It is the only document you need.

## The image

```
ghcr.io/sushruths04/ebim-task3-mission:1.2.0
```

Built for **linux/amd64 and linux/arm64** — the companion is a Jetson, and
Docker selects the right build automatically. Nothing is emulated on the robot.

## What it can do

| scenario | command | what happens |
|---|---|---|
| **Single pick-and-place** | `autorun --object "the rim of the cup" --onto "the rim of the plate"` | The robot squares up to an object in front of it, picks it, places it on the named destination, and homes. No driving between rooms. |
| **Stage 1 — table setting** | `--stage 1` | From its start pose: drives to the kitchen, finds each item, picks it, raises the rail, drives to the dining table, and releases it at its lettered place. Cup, bowl, then plate. |
| **Stage 4 — clearing** | `--stage 4` | The same in reverse: each item is collected from the dining table and returned to the marked area in the kitchen. |
| **Both** | `--stage all` | Stage 1, then Stage 4. |

Every one of those needs `--i-am-on-the-estop`. Without it nothing is commanded.

## Try it without moving anything

All three are safe to run at any time and take seconds:

```bash
# 1. is the image complete?
docker run --rm ghcr.io/sushruths04/ebim-task3-mission:1.2.0 --selftest

# 2. print the whole driving route with clearances
docker run --rm --network host -e ROS_DOMAIN_ID=0 \
  ghcr.io/sushruths04/ebim-task3-mission:1.2.0 --plan

# 3. on the robot: is it localised in the map?
ros2 run tf2_ros tf2_echo map base_link
```

**[OPERATING.md](OPERATING.md) explains what to do with each answer**, and gives
the full run commands including the two mounts every real run needs.

## Requirements, in brief

| | |
|---|---|
| Machine | the computer running the FR3 arm stack — where `ros2 topic list \| grep /right/` prints topics |
| Stack | ROS 2 Humble control stack accepting goals; cameras publishing |
| Network | `--network host`, and outbound HTTPS for the vision service |
| Compute | no GPU required |
| **Safety** | **an operator must hold the emergency stop.** The container asserts this before it moves and returns the arm home at the end regardless of outcome. |

## Documents

| file | for |
|---|---|
| **[OPERATING.md](OPERATING.md)** | **running it — start here** |
| [README_MISSION.md](README_MISSION.md) | detail on the navigation build |
| this file | what the project is |

## Licence

Proprietary, all rights reserved — see [LICENSE](LICENSE). The benchmark
organizers are granted a licence to run this image for evaluation and to publish
the result. No other use is permitted.
