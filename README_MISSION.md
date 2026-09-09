# EBiM Task 3 — mission build

Table setting and clearing on a dual-arm mobile manipulator: the robot drives
from its start pose to the kitchen table, carries each item to its lettered
place on the dining table, and brings the items back to the marked area in the
kitchen.

> [!IMPORTANT]
> **Built for `linux/amd64` and `linux/arm64`.** The companion is a Jetson —
> arm64 — and the original submitted image is amd64 only, so it cannot run
> there. Docker picks the right architecture automatically; nothing is emulated
> on the robot.
>
> This is a **second, separate image**. The pick-and-place build is unchanged
> and remains what was submitted. This one adds room-scale navigation, and it
> has **not yet been run on a robot** — see *Status* at the bottom before
> planning a session around it.

## Run

Always start here. This plans the whole route against the map, prints it, and
**commands nothing**:

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  ghcr.io/sushruths04/ebim-task3-mission:1.1.0 --plan
```

If every leg reports `PASS`, the robot can drive the route. Then, with an
operator on the emergency stop:

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT=<supplied to the organizers directly> \
  -e TMR_WS=/ws -v $HOME/ros2_ws:/ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.1.0 \
  --stage 1 --i-am-on-the-estop
```

| command | what it does |
|---|---|
| `--plan` | print the route and exit; **moves nothing** |
| `--selftest` | check the image is complete; moves nothing |
| `--stage 1` | kitchen table → the lettered places on the dining table |
| `--stage 4` | dining table → the marked area in the kitchen |
| `--stage all` | both, in order (default) |
| `--fresh` | ignore any part-finished run and start over |
| `autorun ...` | the **single pick-and-place**, same behaviour as the submitted build |

Because the submitted image is amd64 only, this one also carries the plain
pick-and-place so a single arm64 image covers both:

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT=<supplied to the organizers directly> \
  -e TMR_WS=/ws -v /home/tmr-user/ros2_ws:/ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.1.0 \
  autorun --object "the rim of the cup" --onto "the rim of the plate" \
  --i-am-on-the-estop
```

## Environment

| variable | required | meaning |
|---|---|---|
| `VISION_ENDPOINT` | for a real run | the vision service; not needed for `--plan` |
| `TMR_WS` | **yes** | the robot's own ROS workspace, mounted into the container. The policy needs `franka_msgs` from it — the packaged version lacks the `PTPMotion` action. |
| `ROS_DOMAIN_ID` | no | must match the control stack; default `0` |

## Which machine

**The computer that runs the FR3 arm control stack** — the one where
`ros2 topic list | grep /right/` prints topics.

The image is ROS 2 Humble and ROS 2 does not communicate across distros. Started
on a host running a different distro it will report `no /right/* topics` and
stop, while the container and the robot are both perfectly healthy.

## What it does each round

1. Parks the idle arm inside the base envelope — both arms share one rail, so
   the idle one is put away before anything descends or drives.
2. Plans the route around the map and whatever the sensors see now. **A leg with
   less than the robot's own width of clearance is refused, not driven.**
3. Drives to the table, then closes the last stretch on the *object*, not on the
   waypoint.
4. Picks it, carries it, and places it at the destination.
5. Returns the arm home at the end, whatever happened.

Each delivered item is recorded before the next begins, so a run stopped
part-way can be resumed inside the same container.

## Status — read this before planning a session

- The route plans clean against the map: **16 of 16 legs**, all above the
  robot's half-width of clearance.
- Every module loads inside the image; `--selftest` covers both.
- **No part of this has run on a robot.** The pick-and-place it calls is the
  measured one; the driving around it is not.
- The map is of one specific room. If the evaluation room is a different space,
  the waypoints do not apply and only `--plan` is meaningful.

`--plan` is safe anywhere and costs nothing. Run it first, always.

## Exit codes

| code | meaning |
|---|---|
| `0` | finished |
| `1` | a stage failed, or the route is not driveable — the output names which |
| `3` | the image is incomplete; contact us |
| `5` | no robot found — check the control stack, `ROS_DOMAIN_ID`, `--network host` |

## Licence

Proprietary, all rights reserved — see [LICENSE](LICENSE). The benchmark
organizers are granted a licence to run this image for evaluation and to publish
the result. No other use is permitted.
