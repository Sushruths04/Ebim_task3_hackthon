# EBiM Task 3 — mission build

Table setting and clearing on a dual-arm mobile manipulator: the robot drives
from its start pose to the kitchen table, carries each item to its lettered
place on the dining table, and brings the items back to the marked area in the
kitchen.

> [!IMPORTANT]
> **Built for `linux/amd64` and `linux/arm64`.** The companion is a Jetson —
> arm64 — and the original submitted image is amd64 only, so it cannot run
> there. Docker picks the right architecture automatically; nothing is emulated
> on the robot. On the Jetson every `docker` command needs `--network host`.
>
> This is a **second, separate image**. The pick-and-place build is unchanged
> and remains what was submitted. This one adds room-scale navigation.
> **Please begin with the test run below** — it commands nothing and confirms
> the route against your cell before anything moves.

## Run

**Start with the test run.** It plans the whole route against the map, prints
every leg with its clearance, and **commands nothing**:

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.1 --plan
```

If every leg reports `PASS`, the robot can drive the route. Then set the vision
service once — `export VISION_ENDPOINT='<supplied to you directly>'` — and,
with an operator on the emergency stop:

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT \
  -v /home/tmr-user/fastdds_udp_only.xml:/home/tmr-user/fastdds_udp_only.xml:ro \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/home/tmr-user/fastdds_udp_only.xml \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.1 \
  --stage 1 --i-am-on-the-estop
```

| command | what it does |
|---|---|
| `--plan` | print the route and exit; **moves nothing** |
| `--selftest` | check the image is complete, and the robot workspaces if mounted; moves nothing |
| `--stage 1` | kitchen table → the lettered places on the dining table |
| `--stage 4` | dining table → the marked area in the kitchen |
| `--stage all` | both, in order (default) |
| `--fresh` | ignore any part-finished run and start over |
| `--assign cup=b` | which lettered place each object goes to; default `cup=c,bowl=b,plate=a`. Naming one object is enough; the others fill the free letters. Pass the same to both stages. |
| `--move cup=a` | move **only** the named object(s): `--stage 1 --move cup=a` takes the cup from the kitchen table to **a**; `--stage 4 --move cup=a` brings it back from **a**. Several: `cup=a,plate=c`. |
| `autorun ...` | the **single pick-and-place**, same behaviour as the submitted build |

Because the submitted image is amd64 only, this one also carries the plain
pick-and-place so a single arm64 image covers both:

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT \
  -v /home/tmr-user/fastdds_udp_only.xml:/home/tmr-user/fastdds_udp_only.xml:ro \
  -e FASTRTPS_DEFAULT_PROFILES_FILE=/home/tmr-user/fastdds_udp_only.xml \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.1 \
  autorun --object "the rim of the cup" --onto "the rim of the plate" \
  --i-am-on-the-estop
```

## Environment and mounts

| setting | required | meaning |
|---|---|---|
| `VISION_ENDPOINT` | for a real run | the vision service; not needed for `--plan` |
| `ros2_ws` mount | **yes** | the robot's arm workspace. The policy needs `franka_msgs` from it — the packaged version lacks the `PTPMotion` action. |
| `tams_ws` mount | **yes** | the robot's spine workspace, for `franka_spine_msgs` |
| `ROS_DOMAIN_ID` | no | must match the control stack; default `0` |

Mount each workspace **at its own path** (`-v X:X:ro`), as above: they are built
with symlinks to absolute paths, so mounted anywhere else they appear empty. The
container detects that and prints the path to use.

The robot's home pose is carried in the image; mounting
`/home/tmr-user/teleop_home_pose.yaml` onto the same path uses the host's
current file instead.

## Which machine

**The computer that runs the FR3 arm control stack** — the one where
`ros2 topic list | grep /right/` prints topics.

The image is ROS 2 Humble and ROS 2 does not communicate across distros. Started
on a host running a different distro it will report `no /right/* topics` and
stop, while the container and the robot are both perfectly healthy.

## What it does each round

1. Parks the idle arm inside the base envelope — both arms share one rail, so
   the idle one is put away before anything descends or drives.
2. Plans the route around the mapped walls and furniture, through the doorway's
   centre line, and drives exactly that route. **A route with less than the
   robot's own half-width of clearance is refused, not driven.**
3. Drives to the table, then closes the last stretch on the *object*, not on the
   waypoint.
4. Picks it, brings the arm home, raises the rail, carries it, places it at the
   destination, and homes the arm again.
5. Returns the arm home at the end, whatever happened.

Each delivered item is recorded before the next begins, so a run stopped
part-way can be resumed inside the same container.

## Test run first — recommended sequence

1. **`--selftest`**, with the two workspace mounts — confirms the image is
   complete and the workspaces are visible. Moves nothing.
2. **`--plan`** — prints the route with clearances. Moves nothing.
3. **`tf2_echo map base_link`** on the robot — confirms it is localised in the
   map, which is what makes the waypoints meaningful in your cell.
4. **`--stage 1`** with an operator on the emergency stop.

Steps 1 to 3 are free and take seconds. Please run them before step 4.

The route currently plans clean: **16 of 16 legs**, every one above the robot's
own half-width of clearance. The map is of one specific cell — if the evaluation
room is a different space, step 3 is what tells you, and only `--plan` is
meaningful until it passes.

## Exit codes

| code | meaning |
|---|---|
| `0` | finished |
| `1` | a stage failed, or the route is not driveable — the output names which |
| `3` | the image is incomplete; contact us |
| `4` | vision service unset, or no answer within 45 s |
| `5` | no robot found — check the control stack, `ROS_DOMAIN_ID`, `--network host` |
| `6` | the robot workspaces are not visible — mount each at its own path |

## Licence

Proprietary, all rights reserved — see [LICENSE](LICENSE). The benchmark
organizers are granted a licence to run this image for evaluation and to publish
the result. No other use is permitted.
