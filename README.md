# EBiM Task 3 — Assisted Living & Feeding

Autonomous table setting and clearing on a dual-arm mobile manipulator.
Everything needed to run it is on this page.

> [OPERATING.md](OPERATING.md) is the same material laid out as a step-by-step
> operating guide, and [ALTERNATIVES.md](ALTERNATIVES.md) covers hosting the
> perception yourself.

## The robot running it

Five recordings from the arena. Each is one continuous take, filmed on a phone,
so they are portrait and softer than the rig's own cameras.

**Bowl into the tray** · 12 s · fixed camera, the closest view of the grasp

https://github.com/user-attachments/assets/e68bbfc2-5bac-4e6a-bb27-80af4ec4ed0d

**Cup into the tray** · 22 s

https://github.com/user-attachments/assets/cd995226-40ab-4beb-bae3-06b91c131416

**Cup from the tray onto the plate, take one** · 35 s · the operator's screen
shows the live rim fit the grasp is planned from

https://github.com/user-attachments/assets/aa7a602c-c86c-49de-bc83-7bc601628b17

**Cup from the tray onto the plate, take two** · 39 s

https://github.com/user-attachments/assets/517f82cb-209f-4633-b268-a3b2c2b10497

**Cup into the bowl** · 24 s as shown

> **This clip is slowed down from the original time-lapse recording. It does
> not play at the robot's real speed.** Every other clip on this page is real
> time, straight off the phone.

https://github.com/user-attachments/assets/1476cb16-28f6-4556-b693-a7c65c700f21

## 1 · The image

```
ghcr.io/sushruths04/ebim-task3-mission:1.4.2
```

Built for **linux/amd64 and linux/arm64**. The companion is a Jetson (arm64) and
Docker selects the right build automatically — nothing is emulated on the robot.

Nothing to build. Pull it, or let `docker run` pull it on first use.

**On the Jetson, every `docker` command needs `--network host`.** Its kernel
lacks a module Docker's default bridge networking needs; every command on this
page already includes it.

### What changed in 1.4.2

Includes the 1.3.0 packaging fixes and these updates for the 1.4.1 report:

- Each capture selects the newest available colour transport, so a stopped
  compressed stream cannot hide newer raw frames.
- A camera failure stops acquisition before the base can start an object search.
- `--check-data` checks repeated arm and wrist samples without moving the robot.

- `control_msgs` is now in the image.
- The head-camera and wrist-camera calibrations, and the robot's home pose, are
  now in the image.
- The two robot workspaces are mounted at their own paths (section 8). The
  earlier `/ws` mount left them empty.
- `--selftest` now builds every robot client and reads every calibration file
  exactly as a real run does, and it verifies the robot workspaces when they are
  mounted. The build runs it too.
- A real run checks the robot's messages, the vision service and the robot
  itself before anything moves, and stops with a clear message and exit code
  (section 9).
- Stages 1 and 4 drive the planned route between rooms, turn in open floor
  before docking at a table, home the arm before the rail goes up and the base
  drives, and slide the base along the table when needed so the arm can reach
  the release point.
- `--assign` sets the lettered places for a round; `--move cup=a` moves a single
  object (section 6).
- The image uses the robot's own UDP-only DDS profile, so data from the camera
  and lidar drivers reaches the container, and a real run now checks that arm
  and camera data actually arrives (exit 7 if not).
- The base is re-activated before each navigation route and manipulation
  approach/release move. If re-activation reports a problem, the drive is
  still attempted; the base refuses by itself if it does not arrive.
- `localisation/` starts localisation in the arena map when the robot has none
  (section 4d).

### Run it in this order

On the computer that runs the arm control stack (section 2), in one shell:

```bash
# 0. the vision service -- supplied to you directly. Every real run reads it.
export VISION_ENDPOINT='<supplied to you directly>'
```

1. **`--selftest`** — section 4a. Moves nothing.
2. **`--plan`** — section 4b. Moves nothing.
3. **`tf2_echo map base_link`** — section 4c. Tells you whether the stages can
   drive in this room.
4. **Stage 1** — section 6, with an operator on the emergency stop.
5. **Stage 4** — section 6.

If step 3 finds no map frame, start localisation (section 4d) and repeat step 3.
If it still finds none, run the single pick-and-place (section 5) instead of the
stages.

## 2 · Which machine to run it on

**The computer that runs the FR3 arm control stack** — the one where this prints
topics:

```bash
ros2 topic list | grep /right/
```

Use the ROS 2 Humble control stack and matching message workspaces. Communication
with other ROS distributions is not part of this setup; a missing topic alone
does not identify a distribution mismatch.

## 3 · What must already be running

- the FR3 arm control stack, accepting goals
- the cameras publishing
- the mobile base
- **an operator holding the emergency stop** — the container refuses to move
  without `--i-am-on-the-estop`. After a run starts, it attempts to return the
  arm home; a controller fault can prevent that

## 4 · Checks that move nothing

These checks send no motion commands. Please run them first.

```bash
# a) is the image complete, and can it see the robot's workspaces?
docker run --rm --network host \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.2 --selftest

# b) print the whole driving route with clearances
docker run --rm --network host -e ROS_DOMAIN_ID=0 \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.2 --plan

# c) on the companion, using the same domain and DDS profile as localisation
source /opt/ros/humble/setup.bash
export ROS_DOMAIN_ID=0
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export FASTRTPS_DEFAULT_PROFILES_FILE="$HOME/fastdds_udp_only.xml"
ros2 run tf2_ros tf2_echo map base_link
```

- **(a)** should end `self-test PASSED -- image complete, robot workspaces verified`.
  If a workspace is mounted at the wrong path it says so and prints the path to
  use.
- **(b)** should end `16/16 legs clear. ROUTE IS DRIVEABLE.`
- **(c)** is the important one. The autonomous stages drive to coordinates in a
  map, so the robot must place itself in that map for them to mean anything. If
  this prints a changing timestamp, verify it comes from the supplied arena
  localisation and that the dock pose is near x 1.86, y 3.48, yaw −90.5°. A
  transform alone does not establish which map is loaded. **If it cannot find
  the frame, start localisation (section 4d) and check again.** Only if it still
  cannot, run the single pick-and-place (section 5).

## 4d · If (c) cannot find the map frame — start localisation

The autonomous stages need the robot placed in the arena map. If
`tf2_echo map base_link` says the frame does not exist, nothing is running that
places it there. The `localisation/` folder in this repository starts it, using
the saved pose graph recorded with `map_final.yaml` and `waypoints_final.json`.
It runs **slam_toolbox localisation**, not a new mapping session or AMCL.
Install the localiser on the companion if needed:

```bash
sudo apt install ros-humble-slam-toolbox
```

Do not start it alongside another localiser or mapper. The existing map must be
the supplied arena map before the stages can use its coordinates.

1. **Put the robot on its charging dock.** Localisation starts from there.
2. On the companion, in its own terminal, and leave it running:

   ```bash
   git clone https://github.com/Sushruths04/Ebim_task3_hackthon.git
   cd Ebim_task3_hackthon
   # For an existing clone: cd to it and run git pull --ff-only instead.
   export ROS_DOMAIN_ID=0
   ./localisation/start_localisation.sh
   ```

   It supplies missing lidar frames (`lidar_front` / `lidar_rear`), supplies the
   odometry transform `base -> base_link` from `/swerve_drive_controller/odom` if
   it is absent, and runs `slam_toolbox` in localisation mode against
   `arena_map`. It uses the robot's
   own DDS profile (`~/fastdds_udp_only.xml`) when present, and respects
   `FASTRTPS_DEFAULT_PROFILES_FILE` or `TMR_DDS_PROFILE` overrides. It refuses a
   stale odometry edge or a conflicting parent instead of adding another source.
   The saved rig used `base` as its odometry frame; no `odom` frame is required.
3. In another terminal, repeat (c): `ros2 run tf2_ros tf2_echo map base_link`
   should now print fresh transforms near x 1.86, y 3.48, yaw −90.5° — the dock.
   Check this before moving. A frame with a different origin is not usable.
4. Run the stages (section 6) while it keeps running. Ctrl-C stops it afterwards.

The initial pose is `charging_dock_start`, configured as
`map_start_pose: [1.8562, 3.4822, -1.5794]` (yaw in radians). To start elsewhere,
a measured pose in this same map must be supplied; do not use the dock seed.

Before the first pick, check camera data without sending goals:

```bash
docker run --rm --network host -e ROS_DOMAIN_ID=0 \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.2 --check-data
```

It accepts either supported colour transport, requires repeated fresh samples,
and prints callback counts. Exit 7 means the data check failed.

## 5 · Scenario A — a single pick-and-place

The robot picks one named object and puts it on another, both in front of it.
**It does not drive between rooms.** Put the cup and the plate on the table in
front of the robot.

First, set the vision service once in that shell — it is supplied to you
directly; `-e VISION_ENDPOINT` below passes it into the container:

```bash
export VISION_ENDPOINT='<supplied to you directly>'
```

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.2 \
  autorun --object "the rim of the cup" --onto "the rim of the plate" \
  --i-am-on-the-estop
```

What it does, printed as it goes:

1. Parks the left arm inside the base envelope — both arms share one rail, so
   the idle one is put away before anything descends.
2. Homes the right arm.
3. Drives the base in small steps until the object is reachable, searching back,
   then left, then right if neither camera can see it. Gives up bounded rather
   than wandering.
4. Locates the object and the destination in one look, in the arm's own frame.
5. Picks, returns through home, places, releases.
6. Homes the arm.

The calibrated rim-grasp objects are the cup, bowl and plate. Naming another
object does not give it a calibrated grasp.

**The object is always set down in the centre of the destination.** `--onto
"the rim of the plate"` means *in the middle of the plate*: the robot finds the
plate by its rim, fits a circle to it, and releases at the circle's centre, onto
the plate's floor. So to put the **bowl in the centre of the plate**:

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.2 \
  autorun --object "the rim of the bowl" --onto "the rim of the plate" \
  --i-am-on-the-estop
```

Name the **rim**, not the middle, for both: the rim is what the robot grips the
object by and what it measures the destination by. Keep other objects a hand's
width away from the destination, so its rim is seen on its own.

## 6 · Scenario B — the autonomous stages

### Stage 1 — kitchen to the dining table

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.2 \
  --stage 1 --i-am-on-the-estop
```

Starting from its dock, for the cup, then the bowl, then the plate:

| step | what happens |
|---|---|
| **0** | Parks the left arm and homes the right one. Once, at the start. |
| **1** | Plans a route around the mapped walls and furniture — through the doorway's centre line between rooms — and drives exactly that route to the kitchen table. A route with less than the robot's own half-width of clearance is **refused, not driven**. |
| **2** | At the table, closes the last stretch on the **object**, not on the waypoint — steering on what the cameras report, searching back / left / right if it is not yet in view. |
| **3** | Picks it. |
| **4** | **Brings the arm home, raises the rail, then drives.** The pick leaves the arm low over the table it took from; nothing is carried in that posture. |
| **5** | Drives to the dining table, to the pose facing that item's lettered place. |
| **6** | Finds the letter with the head camera, slides the base along the table if the arm needs it to reach, releases the object there, and homes the arm. If the letter cannot be seen it places in front of the pose that faces that letter, and says so in the log. |
| **7** | Returns to the kitchen for the next item. |

The arm is homed on completion; if homing fails, the run reports failure.

**Which letter is whose.** By default the cup goes to **c**, the bowl to **b**
and the plate to **a**. If the round assigns them differently, add `--assign`,
for example `--stage 1 --assign cup=b --i-am-on-the-estop`. Naming one object is
enough — the others keep their letter if it is free and otherwise take the one
left over (`cup=b` gives cup b, bowl c, plate a); the run prints the result
first. Pass the same `--assign` to Stage 4, so each item is collected from where
it was put.

**One object, one letter.** `--move` moves only what it names:
`--stage 1 --move cup=a --i-am-on-the-estop` takes the cup from the kitchen
table and places it on **a**, and nothing else;
`--stage 4 --move cup=a --i-am-on-the-estop` brings the cup back from **a** to
the black rectangle. Several at once: `--move cup=a,plate=c`.

### Stage 4 — dining table back to the kitchen

```bash
docker run --rm --network host \
  -e ROS_DOMAIN_ID=0 \
  -e VISION_ENDPOINT \
  -v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro \
  -v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro \
  ghcr.io/sushruths04/ebim-task3-mission:1.4.2 \
  --stage 4 --i-am-on-the-estop
```

The same motion in the other direction: each item is collected from its lettered
place on the dining table and placed on the black rectangle on the left of the
kitchen table.

`--stage all` runs Stage 1 and then Stage 4.

## 7 · Every command

| command | moves the robot? | what it does |
|---|---|---|
| `--selftest` | no | the image is complete; with the workspaces mounted, verifies them too |
| `--plan` | no | prints the whole route with clearances |
| `--version` | no | build stamp |
| `--check-data [--side right]` | no | repeated arm and wrist samples; no service or action goals |
| `autorun --object X --onto Y --i-am-on-the-estop` | **yes** | one pick-and-place |
| `--stage 1 --i-am-on-the-estop` | **yes** | kitchen → dining table |
| `--stage 4 --i-am-on-the-estop` | **yes** | dining table → kitchen |
| `--stage all --i-am-on-the-estop` | **yes** | both, in order |
| `--fresh` | — | ignore a part-finished run and start over |
| `--assign cup=b` | — | which lettered place each object goes to; default `cup=c,bowl=b,plate=a` |
| `--move cup=a` | — | move only the named object(s): stage 1 to that letter, stage 4 back from it |

**Without `--i-am-on-the-estop` no motion goals are sent.**

## 8 · Environment and mounts

| setting | required | meaning |
|---|---|---|
| `VISION_ENDPOINT` | for a real run | the vision service, supplied to you directly. Hosted and operated by us — nothing to set up, no credential for you to manage. Not needed for `--plan` or `--selftest`. |
| `-v /home/tmr-user/ros2_ws:/home/tmr-user/ros2_ws:ro` | **yes, for a real run** | the robot's arm workspace. The policy needs `franka_msgs` from it; the packaged version lacks the `PTPMotion` action. |
| `-v /home/tmr-user/tams_ws:/home/tmr-user/tams_ws:ro` | **yes, for a real run** | the robot's spine workspace, for `franka_spine_msgs`. |
| `ROS_DOMAIN_ID` | no | must match the control stack; default `0` |
| `FASTRTPS_DEFAULT_PROFILES_FILE` | no | the image supplies a UDP-only profile. To override, mount your XML and pass its container path with `-e FASTRTPS_DEFAULT_PROFILES_FILE=...`. |
| `-v /home/tmr-user/teleop_home_pose.yaml:/home/tmr-user/teleop_home_pose.yaml:ro` | no | the image carries the robot's recorded home pose; this mount uses the host's current file instead. The log says which is in use. |

**Mount each workspace at its own path**, as above. They are built with
symlinks to absolute paths, so mounted anywhere else they appear empty — the
container detects that and prints the path to use. At these paths the container
finds them with no further setting; `TMR_WS` / `TAMS_WS` override the location
if a workspace lives elsewhere.

## 8b · Hosting perception yourself

The policy asks one thing of perception: an image and a phrase in, one pixel
out. No calibration, no knowledge of the robot. Any service answering that
contract can be substituted — including one you host.
See [ALTERNATIVES.md](ALTERNATIVES.md) for the contract and for open models that
fit it unmodified.

## 9 · If something goes wrong

Before anything moves, a real run checks the robot's messages, the vision
service, the robot, and that data from the arm and the wrist camera actually
arrives, in that order, and stops at the first that fails.

| exit code | meaning | what to check |
|---|---|---|
| `0` | finished | — |
| `1` | a stage failed, or the route is not driveable | the output names which |
| `3` | the image is incomplete | contact us |
| `4` | vision service unset, or no answer within 45 s | `VISION_ENDPOINT`, and outbound HTTPS from that machine |
| `5` | no robot found | the control stack, `ROS_DOMAIN_ID`, `--network host`, and that you are on the arm computer |
| `6` | the robot workspaces are not visible | the two workspace mounts, each at its own path (section 8) |
| `7` | robot topics are visible but their data does not arrive | arm and wrist publishers, the callback counts from `--check-data`, and that any overridden DDS profile is readable and correct for this network |

A pre-flight refusal occurs before motion. After a run starts, cleanup attempts
to home the arm; a controller fault may prevent homing.

- *"the route to '…' passes N cm from an obstacle, under the robot's 45 cm
  half-width. Refused, not driven."* — something blocks the way; the base did
  not drive it. (`--plan` shows the same leg as `FAIL`.)
- *"the map pose is N s old … Refusing to drive on a frozen pose."* —
  localisation stopped publishing `map -> base_link`; check it with
  `tf2_echo map base_link` (section 4c).
- *"joint N would move X rad, over the 0.6 rad cap"* — an arm goal was rejected
  before being sent. Nothing moved; the object is still held.
- *"… goal was ABORTED by the controller"* — the arm or rail controller stopped
  a move (for example a contact reflex). The run stops rather than continue on
  a wrong idea of where the arm is.

Every stage prints its own measurement, and a run stops at the first stage that
fails rather than carrying a fault forward. **If a run fails, those printed
lines are the diagnosis** — please send the last 20 lines and the output of
`--version`.

## 10 · Between rounds

Return the robot to its marked start pose, put the items back on the kitchen
table, clear the destination surface, and open the gripper. Each `docker run`
starts a fresh round; nothing carries over.

To run **Stage 4 on its own**, put each item on its lettered place on the
dining table first (cup on **c**, bowl on **b**, plate on **a**, or as the
round's `--assign` says) and pass the same `--assign` or `--move`.

To preview any run without moving anything, give it the same options with
`--plan`, e.g. `--plan --stage 1 --move cup=a`: it prints which object goes to
which letter and every drive it would make.

## Licence

Proprietary, all rights reserved — see [LICENSE](LICENSE). The benchmark
organizers are granted a licence to run this image for evaluation and to publish
the result. No other use is permitted.
