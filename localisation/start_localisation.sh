#!/usr/bin/env bash
# Start localisation on the robot's companion, BEFORE the mission image.
#
#   1. put the robot on its charging dock
#   2. run this, and leave it running (Ctrl-C stops it)
#   3. in another terminal, `ros2 run tf2_ros tf2_echo map base_link` must now
#      print a transform -- then run the stages as the README describes
#
# It publishes the two lidar frames (the robot's scans are stamped
# `lidar_front` / `lidar_rear`, and nothing on the robot publishes them) and runs
# slam_toolbox in localisation mode against the arena map in this folder,
# starting from the charging dock.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

set +u
# shellcheck disable=SC1091
source /opt/ros/humble/setup.bash
set -u

for f in arena_map.posegraph arena_map.data slam_localisation.yaml odom_to_tf.py tf_probe.py; do
  [ -f "$HERE/$f" ] || { echo "missing $f next to this script"; exit 1; }
done
ros2 pkg prefix slam_toolbox >/dev/null 2>&1 \
  || { echo "slam_toolbox is not installed: sudo apt install ros-humble-slam-toolbox"; exit 1; }

# The same DDS profile the robot's own start scripts use (UDP only; shared memory
# fails on this Jetson). Our mapping run used it too. TMR_DDS_PROFILE overrides.
DDS_PROFILE="${TMR_DDS_PROFILE:-${FASTRTPS_DEFAULT_PROFILES_FILE:-$HOME/fastdds_udp_only.xml}}"
if [ -f "$DDS_PROFILE" ]; then
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export FASTRTPS_DEFAULT_PROFILES_FILE="$DDS_PROFILE"
  echo "DDS profile: $DDS_PROFILE"
elif [ -n "${TMR_DDS_PROFILE:-}${FASTRTPS_DEFAULT_PROFILES_FILE:-}" ]; then
  echo "configured DDS profile does not exist: $DDS_PROFILE" >&2
  exit 1
fi
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
USE_SIM_TIME="${USE_SIM_TIME:-false}"

PARAMS="$(mktemp --suffix=.yaml)"
python3 - "$HERE" "$PARAMS" <<'PY'
import pathlib, sys
root = pathlib.Path(sys.argv[1])
pathlib.Path(sys.argv[2]).write_text(
    (root / 'slam_localisation.yaml').read_text().replace('__MAP__', str(root / 'arena_map')))
PY

cleanup() { kill $(jobs -p) 2>/dev/null || true; rm -f "$PARAMS"; }
trap cleanup EXIT

probe() {
  local rc=0
  python3 "$HERE/tf_probe.py" "$@" --ros-args -p use_sim_time:="$USE_SIM_TIME" || rc=$?
  # Only absence permits a new publisher. A stale or conflicting edge aborts.
  if [ "$rc" != 0 ] && [ "$rc" != 10 ]; then exit "$rc"; fi
  return "$rc"
}

if ! probe base_link lidar_front --allow-static; then
  ros2 run tf2_ros static_transform_publisher --x 0.32 --y 0.27 --z 0.19065 \
    --yaw 3.972369 --pitch 3.141593 --roll 0 --frame-id base_link --child-frame-id lidar_front &
fi
if ! probe base_link lidar_rear --allow-static; then
  ros2 run tf2_ros static_transform_publisher --x -0.32 --y -0.197 --z 0.19065 \
    --yaw 0.787493 --pitch 3.141593 --roll 0 --frame-id base_link --child-frame-id lidar_rear &
fi

# THE ODOMETRY TF EDGE. On this robot it is `base -> base_link` -- there is no
# `odom` frame. If the robot's own stack is not publishing it, publish it from the
# odometry topic -- never both, which would give base_link two sources.
if probe base base_link; then
  echo "odometry TF: base -> base_link is published by the robot"
else
  echo "odometry TF: base -> base_link not found; publishing it from /swerve_drive_controller/odom"
  python3 "$HERE/odom_to_tf.py" --ros-args -p use_sim_time:="$USE_SIM_TIME" &
  sleep 2
fi

echo "localisation: map $HERE/arena_map, starting at the charging dock"
ros2 launch slam_toolbox localization_launch.py \
  slam_params_file:="$PARAMS" use_sim_time:="$USE_SIM_TIME"
