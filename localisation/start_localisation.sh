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

for f in arena_map.posegraph arena_map.data slam_localisation.yaml; do
  [ -f "$HERE/$f" ] || { echo "missing $f next to this script"; exit 1; }
done
ros2 pkg prefix slam_toolbox >/dev/null 2>&1 \
  || { echo "slam_toolbox is not installed: sudo apt install ros-humble-slam-toolbox"; exit 1; }

# The same DDS profile the robot's own start scripts use (UDP only; shared memory
# fails on this Jetson). Our mapping run used it too. TMR_DDS_PROFILE overrides.
DDS_PROFILE="${TMR_DDS_PROFILE:-$HOME/fastdds_udp_only.xml}"
if [ -f "$DDS_PROFILE" ]; then
  export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
  export FASTRTPS_DEFAULT_PROFILES_FILE="$DDS_PROFILE"
  echo "DDS profile: $DDS_PROFILE"
fi

PARAMS="$(mktemp --suffix=.yaml)"
sed "s#__MAP__#$HERE/arena_map#" "$HERE/slam_localisation.yaml" > "$PARAMS"

cleanup() { kill $(jobs -p) 2>/dev/null || true; rm -f "$PARAMS"; }
trap cleanup EXIT

#                                            x      y      z       yaw      pitch    roll
ros2 run tf2_ros static_transform_publisher  0.32   0.27   0.19065 3.972369 3.141593 0 base_link lidar_front &
ros2 run tf2_ros static_transform_publisher -0.32  -0.197  0.19065 0.787493 3.141593 0 base_link lidar_rear &
sleep 2

echo "localisation: map $HERE/arena_map, starting at the charging dock"
ros2 launch slam_toolbox localization_launch.py \
  slam_params_file:="$PARAMS" use_sim_time:=false
