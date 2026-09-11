#!/usr/bin/env python3
"""Exit 0 for a usable TF edge, 10 if absent, 2 for stale/conflicting TF."""
import argparse
import time

import rclpy
import yaml
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("parent")
    ap.add_argument("child")
    ap.add_argument("--allow-static", action="store_true")
    args, ros_args = ap.parse_known_args()
    rclpy.init(args=ros_args)
    node = Node("ebim_tf_probe")
    buf = Buffer()
    listener = TransformListener(buf, node)
    deadline = time.monotonic() + 5
    try:
        while time.monotonic() < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        try:
            t = buf.lookup_transform(args.parent, args.child, rclpy.time.Time())
        except Exception:
            frames = yaml.safe_load(buf.all_frames_as_yaml()) or {}
            if args.child in frames:
                print(f"TF conflict: {args.child} already has parent "
                      f"{frames[args.child].get('parent')}; cannot add {args.parent}")
                return 2
            print(f"TF absent: {args.parent} -> {args.child}")
            return 10
        stamp = t.header.stamp.sec + t.header.stamp.nanosec * 1e-9
        age = node.get_clock().now().nanoseconds * 1e-9 - stamp
        if not (args.allow_static and stamp == 0) and not -0.5 <= age <= 1.0:
            print(f"TF stale: {args.parent} -> {args.child}, age {age:.2f} s")
            return 2
        print(f"TF present: {args.parent} -> {args.child}")
        return 0
    finally:
        listener.unregister()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
