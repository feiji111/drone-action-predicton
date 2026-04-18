#!/usr/bin/env python3
"""Extract IMU data and aligned images from a ROS 1 bag file.

Outputs:
  - imu/imu.csv: all IMU messages in timestamp order
  - images/frame_XXXXXX.<ext>: extracted image frames
  - aligned_pairs.csv: nearest IMU sample for each image
  - summary.json: extraction summary and topic metadata

This script targets ROS 1 ``.bag`` files and supports:
  - sensor_msgs/Imu
  - sensor_msgs/Image
  - sensor_msgs/CompressedImage
"""

from __future__ import annotations

import argparse
import csv
import json
from bisect import bisect_left
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

try:
    import rosbag
except ImportError as exc:  # pragma: no cover - runtime dependency
    raise SystemExit(
        "Failed to import rosbag. Please run this script inside a ROS 1 environment."
    ) from exc


def message_stamp_ns(msg: Any, fallback_time: Any) -> int:
    """Prefer message header stamp; fall back to bag event time."""
    if hasattr(msg, "header") and hasattr(msg.header, "stamp"):
        stamp = msg.header.stamp
        return int(stamp.secs) * 1_000_000_000 + int(stamp.nsecs)
    return int(fallback_time.secs) * 1_000_000_000 + int(fallback_time.nsecs)


def imu_row(msg: Any, timestamp_ns: int) -> dict[str, float | int]:
    orientation = getattr(msg, "orientation", None)
    angular_velocity = getattr(msg, "angular_velocity", None)
    linear_acceleration = getattr(msg, "linear_acceleration", None)

    return {
        "timestamp_ns": timestamp_ns,
        "orientation_x": getattr(orientation, "x", 0.0),
        "orientation_y": getattr(orientation, "y", 0.0),
        "orientation_z": getattr(orientation, "z", 0.0),
        "orientation_w": getattr(orientation, "w", 1.0),
        "gyro_x": getattr(angular_velocity, "x", 0.0),
        "gyro_y": getattr(angular_velocity, "y", 0.0),
        "gyro_z": getattr(angular_velocity, "z", 0.0),
        "accel_x": getattr(linear_acceleration, "x", 0.0),
        "accel_y": getattr(linear_acceleration, "y", 0.0),
        "accel_z": getattr(linear_acceleration, "z", 0.0),
    }


def decode_raw_image(msg: Any) -> Image.Image:
    width = int(msg.width)
    height = int(msg.height)
    encoding = str(msg.encoding).lower()
    data = np.frombuffer(msg.data, dtype=np.uint8)

    if encoding in {"rgb8", "8uc3"}:
        array = data.reshape(height, width, 3)
        return Image.fromarray(array, mode="RGB")

    if encoding == "bgr8":
        array = data.reshape(height, width, 3)[:, :, ::-1]
        return Image.fromarray(array, mode="RGB")

    if encoding in {"rgba8"}:
        array = data.reshape(height, width, 4)
        return Image.fromarray(array, mode="RGBA").convert("RGB")

    if encoding == "bgra8":
        array = data.reshape(height, width, 4)[:, :, [2, 1, 0, 3]]
        return Image.fromarray(array, mode="RGBA").convert("RGB")

    if encoding in {"mono8", "8uc1"}:
        array = data.reshape(height, width)
        return Image.fromarray(array, mode="L")

    if encoding in {"mono16", "16uc1"}:
        array = np.frombuffer(msg.data, dtype=np.uint16).reshape(height, width)
        array = (array / max(array.max(), 1) * 255.0).astype(np.uint8)
        return Image.fromarray(array, mode="L")

    raise ValueError(f"Unsupported image encoding: {msg.encoding}")


def decode_compressed_image(msg: Any) -> Image.Image:
    from io import BytesIO

    with Image.open(BytesIO(msg.data)) as image:
        return image.convert("RGB")


def save_image_message(msg: Any, output_path: Path) -> str:
    msg_type = getattr(msg, "_type", "")
    if msg_type == "sensor_msgs/CompressedImage":
        image = decode_compressed_image(msg)
    else:
        image = decode_raw_image(msg)

    if image.mode != "RGB":
        image = image.convert("RGB")

    ext = output_path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        image.save(output_path, quality=95)
    else:
        image.save(output_path)
    return output_path.name


def nearest_index(sorted_values: list[int], target: int) -> int:
    idx = bisect_left(sorted_values, target)
    if idx == 0:
        return 0
    if idx == len(sorted_values):
        return len(sorted_values) - 1

    before = sorted_values[idx - 1]
    after = sorted_values[idx]
    if abs(after - target) < abs(target - before):
        return idx
    return idx - 1


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def topic_exists(bag_path: Path, topic_name: str) -> bool:
    with rosbag.Bag(str(bag_path), "r") as bag:
        info = bag.get_type_and_topic_info().topics
        return topic_name in info


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract IMU data and images from a ROS 1 bag file.")
    parser.add_argument("--bag", required=True, help="Path to a ROS 1 .bag file")
    parser.add_argument("--output_dir", required=True, help="Where to save extracted data")
    parser.add_argument("--imu_topic", default="/imu", help="IMU topic, e.g. /imu/data")
    parser.add_argument("--image_topic", default="/camera/image_raw", help="Image topic")
    parser.add_argument(
        "--image_ext",
        choices=["png", "jpg"],
        default="jpg",
        help="Saved image file format",
    )
    parser.add_argument(
        "--max_time_diff_ms",
        type=float,
        default=50.0,
        help="Skip image/IMU matches whose timestamp gap exceeds this threshold",
    )
    args = parser.parse_args()

    bag_path = Path(args.bag).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    imu_dir = output_dir / "imu"
    image_dir = output_dir / "images"
    imu_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    if not bag_path.exists():
        raise SystemExit(f"Bag file not found: {bag_path}")

    if bag_path.suffix != ".bag":
        raise SystemExit(f"Expected a ROS 1 .bag file, got: {bag_path.name}")

    if not topic_exists(bag_path, args.imu_topic):
        raise SystemExit(f"IMU topic not found in bag: {args.imu_topic}")
    if not topic_exists(bag_path, args.image_topic):
        raise SystemExit(f"Image topic not found in bag: {args.image_topic}")

    print("=" * 60)
    print("Extracting IMU and images from ROS bag")
    print(f"bag        : {bag_path}")
    print(f"imu_topic  : {args.imu_topic}")
    print(f"image_topic: {args.image_topic}")
    print(f"output_dir : {output_dir}")
    print("=" * 60)

    imu_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []

    with rosbag.Bag(str(bag_path), "r") as bag:
        total = bag.get_message_count(topic_filters=[args.imu_topic, args.image_topic])
        for index, (topic, msg, bag_time) in enumerate(
            bag.read_messages(topics=[args.imu_topic, args.image_topic]),
            start=1,
        ):
            if index % 500 == 0 or index == total:
                print(f"processed {index}/{total} messages", end="\r", flush=True)

            timestamp_ns = message_stamp_ns(msg, bag_time)
            if topic == args.imu_topic:
                imu_rows.append(imu_row(msg, timestamp_ns))
                continue

            filename = f"frame_{len(image_rows):06d}.{args.image_ext}"
            save_image_message(msg, image_dir / filename)
            image_rows.append(
                {
                    "image_file": filename,
                    "image_timestamp_ns": timestamp_ns,
                }
            )

    print()

    if not imu_rows:
        raise SystemExit(f"No IMU messages found on topic: {args.imu_topic}")
    if not image_rows:
        raise SystemExit(f"No image messages found on topic: {args.image_topic}")

    imu_rows.sort(key=lambda row: int(row["timestamp_ns"]))
    image_rows.sort(key=lambda row: int(row["image_timestamp_ns"]))
    imu_timestamps = [int(row["timestamp_ns"]) for row in imu_rows]

    aligned_rows: list[dict[str, Any]] = []
    max_time_diff_ns = int(args.max_time_diff_ms * 1_000_000)
    dropped_matches = 0

    for image_row in image_rows:
        image_ts = int(image_row["image_timestamp_ns"])
        match_idx = nearest_index(imu_timestamps, image_ts)
        imu_match = imu_rows[match_idx]
        imu_ts = int(imu_match["timestamp_ns"])
        time_diff_ns = abs(image_ts - imu_ts)

        if time_diff_ns > max_time_diff_ns:
            dropped_matches += 1
            continue

        aligned_rows.append(
            {
                "image_file": image_row["image_file"],
                "image_timestamp_ns": image_ts,
                "imu_timestamp_ns": imu_ts,
                "time_diff_ms": time_diff_ns / 1_000_000.0,
                "accel_x": imu_match["accel_x"],
                "accel_y": imu_match["accel_y"],
                "accel_z": imu_match["accel_z"],
                "gyro_x": imu_match["gyro_x"],
                "gyro_y": imu_match["gyro_y"],
                "gyro_z": imu_match["gyro_z"],
                "orientation_x": imu_match["orientation_x"],
                "orientation_y": imu_match["orientation_y"],
                "orientation_z": imu_match["orientation_z"],
                "orientation_w": imu_match["orientation_w"],
            }
        )

    imu_csv = imu_dir / "imu.csv"
    write_csv(
        imu_csv,
        imu_rows,
        [
            "timestamp_ns",
            "orientation_x",
            "orientation_y",
            "orientation_z",
            "orientation_w",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "accel_x",
            "accel_y",
            "accel_z",
        ],
    )

    aligned_csv = output_dir / "aligned_pairs.csv"
    write_csv(
        aligned_csv,
        aligned_rows,
        [
            "image_file",
            "image_timestamp_ns",
            "imu_timestamp_ns",
            "time_diff_ms",
            "accel_x",
            "accel_y",
            "accel_z",
            "gyro_x",
            "gyro_y",
            "gyro_z",
            "orientation_x",
            "orientation_y",
            "orientation_z",
            "orientation_w",
        ],
    )

    summary = {
        "bag_file": str(bag_path),
        "imu_topic": args.imu_topic,
        "image_topic": args.image_topic,
        "imu_count": len(imu_rows),
        "image_count": len(image_rows),
        "aligned_count": len(aligned_rows),
        "dropped_image_matches": dropped_matches,
        "max_time_diff_ms": args.max_time_diff_ms,
        "output": {
            "imu_csv": str(imu_csv),
            "aligned_pairs_csv": str(aligned_csv),
            "image_dir": str(image_dir),
        },
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("=" * 60)
    print("Done")
    print(f"IMU rows       : {len(imu_rows)}")
    print(f"Image frames   : {len(image_rows)}")
    print(f"Aligned pairs  : {len(aligned_rows)}")
    print(f"Dropped images : {dropped_matches}")
    print(f"IMU CSV        : {imu_csv}")
    print(f"Aligned CSV    : {aligned_csv}")
    print(f"Images dir     : {image_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
