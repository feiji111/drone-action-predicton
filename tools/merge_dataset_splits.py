#!/usr/bin/env python3
"""Merge selected splits inside one multimodal dataset."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def sample_ids(split_dir: Path) -> list[str]:
    ids = []
    for meta_path in sorted(split_dir.glob("window_*_meta.json")):
        stem = meta_path.stem
        ids.append(stem.removeprefix("window_").removesuffix("_meta"))
    return ids


def copy_split(source_dir: Path, output_dir: Path, source_splits: list[str], target_split: str) -> int:
    target_dir = output_dir / target_split
    target_dir.mkdir(parents=True, exist_ok=True)

    next_index = 0
    for split in source_splits:
        split_dir = source_dir / split
        for sample_id in sample_ids(split_dir):
            src_imu = split_dir / f"window_{sample_id}_imu.npy"
            src_meta = split_dir / f"window_{sample_id}_meta.json"
            dst_prefix = target_dir / f"window_{next_index:06d}"
            shutil.copy2(src_imu, f"{dst_prefix}_imu.npy")
            shutil.copy2(src_meta, f"{dst_prefix}_meta.json")
            next_index += 1
    return next_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge dataset splits, e.g. train+val -> train.")
    parser.add_argument("--input", required=True, help="Input dataset directory")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    args = parser.parse_args()

    input_dir = Path(args.input).resolve()
    output_dir = Path(args.output).resolve()

    if not input_dir.exists():
        raise SystemExit(f"Input dataset not found: {input_dir}")
    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")

    info_path = input_dir / "dataset_info.json"
    info = json.loads(info_path.read_text(encoding="utf-8")) if info_path.exists() else {}

    train_count = copy_split(input_dir, output_dir, ["train", "val"], "train")
    test_count = copy_split(input_dir, output_dir, ["test"], "test")

    merged_info = {
        "total": train_count + test_count,
        "window_size": info.get("window_size", 100),
        "overlap": info.get("overlap", 0.5),
        "imu_channels": info.get("imu_channels", 6),
        "frames_per_window": info.get("frames_per_window", 4),
        "image_resolution": info.get("image_resolution", "640x480"),
        "classes": info.get("classes", []),
        "num_classes": info.get("num_classes", 0),
        "splits": {
            "train": train_count,
            "test": test_count,
        },
        "source": str(input_dir),
        "merged_from": {
            "train": ["train", "val"],
            "test": ["test"],
        },
    }

    (output_dir / "dataset_info.json").write_text(
        json.dumps(merged_info, indent=2),
        encoding="utf-8",
    )

    print(f"Repacked dataset written to: {output_dir}")
    print(json.dumps(merged_info, indent=2))


if __name__ == "__main__":
    main()
