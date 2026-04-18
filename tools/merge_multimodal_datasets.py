#!/usr/bin/env python3
"""Merge multiple multimodal datasets with train/val/test splits."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


SPLITS = ("train", "val", "test")


def sample_ids(split_dir: Path) -> list[str]:
    ids = []
    for meta_path in sorted(split_dir.glob("window_*_meta.json")):
        stem = meta_path.stem
        ids.append(stem.removeprefix("window_").removesuffix("_meta"))
    return ids


def merge_split(source_dirs: list[Path], split: str, output_dir: Path) -> int:
    split_out = output_dir / split
    split_out.mkdir(parents=True, exist_ok=True)

    next_index = 0
    for source_dir in source_dirs:
        split_in = source_dir / split
        for sample_id in sample_ids(split_in):
            src_imu = split_in / f"window_{sample_id}_imu.npy"
            src_meta = split_in / f"window_{sample_id}_meta.json"
            dst_prefix = split_out / f"window_{next_index:06d}"
            shutil.copy2(src_imu, f"{dst_prefix}_imu.npy")
            shutil.copy2(src_meta, f"{dst_prefix}_meta.json")
            next_index += 1

    return next_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multimodal datasets into one output directory.")
    parser.add_argument("--inputs", nargs="+", required=True, help="Input dataset directories")
    parser.add_argument("--output", required=True, help="Output dataset directory")
    args = parser.parse_args()

    input_dirs = [Path(path).resolve() for path in args.inputs]
    output_dir = Path(args.output).resolve()

    for input_dir in input_dirs:
        if not input_dir.exists():
            raise SystemExit(f"Input dataset not found: {input_dir}")
        for split in SPLITS:
            if not (input_dir / split).exists():
                raise SystemExit(f"Missing split '{split}' in dataset: {input_dir}")

    if output_dir.exists():
        raise SystemExit(f"Output directory already exists: {output_dir}")

    split_counts: dict[str, int] = {}
    for split in SPLITS:
        split_counts[split] = merge_split(input_dirs, split, output_dir)

    classes: set[str] = set()
    for input_dir in input_dirs:
        info_path = input_dir / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            classes.update(info.get("classes", []))

    merged_info = {
        "total": sum(split_counts.values()),
        "window_size": 100,
        "overlap": 0.5,
        "imu_channels": 6,
        "frames_per_window": 4,
        "image_resolution": "640x480",
        "classes": sorted(classes),
        "num_classes": len(classes),
        "splits": split_counts,
        "sources": [str(path) for path in input_dirs],
    }

    (output_dir / "dataset_info.json").write_text(
        json.dumps(merged_info, indent=2),
        encoding="utf-8",
    )

    print(f"Merged dataset written to: {output_dir}")
    print(json.dumps(merged_info, indent=2))


if __name__ == "__main__":
    main()
