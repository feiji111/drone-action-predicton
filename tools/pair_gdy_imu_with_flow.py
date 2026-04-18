#!/usr/bin/env python3
"""
将 gdy 目录下的 IMU 序列与 data/racing_multimodal_flow 下的光流图按动作配对。

规则：
1. 仅在相同动作目录内配对。
2. 每个 .npy 文件配对一张光流图。
3. 将光流图复制到 gdy 下对应动作目录，并重命名为该 .npy 文件的文件名。
4. spiral 动作跳过，不做处理。

默认按文件名字典序一一配对，保证过程可复现。
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path


ACTIONS = ["ascend", "descend", "forward", "left_turn", "right_turn"]
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pair gdy IMU .npy files with optical-flow images by action."
    )
    parser.add_argument(
        "--flow-root",
        type=Path,
        default=Path("data/racing_multimodal_flow"),
        help="光流数据根目录",
    )
    parser.add_argument(
        "--imu-root",
        type=Path,
        default=Path("gdy"),
        help="IMU 序列根目录",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果目标文件已存在，则覆盖",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印将要执行的操作，不实际复制文件",
    )
    return parser.parse_args()


def list_flow_images(action_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in action_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_SUFFIXES
        and "_flow" in path.stem
    )


def list_imu_files(action_dir: Path) -> list[Path]:
    return sorted(path for path in action_dir.iterdir() if path.is_file() and path.suffix == ".npy")


def validate_dirs(flow_root: Path, imu_root: Path) -> None:
    if not flow_root.exists():
        raise FileNotFoundError(f"光流根目录不存在: {flow_root}")
    if not imu_root.exists():
        raise FileNotFoundError(f"IMU 根目录不存在: {imu_root}")

    for action in ACTIONS + ["spiral"]:
        imu_dir = imu_root / action
        if not imu_dir.exists():
            raise FileNotFoundError(f"缺少 IMU 动作目录: {imu_dir}")

    for action in ACTIONS:
        flow_dir = flow_root / action
        if not flow_dir.exists():
            raise FileNotFoundError(f"缺少光流动作目录: {flow_dir}")


def pair_action(
    action: str,
    flow_root: Path,
    imu_root: Path,
    overwrite: bool,
    dry_run: bool,
) -> tuple[int, int]:
    flow_dir = flow_root / action
    imu_dir = imu_root / action

    flow_images = list_flow_images(flow_dir)
    imu_files = list_imu_files(imu_dir)

    if not flow_images:
        raise RuntimeError(f"动作 {action} 没有可用光流图: {flow_dir}")
    if len(flow_images) < len(imu_files):
        raise RuntimeError(
            f"动作 {action} 的光流图数量不足: flow={len(flow_images)}, imu={len(imu_files)}"
        )

    copied = 0
    skipped = 0

    for imu_file, flow_image in zip(imu_files, flow_images):
        target_path = imu_dir / f"{imu_file.stem}{flow_image.suffix.lower()}"
        if target_path.exists() and not overwrite:
            skipped += 1
            print(f"[SKIP] {target_path} 已存在")
            continue

        print(f"[PAIR] {action}: {flow_image.name} -> {target_path.name}")
        if not dry_run:
            shutil.copy2(flow_image, target_path)
        copied += 1

    return copied, skipped


def main() -> None:
    args = parse_args()
    flow_root = args.flow_root.resolve()
    imu_root = args.imu_root.resolve()

    validate_dirs(flow_root, imu_root)

    print("=" * 60)
    print("gdy IMU 与光流图配对脚本")
    print(f"flow_root : {flow_root}")
    print(f"imu_root  : {imu_root}")
    print("skip      : spiral")
    print(f"dry_run   : {args.dry_run}")
    print(f"overwrite : {args.overwrite}")
    print("=" * 60)

    total_copied = 0
    total_skipped = 0

    for action in ACTIONS:
        copied, skipped = pair_action(
            action=action,
            flow_root=flow_root,
            imu_root=imu_root,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )
        total_copied += copied
        total_skipped += skipped
        print(
            f"[DONE] {action}: copied={copied}, skipped={skipped}, "
            f"imu_count={len(list_imu_files(imu_root / action))}"
        )

    print("=" * 60)
    print(f"total_copied  : {total_copied}")
    print(f"total_skipped : {total_skipped}")
    print("spiral 未处理。")
    print("=" * 60)


if __name__ == "__main__":
    main()
