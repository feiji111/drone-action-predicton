#!/usr/bin/env python3
"""
从 gdy 数据集中按类别抽取验证集。

数据格式：
- gdy/<action>/<sample>.npy
- gdy/<action>/<sample>.<image_ext>

行为：
1. 在 gdy/val 下创建 6 个动作子目录。
2. 从 6 个动作中抽取指定数量样本，移动到 gdy/val/<action>/。
3. 非 spiral 样本由一个 .npy 和一张同 stem 图片组成。
4. spiral 缺少光流图时，在 val 中生成一张全零占位图。
5. 可选先将已有 val 样本回滚到训练目录，再重新划分。
"""

from __future__ import annotations

import argparse
import math
import random
import shutil
import struct
import zlib
from collections import defaultdict
from pathlib import Path

ACTIONS = ["ascend", "descend", "forward", "left_turn", "right_turn", "spiral"]
IMAGE_SUFFIXES = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
PLACEHOLDER_SUFFIX = ".png"
PLACEHOLDER_SIZE = (224, 224)
PLACEHOLDER_VALUE = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split 100 paired samples from gdy into gdy/val.")
    parser.add_argument("--data-root", type=Path, default=Path("gdy"), help="gdy 数据根目录")
    parser.add_argument("--val-size", type=int, default=100, help="验证集总样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument(
        "--copy",
        action="store_true",
        help="复制到 val，而不是从训练目录移动到 val",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只打印计划，不实际执行",
    )
    parser.add_argument(
        "--reset-val",
        action="store_true",
        help="先将已有 val 样本回滚到训练目录，再重新划分",
    )
    parser.add_argument(
        "--balanced",
        action="store_true",
        help="按类别均衡划分，要求 val_size 能被动作类别数整除",
    )
    return parser.parse_args()


def find_image_for_stem(action_dir: Path, stem: str) -> Path | None:
    for suffix in IMAGE_SUFFIXES:
        candidate = action_dir / f"{stem}{suffix}"
        if candidate.exists():
            return candidate
    return None


def collect_pairs(action: str, action_dir: Path) -> list[tuple[Path, Path | None]]:
    pairs: list[tuple[Path, Path | None]] = []
    for npy_path in sorted(action_dir.glob("*.npy")):
        image_path = find_image_for_stem(action_dir, npy_path.stem)
        if action == "spiral":
            pairs.append((npy_path, image_path))
        elif image_path is not None:
            pairs.append((npy_path, image_path))
    return pairs


def ensure_val_dirs(data_root: Path) -> None:
    val_root = data_root / "val"
    for action in ACTIONS:
        (val_root / action).mkdir(parents=True, exist_ok=True)


def allocate_counts(total: int, counts: dict[str, int]) -> dict[str, int]:
    total_available = sum(counts.values())
    if total > total_available:
        raise ValueError(f"验证集数量不足: requested={total}, available={total_available}")

    raw = {
        action: total * count / total_available
        for action, count in counts.items()
    }
    allocated = {
        action: min(counts[action], math.floor(raw[action]))
        for action in counts
    }
    remaining = total - sum(allocated.values())

    remainders = sorted(
        counts.keys(),
        key=lambda action: (raw[action] - allocated[action], counts[action]),
        reverse=True,
    )

    idx = 0
    while remaining > 0:
        action = remainders[idx % len(remainders)]
        if allocated[action] < counts[action]:
            allocated[action] += 1
            remaining -= 1
        idx += 1

    return allocated


def allocate_balanced_counts(total: int, counts: dict[str, int]) -> dict[str, int]:
    num_classes = len(counts)
    if total % num_classes != 0:
        raise ValueError(
            f"均衡划分要求 val_size 能被类别数整除: val_size={total}, classes={num_classes}"
        )

    per_class = total // num_classes
    insufficient = {
        action: count for action, count in counts.items() if count < per_class
    }
    if insufficient:
        raise ValueError(
            f"以下类别样本不足，无法均衡划分每类 {per_class} 条: {insufficient}"
        )

    return {action: per_class for action in counts}


def move_or_copy(src: Path, dst: Path, copy_mode: bool, dry_run: bool) -> None:
    if dry_run:
        return
    if copy_mode:
        shutil.copy2(src, dst)
    else:
        shutil.move(src, dst)


def write_placeholder_image(dst: Path, dry_run: bool) -> None:
    if dry_run:
        return
    width, height = PLACEHOLDER_SIZE
    pixel = bytes([PLACEHOLDER_VALUE, PLACEHOLDER_VALUE, PLACEHOLDER_VALUE])
    raw = b"".join(b"\x00" + pixel * width for _ in range(height))
    compressed = zlib.compress(raw)

    def chunk(chunk_type: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png_bytes = b"".join(
        [
            b"\x89PNG\r\n\x1a\n",
            chunk(b"IHDR", ihdr),
            chunk(b"IDAT", compressed),
            chunk(b"IEND", b""),
        ]
    )
    dst.write_bytes(png_bytes)


def reset_val_split(data_root: Path, dry_run: bool) -> None:
    val_root = data_root / "val"
    if not val_root.exists():
        return

    for action in ACTIONS:
        src_dir = val_root / action
        dst_dir = data_root / action
        if not src_dir.exists():
            continue

        for npy_path in sorted(src_dir.glob("*.npy")):
            dst_npy = dst_dir / npy_path.name
            if dst_npy.exists():
                raise FileExistsError(f"回滚冲突，目标已存在: {dst_npy}")
            print(f"[RESET] {action}: {npy_path.name} -> {dst_npy}")
            move_or_copy(npy_path, dst_npy, copy_mode=False, dry_run=dry_run)

        for image_path in sorted(src_dir.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            dst_img = dst_dir / image_path.name
            if action == "spiral":
                print(f"[RESET] {action}: remove placeholder {image_path.name}")
                if not dry_run:
                    image_path.unlink()
                continue

            if dst_img.exists():
                raise FileExistsError(f"回滚冲突，目标已存在: {dst_img}")
            print(f"[RESET] {action}: {image_path.name} -> {dst_img}")
            move_or_copy(image_path, dst_img, copy_mode=False, dry_run=dry_run)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    val_root = data_root / "val"

    ensure_val_dirs(data_root)
    if args.reset_val:
        reset_val_split(data_root, args.dry_run)

    action_pairs: dict[str, list[tuple[Path, Path | None]]] = {}
    available_counts: dict[str, int] = {}

    for action in ACTIONS:
        action_dir = data_root / action
        if not action_dir.exists():
            raise FileNotFoundError(f"动作目录不存在: {action_dir}")
        pairs = collect_pairs(action, action_dir)
        if not pairs:
            raise RuntimeError(f"动作目录中没有可用的样本对: {action_dir}")
        action_pairs[action] = pairs
        available_counts[action] = len(pairs)

    allocation = (
        allocate_balanced_counts(args.val_size, available_counts)
        if args.balanced
        else allocate_counts(args.val_size, available_counts)
    )

    rng = random.Random(args.seed)
    selected: dict[str, list[tuple[Path, Path | None]]] = defaultdict(list)
    for action in ACTIONS:
        pairs = action_pairs[action][:]
        rng.shuffle(pairs)
        selected[action] = sorted(pairs[: allocation[action]], key=lambda pair: pair[0].stem)

    print("=" * 60)
    print("gdy 验证集划分")
    print(f"data_root : {data_root}")
    print(f"val_root  : {val_root}")
    print(f"val_size  : {args.val_size}")
    print(f"seed      : {args.seed}")
    print(f"mode      : {'copy' if args.copy else 'move'}")
    print(f"dry_run   : {args.dry_run}")
    print(f"reset_val : {args.reset_val}")
    print(f"balanced  : {args.balanced}")
    print(f"spiral_placeholder : value={PLACEHOLDER_VALUE}, size={PLACEHOLDER_SIZE}")
    print("=" * 60)

    moved_total = 0
    for action in ACTIONS:
        target_dir = val_root / action
        selected_pairs = selected[action]
        for npy_path, image_path in selected_pairs:
            dst_npy = target_dir / npy_path.name
            dst_img_name = (
                image_path.name if image_path is not None else f"{npy_path.stem}{PLACEHOLDER_SUFFIX}"
            )
            dst_img = target_dir / dst_img_name
            if dst_npy.exists() or dst_img.exists():
                raise FileExistsError(f"目标文件已存在: {dst_npy} or {dst_img}")

            print(f"[SPLIT] {action}: {npy_path.name}, {dst_img.name}")
            move_or_copy(npy_path, dst_npy, args.copy, args.dry_run)
            if image_path is None:
                write_placeholder_image(dst_img, args.dry_run)
            else:
                move_or_copy(image_path, dst_img, args.copy, args.dry_run)
            moved_total += 1

        print(
            f"[DONE] {action}: selected={len(selected_pairs)}, "
            f"remaining={available_counts[action] - len(selected_pairs)}"
        )

    print("=" * 60)
    print(f"total_selected : {moved_total}")
    print("val 目录已就绪，包含 6 个动作子目录。")
    print("=" * 60)


if __name__ == "__main__":
    main()
