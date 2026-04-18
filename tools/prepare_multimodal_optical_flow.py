#!/usr/bin/env python3
"""
基于原始 500Hz 同步数据构建“传感器 + 单张光流图”的多模态数据集。

增强点：
1. 多尺度窗口，不再只依赖单一 1 秒长度。
2. 每个窗口可生成多张光流图，丰富视觉模式。
3. 更高 overlap，扩大样本量。
4. 光流采用候选帧对搜索，而不是固定只算一组。
"""

from __future__ import annotations

import argparse
import itertools
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "This script requires OpenCV. Install 'opencv-python-headless' first."
    ) from exc


LABELS = [
    "ascend",
    "descend",
    "left_turn",
    "right_turn",
    "forward",
    "spiral",
]

LABEL_DISPLAY = {
    "ascend": "上升",
    "descend": "下降",
    "left_turn": "左转弯",
    "right_turn": "右转弯",
    "forward": "平飞",
    "spiral": "盘旋",
}

SENSOR_COLS = [
    "accel_x",
    "accel_y",
    "accel_z",
    "gyro_x",
    "gyro_y",
    "gyro_z",
    "drone_velocity_linear_x",
    "drone_velocity_linear_y",
    "drone_velocity_linear_z",
    "drone_roll",
    "drone_pitch",
    "drone_yaw",
]

RULE_COLS = SENSOR_COLS + ["timestamp", "img_filename"]


def wrap_angle_rad(values: np.ndarray) -> np.ndarray:
    return (values + np.pi) % (2 * np.pi) - np.pi


def label_window(window: pd.DataFrame) -> tuple[str | None, dict[str, float]]:
    vx = window["drone_velocity_linear_x"].to_numpy(dtype=np.float32)
    vy = window["drone_velocity_linear_y"].to_numpy(dtype=np.float32)
    vz = window["drone_velocity_linear_z"].to_numpy(dtype=np.float32)
    gz = window["gyro_z"].to_numpy(dtype=np.float32)
    roll = window["drone_roll"].to_numpy(dtype=np.float32)
    pitch = window["drone_pitch"].to_numpy(dtype=np.float32)
    yaw = wrap_angle_rad(window["drone_yaw"].to_numpy(dtype=np.float32))

    vxy = np.sqrt(vx**2 + vy**2)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)
    yaw_unwrapped = np.unwrap(yaw)
    yaw_delta = float(yaw_unwrapped[-1] - yaw_unwrapped[0])

    stats = {
        "vz_mean": float(vz.mean()),
        "vz_abs_mean": float(np.abs(vz).mean()),
        "vxy_mean": float(vxy.mean()),
        "speed_mean": float(speed.mean()),
        "gz_mean": float(gz.mean()),
        "gz_abs_mean": float(np.abs(gz).mean()),
        "yaw_delta": yaw_delta,
        "pitch_abs_mean": float(np.abs(pitch).mean()),
        "roll_abs_mean": float(np.abs(roll).mean()),
        "ascend_ratio": float((vz > 0.35).mean()),
        "descend_ratio": float((vz < -0.35).mean()),
        "left_turn_ratio": float((gz > 0.75).mean()),
        "right_turn_ratio": float((gz < -0.75).mean()),
        "forward_ratio": float((vxy > 1.0).mean()),
        "still_ratio": float(
            ((np.abs(vz) < 0.15) & (vxy < 0.45) & (np.abs(gz) < 0.3)).mean()
        ),
    }

    if (
        stats["ascend_ratio"] > 0.58
        and stats["vz_mean"] > 0.30
        and stats["gz_abs_mean"] < 1.3
    ):
        return "ascend", stats

    if (
        stats["descend_ratio"] > 0.58
        and stats["vz_mean"] < -0.30
        and stats["gz_abs_mean"] < 1.3
    ):
        return "descend", stats

    if (
        stats["left_turn_ratio"] > 0.42
        and stats["gz_mean"] > 0.70
        and stats["yaw_delta"] > 0.35
    ):
        return "left_turn", stats

    if (
        stats["right_turn_ratio"] > 0.42
        and stats["gz_mean"] < -0.70
        and stats["yaw_delta"] < -0.35
    ):
        return "right_turn", stats

    if (
        stats["forward_ratio"] > 0.50
        and stats["vxy_mean"] > 0.95
        and abs(stats["vz_mean"]) < 0.50
        and stats["gz_abs_mean"] < 1.0
    ):
        return "forward", stats

    return None, stats


def load_rgb_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = ((angle * 180) / np.pi / 2).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def frame_index_from_name(name: str) -> int:
    stem = Path(name).stem
    prefix = stem.split("_", 1)[0]
    digits = "".join(ch for ch in prefix if ch.isdigit())
    return int(digits) if digits else 0


def compute_flow_image(first_image: Path, last_image: Path) -> tuple[np.ndarray, float]:
    first_rgb = load_rgb_image(first_image)
    last_rgb = load_rgb_image(last_image)
    first_gray = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY)
    last_gray = cv2.cvtColor(last_rgb, cv2.COLOR_RGB2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        first_gray,
        last_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=21,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    magnitude = cv2.cartToPolar(flow[..., 0], flow[..., 1])[0]
    image_delta = cv2.absdiff(first_gray, last_gray)
    flow_score = float(
        magnitude.mean()
        + 0.25 * magnitude.max()
        + 0.20 * image_delta.mean()
        + 0.10 * image_delta.std()
    )
    return flow_to_rgb(flow), flow_score


def resolve_img_dir(autonomous_dir: Path, flight_name: str) -> Path | None:
    flight_dir = autonomous_dir / flight_name
    img_dir = flight_dir / "images" / f"camera_{flight_name}"
    if img_dir.exists():
        return img_dir
    fallback_dir = flight_dir / f"camera_{flight_name}"
    if fallback_dir.exists():
        return fallback_dir
    return None


def select_flow_pairs(
    image_names: list[str],
    top_k: int,
    min_gap_frames: int,
) -> list[tuple[str, str]]:
    indexed = [(frame_index_from_name(name), name) for name in image_names]
    candidates: list[tuple[int, str, str]] = []
    for (left_idx, left_name), (right_idx, right_name) in itertools.combinations(
        indexed, 2
    ):
        gap = abs(right_idx - left_idx)
        if gap >= min_gap_frames:
            candidates.append((gap, left_name, right_name))

    if not candidates and len(image_names) >= 2:
        return [(image_names[0], image_names[-1])]

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for _, left_name, right_name in candidates:
        pair = (left_name, right_name)
        if pair in seen:
            continue
        selected.append(pair)
        seen.add(pair)
        if len(selected) >= top_k:
            break
    return selected


def process_window(
    window: pd.DataFrame,
    img_dir: Path,
    label: str,
    stats: dict[str, float],
    sensor_cols: list[str],
    top_flow_variants: int,
    min_gap_frames: int,
    min_flow_score: float,
    window_tag: str,
) -> list[dict]:
    image_names = [str(name) for name in window["img_filename"].tolist()]
    pair_candidates = select_flow_pairs(
        image_names=image_names,
        top_k=top_flow_variants,
        min_gap_frames=min_gap_frames,
    )

    sensor = window[sensor_cols].to_numpy(dtype=np.float32)
    variants: list[dict] = []
    for variant_id, (first_img_name, last_img_name) in enumerate(pair_candidates):
        first_img_path = img_dir / first_img_name
        last_img_path = img_dir / last_img_name
        if not first_img_path.exists() or not last_img_path.exists():
            continue

        flow_rgb, flow_score = compute_flow_image(first_img_path, last_img_path)
        if flow_score < min_flow_score:
            continue

        variants.append(
            {
                "sensor": sensor,
                "flow_rgb": flow_rgb,
                "label": label,
                "first_image": first_img_name,
                "last_image": last_img_name,
                "flow_score": flow_score,
                "flow_variant_id": variant_id,
                "window_tag": window_tag,
                "stats": stats,
            }
        )
    return variants


def process_flight(
    csv_path: Path,
    autonomous_dir: Path,
    window_sizes: list[int],
    overlap: float,
    min_flow_score: float,
    top_flow_variants: int,
    min_gap_frames: int,
) -> list[dict]:
    flight_name = csv_path.parent.name
    img_dir = resolve_img_dir(autonomous_dir, flight_name)
    if img_dir is None:
        print(f"  [SKIP] images not found for {flight_name}")
        return []

    df = pd.read_csv(csv_path, usecols=RULE_COLS)
    samples: list[dict] = []

    for window_size in window_sizes:
        step = max(1, int(round(window_size * (1.0 - overlap))))
        for start in range(0, len(df) - window_size + 1, step):
            window = df.iloc[start : start + window_size]
            if len(window) < window_size:
                continue

            label, stats = label_window(window)
            if label is None:
                continue

            window_tag = (
                f"{flight_name}_{start}_{start + window_size - 1}_{window_size}"
            )
            variants = process_window(
                window=window,
                img_dir=img_dir,
                label=label,
                stats=stats,
                sensor_cols=SENSOR_COLS,
                top_flow_variants=top_flow_variants,
                min_gap_frames=min_gap_frames,
                min_flow_score=min_flow_score,
                window_tag=window_tag,
            )
            for variant in variants:
                variant["flight"] = flight_name
                variant["window_start_index"] = int(start)
                variant["window_end_index"] = int(start + window_size - 1)
                variant["window_size"] = int(window_size)
            samples.extend(variants)

    return samples


def save_by_label(samples: list[dict], output_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    grouped: dict[str, list[dict]] = {}
    for sample in samples:
        grouped.setdefault(sample["label"], []).append(sample)

    for label, records in grouped.items():
        label_dir = output_dir / label
        label_dir.mkdir(parents=True, exist_ok=True)
        counts[label] = len(records)
        for i, sample in enumerate(
            tqdm(records, desc=f"  saving {label}", leave=False)
        ):
            prefix = label_dir / f"window_{i:06d}"
            np.save(f"{prefix}_imu.npy", sample["sensor"])
            Image.fromarray(sample["flow_rgb"]).save(f"{prefix}_flow.png")
            meta = {
                "label": sample["label"],
                "flight": sample["flight"],
                "flow_image": f"{prefix.name}_flow.png",
                "pair_type": "real_optical_flow",
                "source_images": [sample["first_image"], sample["last_image"]],
                "flow_score": sample["flow_score"],
                "flow_variant_id": sample["flow_variant_id"],
                "window_tag": sample["window_tag"],
                "window_start_index": sample["window_start_index"],
                "window_end_index": sample["window_end_index"],
                "window_size": sample["window_size"],
                "label_stats": sample["stats"],
            }
            with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
                json.dump(meta, f, ensure_ascii=False, indent=2)
    return counts


def save_empty_labels(output_dir: Path, labels: list[str]) -> None:
    for label in labels:
        (output_dir / label).mkdir(parents=True, exist_ok=True)


def save_split(samples: list[dict], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    for i, sample in enumerate(
        tqdm(samples, desc=f"  saving {split_dir.name}", leave=False)
    ):
        prefix = split_dir / f"window_{i:06d}"
        np.save(f"{prefix}_imu.npy", sample["sensor"])
        Image.fromarray(sample["flow_rgb"]).save(f"{prefix}_flow.png")
        meta = {
            "label": sample["label"],
            "flight": sample["flight"],
            "flow_image": f"{prefix.name}_flow.png",
            "pair_type": "real_optical_flow",
            "source_images": [sample["first_image"], sample["last_image"]],
            "flow_score": sample["flow_score"],
            "flow_variant_id": sample["flow_variant_id"],
            "window_tag": sample["window_tag"],
            "window_start_index": sample["window_start_index"],
            "window_end_index": sample["window_end_index"],
            "window_size": sample["window_size"],
            "label_stats": sample["stats"],
        }
        with open(f"{prefix}_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--autonomous_dir",
        default="data/raw/autonomous",
        help="原始 autonomous 数据目录",
    )
    parser.add_argument(
        "--output_dir",
        default="data/racing_multimodal_flow",
        help="输出数据集目录",
    )
    parser.add_argument(
        "--window_seconds",
        default="0.8,1.0,1.2",
        help="多个滑窗时长，秒，逗号分隔",
    )
    parser.add_argument(
        "--sample_rate_hz",
        type=int,
        default=500,
        help="同步数据采样率",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.75,
        help="滑窗重叠率",
    )
    parser.add_argument(
        "--top_flow_variants",
        type=int,
        default=3,
        help="每个窗口最多保留多少张光流图",
    )
    parser.add_argument(
        "--min_gap_frames",
        type=int,
        default=150,
        help="构造光流时，候选帧对的最小帧间隔",
    )
    parser.add_argument(
        "--min_flow_score",
        type=float,
        default=2.0,
        help="过滤过弱光流窗口",
    )
    args = parser.parse_args()

    autonomous_dir = Path(args.autonomous_dir)
    output_dir = Path(args.output_dir)
    window_seconds = [
        float(item.strip()) for item in args.window_seconds.split(",") if item.strip()
    ]
    window_sizes = sorted(
        {
            max(2, int(round(seconds * args.sample_rate_hz)))
            for seconds in window_seconds
        }
    )

    print("=" * 60)
    print("多模态数据集构建 (传感器 + 多尺度多光流)")
    print(f"  autonomous_dir     : {autonomous_dir.resolve()}")
    print(f"  output_dir         : {output_dir.resolve()}")
    print(f"  window_seconds     : {window_seconds}")
    print(f"  window_sizes       : {window_sizes}")
    print(f"  sample_rate_hz     : {args.sample_rate_hz}")
    print(f"  overlap            : {args.overlap}")
    print(f"  top_flow_variants  : {args.top_flow_variants}")
    print(f"  min_gap_frames     : {args.min_gap_frames}")
    print(f"  min_flow_score     : {args.min_flow_score}")
    print("=" * 60)

    csv_files = sorted(autonomous_dir.glob("*/flight-*_500hz_freq_sync.csv"))
    print(f"\n找到 {len(csv_files)} 个 flight CSV")

    all_samples: list[dict] = []
    for csv_path in csv_files:
        print(f"处理 {csv_path.parent.name} ...")
        samples = process_flight(
            csv_path=csv_path,
            autonomous_dir=autonomous_dir,
            window_sizes=window_sizes,
            overlap=args.overlap,
            min_flow_score=args.min_flow_score,
            top_flow_variants=args.top_flow_variants,
            min_gap_frames=args.min_gap_frames,
        )
        print(f"  -> {len(samples)} 个有效样本")
        all_samples.extend(samples)

    if not all_samples:
        raise SystemExit("没有生成任何有效样本，请调低阈值或检查原始数据。")

    labels = [sample["label"] for sample in all_samples]
    dist = Counter(labels)
    print(f"\n总样本数: {len(all_samples)}")
    print("标签分布:")
    for label in LABELS:
        count = dist.get(label, 0)
        ratio = count / len(all_samples) * 100.0
        print(f"  {label:12s} ({LABEL_DISPLAY[label]}): {count:6d} ({ratio:5.1f}%)")

    label_counts = save_by_label(all_samples, output_dir)
    save_empty_labels(
        output_dir,
        [label for label in LABELS if label not in label_counts],
    )

    print("\n按类别写入完成:")
    for label in LABELS:
        print(f"  {label:12s}: {label_counts.get(label, 0)}")

    info = {
        "total": len(all_samples),
        "window_seconds": window_seconds,
        "sample_rate_hz": args.sample_rate_hz,
        "window_sizes": window_sizes,
        "overlap": args.overlap,
        "sensor_channels": len(SENSOR_COLS),
        "sensor_columns": SENSOR_COLS,
        "visual_mode": "optical_flow_multi_variant",
        "top_flow_variants": args.top_flow_variants,
        "min_gap_frames": args.min_gap_frames,
        "min_flow_score": args.min_flow_score,
        "classes_present": sorted(dist.keys()),
        "classes_defined": LABELS,
        "class_display_names": LABEL_DISPLAY,
        "empty_classes": [label for label in LABELS if dist.get(label, 0) == 0],
        "num_classes_present": len(dist),
        "layout": "single_root_with_label_subdirs",
        "counts_by_label": {label: label_counts.get(label, 0) for label in LABELS},
        "notes": [
            "spiral is intentionally left empty for this raw rotary-wing dataset",
            "multiple flow variants are generated per motion window",
            "samples are stored under output_dir/<label>/",
        ],
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)

    print("\n完成")
    print(f"输出目录: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
