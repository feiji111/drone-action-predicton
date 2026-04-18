#!/usr/bin/env python3
"""
多模态数据集预处理脚本
每个样本保存：
  window_XXXXXX_imu.npy       —— (100, 6) float32
  window_XXXXXX_meta.json     —— {'label', 'flight', 'images': [4个jpg文件名]}
"""

import os, json, argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


def wrap_angle_rad(values: np.ndarray) -> np.ndarray:
    """Wrap angles to [-pi, pi]."""
    return (values + np.pi) % (2 * np.pi) - np.pi


# ─── 标注规则：速度 + IMU + 姿态联合判断 ──────────────────────────────
def label_window(window: pd.DataFrame) -> str | None:
    vx = window['drone_velocity_linear_x'].to_numpy()
    vy = window['drone_velocity_linear_y'].to_numpy()
    vz = window['drone_velocity_linear_z'].to_numpy()
    gz = window['gyro_z'].to_numpy()
    roll = window['drone_roll'].to_numpy()
    pitch = window['drone_pitch'].to_numpy()
    yaw = wrap_angle_rad(window['drone_yaw'].to_numpy())

    vxy = np.sqrt(vx**2 + vy**2)
    speed = np.sqrt(vx**2 + vy**2 + vz**2)

    vz_mean = float(vz.mean())
    vxy_mean = float(vxy.mean())
    speed_mean = float(speed.mean())
    gz_mean_abs = float(np.abs(gz).mean())
    roll_mean_abs = float(np.abs(roll).mean())
    pitch_mean_abs = float(np.abs(pitch).mean())
    yaw_delta = float(np.abs(np.unwrap(yaw)[-1] - np.unwrap(yaw)[0]))

    turn_ratio = float((np.abs(gz) > 1.0).mean())
    ascend_ratio = float((vz > 0.35).mean())
    descend_ratio = float((vz < -0.35).mean())
    forward_ratio = float((vxy > 1.2).mean())
    still_ratio = float(((np.abs(vz) < 0.15) & (vxy < 0.5) & (np.abs(gz) < 0.3)).mean())

    # 真正的悬停/低动态状态：速度、角速度和姿态变化都很小。
    if (
        still_ratio > 0.8
        and speed_mean < 0.6
        and gz_mean_abs < 0.35
        and roll_mean_abs < 0.12
        and pitch_mean_abs < 0.12
        and yaw_delta < 0.25
    ):
        return 'hover_idle'

    # 转弯：偏航角速度占主导，或者窗口内航向累计变化明显。
    if turn_ratio > 0.45 and (gz_mean_abs > 1.0 or yaw_delta > 0.5):
        return 'turn'

    # 垂直运动：要求持续向上/向下，避免只凭瞬时峰值。
    if ascend_ratio > 0.55 and vz_mean > 0.35 and gz_mean_abs < 1.2:
        return 'ascend'

    if descend_ratio > 0.55 and vz_mean < -0.35 and gz_mean_abs < 1.2:
        return 'descend'

    # 前飞：水平速度占主导，同时机体通常有一定前倾。
    if (
        forward_ratio > 0.55
        and vxy_mean > 1.2
        and abs(vz_mean) < 0.45
        and pitch_mean_abs > 0.08
        and gz_mean_abs < 1.0
    ):
        return 'forward'

    # 边界样本仍需归入 5 类时，优先按主导运动模式分配，而不是全部落到 hover_idle。
    scores = {
        'turn': 0.60 * turn_ratio + 0.25 * min(gz_mean_abs / 1.5, 1.5) + 0.15 * min(yaw_delta / 0.8, 1.5),
        'ascend': 0.70 * ascend_ratio + 0.30 * max(vz_mean, 0.0),
        'descend': 0.70 * descend_ratio + 0.30 * max(-vz_mean, 0.0),
        'forward': 0.65 * forward_ratio + 0.20 * min(vxy_mean / 2.5, 1.5) + 0.15 * min(pitch_mean_abs / 0.2, 1.5),
        'hover_idle': 0.70 * still_ratio + 0.30 * max(0.0, 1.0 - speed_mean),
    }
    best_label = max(scores, key=scores.get)
    best_score = scores[best_label]

    # 过于模糊的窗口直接跳过，减少脏标签。
    if best_score < 0.45:
        return None
    return best_label


# ─── 从100行里均匀选4张不重复图帧 ────────────────────────────────────
def pick_frames(window: pd.DataFrame, n_frames: int = 4) -> list[str]:
    img_col  = window['img_filename'].values
    # 均匀取4个时间点
    indices  = np.linspace(0, len(img_col) - 1, n_frames, dtype=int)
    selected = [img_col[i] for i in indices]
    # 去重（相邻帧可能相同），不足补最后一帧
    dedup = []
    seen  = set()
    for name in selected:
        if name not in seen:
            dedup.append(name)
            seen.add(name)
    while len(dedup) < n_frames:
        dedup.append(dedup[-1])
    return dedup[:n_frames]


IMU_COLS = ['accel_x','accel_y','accel_z','gyro_x','gyro_y','gyro_z']
ALL_COLS  = IMU_COLS + [
    'drone_velocity_linear_x','drone_velocity_linear_y','drone_velocity_linear_z',
    'drone_roll', 'drone_pitch', 'drone_yaw',
    'gyro_z', 'img_filename'
]


def process_flight(csv_path: Path, window_size: int, step: int,
                   img_root: Path) -> list[dict]:
    """处理单个flight，返回样本列表。"""
    flight_name = csv_path.parent.name
    flight_dir   = img_root / flight_name
    img_dir      = flight_dir / 'images' / f'camera_{flight_name}'
    fallback_dir = flight_dir / f'camera_{flight_name}'

    if not img_dir.exists() and fallback_dir.exists():
        img_dir = fallback_dir

    if not img_dir.exists():
        print(f"  [SKIP] images not found: {img_dir}")
        return []

    df = pd.read_csv(csv_path, usecols=ALL_COLS + ['timestamp'])
    samples = []

    for start in range(0, len(df) - window_size, step):
        window = df.iloc[start : start + window_size]
        if len(window) < window_size:
            continue

        label  = label_window(window)
        if label is None:
            continue
        images = pick_frames(window, n_frames=4)

        # 确认图片实际存在
        valid_imgs = [img for img in images if (img_dir / img).exists()]
        if len(valid_imgs) < 4:
            continue   # 缺帧跳过

        imu_data = window[IMU_COLS].values.astype(np.float32)  # (100, 6)

        samples.append({
            'imu':    imu_data,
            'label':  label,
            'flight': flight_name,
            'images': valid_imgs[:4],
        })

    return samples


def save_split(samples: list[dict], split_dir: Path):
    split_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(tqdm(samples, desc=f'  saving {split_dir.name}', leave=False)):
        prefix = split_dir / f'window_{i:06d}'
        np.save(str(prefix) + '_imu.npy', s['imu'])
        meta = {'label': s['label'], 'flight': s['flight'], 'images': s['images']}
        with open(str(prefix) + '_meta.json', 'w') as f:
            json.dump(meta, f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--autonomous_dir',
        default='autonomous',
        help='drone-racing-dataset/autonomous目录')
    parser.add_argument('--output_dir',
        default='data/racing_multimodal',
        help='输出数据集目录（在项目根下）')
    parser.add_argument('--window_size', type=int, default=100)
    parser.add_argument('--overlap',     type=float, default=0.5)
    args = parser.parse_args()

    autonomous_dir = Path(args.autonomous_dir)
    output_dir     = Path(args.output_dir)
    step           = int(args.window_size * (1 - args.overlap))

    print('='*60)
    print('多模态数据集构建 (IMU + 图像帧路径)')
    print(f'  autonomous_dir : {autonomous_dir.resolve()}')
    print(f'  output_dir     : {output_dir.resolve()}')
    print(f'  window={args.window_size}, overlap={args.overlap}, step={step}')
    print('='*60)

    # 找所有flight的CSV
    csv_files = sorted(autonomous_dir.glob('*/flight-*_500hz_freq_sync.csv'))
    print(f'\n找到 {len(csv_files)} 个flight CSV')

    all_samples: list[dict] = []
    for csv_path in csv_files:
        print(f'处理 {csv_path.parent.name} ...')
        samples = process_flight(csv_path, args.window_size, step,
                                 img_root=autonomous_dir)
        print(f'  -> {len(samples)} 个有效窗口')
        all_samples.extend(samples)

    print(f'\n总样本数: {len(all_samples)}')

    # 标签分布
    labels = [s['label'] for s in all_samples]
    from collections import Counter
    dist = Counter(labels)
    print('标签分布:')
    for k, v in sorted(dist.items()):
        print(f'  {k:12s}: {v:4d} ({v/len(labels)*100:.1f}%)')

    # 分层划分 train/val/test = 70/15/15
    indices = list(range(len(all_samples)))
    train_idx, tmp_idx = train_test_split(
        indices, test_size=0.30, random_state=42, stratify=labels)
    tmp_labels = [labels[i] for i in tmp_idx]
    val_idx, test_idx = train_test_split(
        tmp_idx, test_size=0.50, random_state=42, stratify=tmp_labels)

    print(f'\n划分: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}')

    save_split([all_samples[i] for i in train_idx], output_dir / 'train')
    save_split([all_samples[i] for i in val_idx],   output_dir / 'val')
    save_split([all_samples[i] for i in test_idx],  output_dir / 'test')

    # 写 dataset_info.json
    info = {
        'total': len(all_samples),
        'window_size': args.window_size,
        'overlap': args.overlap,
        'imu_channels': 6,
        'frames_per_window': 4,
        'image_resolution': '640x480',
        'classes': sorted(dist.keys()),
        'num_classes': len(dist),
        'splits': {'train': len(train_idx), 'val': len(val_idx), 'test': len(test_idx)},
    }
    with open(output_dir / 'dataset_info.json', 'w') as f:
        json.dump(info, f, indent=2)

    print('\n✅ 完成！')
    print(f'   输出目录: {output_dir.resolve()}')


if __name__ == '__main__':
    main()
