#!/usr/bin/env python3
"""Prepare TII Racing Dataset for UAV motion recognition."""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import json
from tqdm import tqdm

def parse_flight_data(csv_path):
    """Parse flight CSV and extract IMU + motion data."""
    df = pd.read_csv(csv_path)

    # Select key columns
    cols = ['timestamp', 'accel_x', 'accel_y', 'accel_z',
            'gyro_x', 'gyro_y', 'gyro_z',
            'drone_velocity_linear_x', 'drone_velocity_linear_y', 'drone_velocity_linear_z',
            'drone_roll', 'drone_pitch', 'drone_yaw']

    return df[cols].copy()

def label_motion_from_data(window_df):
    """Label motion state from IMU and velocity data."""
    # Compute statistics
    vz = window_df['drone_velocity_linear_z'].mean()
    vxy = np.sqrt(window_df['drone_velocity_linear_x']**2 +
                  window_df['drone_velocity_linear_y']**2).mean()
    gyro_z = abs(window_df['gyro_z'].mean())

    # Motion classification
    if gyro_z > 1.5:  # rad/s
        return 'turn'
    elif vz > 0.5:  # m/s upward
        return 'ascend'
    elif vz < -0.5:  # m/s downward
        return 'descend'
    elif vxy > 2.0:  # m/s forward
        return 'forward'
    else:
        return 'hover_idle'

def create_windows(df, window_size=100, overlap=0.5):
    """Create sliding windows."""
    step = int(window_size * (1 - overlap))
    windows = []

    for i in range(0, len(df) - window_size, step):
        window = df.iloc[i:i+window_size].copy()
        if len(window) == window_size:
            windows.append(window)

    return windows

def main():
    print("=" * 60)
    print("Preparing TII Racing Dataset")
    print("=" * 60)

    # Paths
    data_dir = Path("autonomous")
    out_dir = Path("data/racing_prepared")
    out_dir.mkdir(parents=True, exist_ok=True)

    if not data_dir.exists():
        print(f"Error: {data_dir} not found")
        return

    # Find all flights
    flights = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    print(f"\nFound {len(flights)} flights")

    all_windows = []
    all_labels = []

    for flight_dir in tqdm(flights, desc="Processing flights"):
        csv_file = flight_dir / f"{flight_dir.name}_500hz_freq_sync.csv"

        if not csv_file.exists():
            continue

        try:
            df = parse_flight_data(csv_file)
            windows = create_windows(df, window_size=100, overlap=0.5)

            for window in windows:
                label = label_motion_from_data(window)
                all_windows.append(window)
                all_labels.append(label)
        except Exception as e:
            print(f"\nError processing {flight_dir.name}: {e}")
            continue

    print(f"\nTotal windows: {len(all_windows)}")

    # Label distribution
    unique, counts = np.unique(all_labels, return_counts=True)
    print("\nLabel distribution:")
    for label, count in zip(unique, counts):
        print(f"  {label}: {count} ({count/len(all_labels)*100:.1f}%)")

    # Split data
    indices = np.arange(len(all_windows))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=all_labels)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42,
                                         stratify=[all_labels[i] for i in temp_idx])

    # Save splits
    for split_name, split_idx in [('train', train_idx), ('val', val_idx), ('test', test_idx)]:
        split_dir = out_dir / split_name
        split_dir.mkdir(exist_ok=True)

        split_data = []
        for i, idx in enumerate(tqdm(split_idx, desc=f"Saving {split_name}")):
            window = all_windows[idx]
            label = all_labels[idx]

            # Save telemetry (only IMU columns)
            imu_cols = ['accel_x', 'accel_y', 'accel_z', 'gyro_x', 'gyro_y', 'gyro_z']
            telem_file = split_dir / f"window_{i:06d}_telemetry.csv"
            window[imu_cols].to_csv(telem_file, index=False)

            split_data.append({
                'window_id': i,
                'telemetry_file': telem_file.name,
                'label': label
            })

        # Save metadata
        meta_df = pd.DataFrame(split_data)
        meta_df.to_csv(split_dir / "labels.csv", index=False)

        print(f"\n{split_name}: {len(split_idx)} windows")
        label_dist = meta_df['label'].value_counts()
        for label, count in label_dist.items():
            print(f"  {label}: {count}")

    # Save dataset info
    info = {
        'total_windows': len(all_windows),
        'window_size': 100,
        'overlap': 0.5,
        'imu_channels': 6,
        'classes': sorted(list(set(all_labels))),
        'num_classes': len(set(all_labels)),
        'splits': {
            'train': len(train_idx),
            'val': len(val_idx),
            'test': len(test_idx)
        }
    }

    with open(out_dir / "dataset_info.json", 'w') as f:
        json.dump(info, f, indent=2)

    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print(f"Output: {out_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()
