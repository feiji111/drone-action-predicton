from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uav_state_recognition.prepare import add_euler_columns, build_window_specs
from uav_state_recognition.utils import ensure_dir, write_csv


CSV_SUFFIX = "_cam_ts_sync.csv"


def assign_split(index: int, total: int) -> str:
    ratio = index / max(total, 1)
    if ratio < 0.7:
        return "train"
    if ratio < 0.85:
        return "val"
    return "test"


def collect_flights(input_root: Path) -> list[Path]:
    return sorted(path for path in input_root.rglob(f"*{CSV_SUFFIX}") if path.is_file())


def sanitize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        clean = column.strip().lower()
        clean = clean.replace("[", "_").replace("]", "")
        clean = clean.replace("/", "_").replace(" ", "_")
        renamed[column] = clean
    return frame.rename(columns=renamed)


def make_window_rows(root_out: Path, split: str, csv_path: Path) -> list[dict[str, object]]:
    frame = sanitize_columns(pd.read_csv(csv_path))
    rename_map = {
        "drone_roll": "roll",
        "drone_pitch": "pitch",
        "drone_yaw": "yaw",
        "drone_velocity_linear_x": "velocity_x",
        "drone_velocity_linear_y": "velocity_y",
        "drone_velocity_linear_z": "velocity_z",
        "drone_velocity_angular_z": "yaw_rate",
        "channels_thrust": "thrust",
    }
    frame = frame.rename(columns={k: v for k, v in rename_map.items() if k in frame.columns})

    if "yaw_rate" not in frame.columns and "gyro_z" in frame.columns:
        frame["yaw_rate"] = frame["gyro_z"]

    sample_rate = 1.0 / max(frame["elapsed_time"].diff().dropna().median(), 1e-3)
    specs = build_window_specs(
        frame=frame,
        sample_rate=float(sample_rate),
        yaw_rate_col="yaw_rate",
    )

    telemetry_dir = ensure_dir(root_out / "telemetry")
    telemetry_name = f"{csv_path.stem}.csv"
    frame.to_csv(telemetry_dir / telemetry_name, index=False)

    image_root = csv_path.parent / csv_path.stem.replace("_cam_ts_sync", "").replace("flight-", "camera_flight-")
    if not image_root.exists():
        image_root = csv_path.parent / f"camera_{csv_path.stem.replace('_cam_ts_sync', '')}"

    rows: list[dict[str, object]] = []
    for spec in specs:
        chunk = frame.iloc[spec.start_idx : spec.end_idx]
        frame_names = [str(Path(split) / "frames" / csv_path.stem / name) for name in chunk["img_filename"].dropna().astype(str).tolist()]
        rows.append(
            {
                "flight_id": csv_path.stem,
                "telemetry_path": str(Path(split) / "telemetry" / telemetry_name),
                "start_idx": spec.start_idx,
                "end_idx": spec.end_idx,
                "label": spec.label,
                "label_name": spec.label_name,
                "frame_paths": str(frame_names),
            }
        )
    return rows


def copy_frames(csv_path: Path, split_root: Path) -> None:
    frame_dir_name = csv_path.stem.replace("_cam_ts_sync", "")
    candidate_dirs = [
        csv_path.parent / f"camera_{frame_dir_name}",
        csv_path.parent / frame_dir_name.replace("flight-", "camera_flight-"),
        csv_path.parent / frame_dir_name,
    ]
    src_dir = next((path for path in candidate_dirs if path.exists()), None)
    if src_dir is None:
        raise FileNotFoundError(f"Could not find frame directory for {csv_path}")
    dst_dir = ensure_dir(split_root / "frames" / csv_path.stem)
    for image_path in sorted(src_dir.glob("*.jpg")):
        target = dst_dir / image_path.name
        if not target.exists():
            target.symlink_to(image_path.resolve())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = ensure_dir(args.output)
    flights = collect_flights(input_root)
    all_rows: dict[str, list[dict[str, object]]] = {"train": [], "val": [], "test": []}

    for index, csv_path in enumerate(flights):
        split = assign_split(index, len(flights))
        split_root = ensure_dir(output_root / split)
        copy_frames(csv_path, split_root)
        rows = make_window_rows(output_root / split, split, csv_path)
        all_rows[split].extend(rows)

    for split, rows in all_rows.items():
        split_root = ensure_dir(output_root / split)
        write_csv(split_root / "windows.csv", rows, fieldnames=list(rows[0].keys()) if rows else ["flight_id"])

    print(f"Prepared drone racing dataset at {output_root}")


if __name__ == "__main__":
    main()
