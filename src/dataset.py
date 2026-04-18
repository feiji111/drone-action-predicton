from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms


IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
TRAIN_ACTIONS = ("ascend", "descend", "forward", "left_turn", "right_turn", "spiral")


class GdyDataset(Dataset):
    def __init__(
        self,
        data_root: str | Path,
        split: str = "train",
        transform: transforms.Compose | None = None,
    ) -> None:
        self.data_root = Path(data_root)
        self.split = split
        self.split_root = self._resolve_split_root(self.data_root, split)

        self.label_to_idx = {
            "ascend": 0,
            "descend": 1,
            "forward": 2,
            "left_turn": 3,
            "right_turn": 4,
            "spiral": 5,
        }
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

        if transform is None:
            self.transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225],
                    ),
                ]
            )
        else:
            self.transform = transform

        self.samples = self._collect_samples()
        if not self.samples:
            raise RuntimeError(f"No valid samples found under {self.split_root}")

    @staticmethod
    def _resolve_split_root(data_root: Path, split: str) -> Path:
        if split == "train":
            return data_root
        if split == "val":
            return data_root / "val"
        raise ValueError(f"Unsupported split: {split}")

    @staticmethod
    def _find_image_for_stem(action_dir: Path, stem: str) -> Path | None:
        for suffix in IMAGE_SUFFIXES:
            image_path = action_dir / f"{stem}{suffix}"
            if image_path.exists():
                return image_path
        return None

    def _collect_samples(self) -> list[dict[str, object]]:
        samples: list[dict[str, object]] = []
        for action in TRAIN_ACTIONS:
            action_dir = self.split_root / action
            if not action_dir.exists():
                raise FileNotFoundError(f"Missing action directory: {action_dir}")

            for npy_path in sorted(action_dir.glob("*.npy")):
                image_path = self._find_image_for_stem(action_dir, npy_path.stem)
                if image_path is None:
                    continue

                samples.append(
                    {
                        "imu_path": npy_path,
                        "image_path": image_path,
                        "label": self.label_to_idx[action],
                        "action": action,
                    }
                )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        sample = self.samples[idx]
        imu_path = sample["imu_path"]
        image_path = sample["image_path"]
        label = sample["label"]

        imu = np.load(imu_path).astype(np.float32)
        if imu.ndim != 2:
            raise ValueError(f"Expected 2D IMU array, got {imu.shape} from {imu_path}")

        imu_tensor = torch.from_numpy(imu)

        image = Image.open(image_path).convert("RGB")
        frame_tensor = self.transform(image)

        return imu_tensor, frame_tensor, int(label)


def collate_fn(
    batch: list[tuple[torch.Tensor, torch.Tensor, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    imu_batch = torch.stack([item[0] for item in batch], dim=0)
    frame_batch = torch.stack([item[1] for item in batch], dim=0)
    label_batch = torch.tensor([item[2] for item in batch], dtype=torch.long)

    return imu_batch, frame_batch, label_batch


def get_data(args) -> dict[str, DataLoader]:
    data: dict[str, DataLoader] = {}

    train_set = GdyDataset(args.data_dir, split="train")
    val_set = GdyDataset(args.data_dir, split="val")

    train_sampler = DistributedSampler(train_set) if args.distributed else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=train_sampler is None,
        num_workers=args.workers,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=True,
        collate_fn=collate_fn,
        prefetch_factor=5,
    )
    train_loader.num_samples = len(train_set)
    train_loader.num_batches = len(train_loader)

    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=False,
        collate_fn=collate_fn,
    )
    val_loader.num_samples = len(val_set)
    val_loader.num_batches = len(val_loader)

    data["train"] = train_loader
    data["val"] = val_loader
    return data, train_set.label_to_idx, train_set.idx_to_label
