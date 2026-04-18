import argparse, json
import numpy as np
import torch
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models.multimodal_fusion import MultimodalFusion
from src.uav_state_recognition.dataset_multimodal import MultimodalDataset, collate_fn


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imu, frames, labels in loader:
            imu = imu.to(device)
            frames = [f.to(device) for f in frames]
            labels = labels.to(device)
            logits = model(imu, frames)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def evaluate_imu_only(model, loader, device):
    """测试降级模式（仅IMU）"""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imu, frames, labels in loader:
            imu = imu.to(device)
            labels = labels.to(device)
            logits = model.imu_only_forward(imu)
            all_preds.extend(logits.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return np.array(all_preds), np.array(all_labels)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/racing_multimodal")
    parser.add_argument(
        "--flight_img_root", default="data/raw/racing/drone-racing-dataset/autonomous"
    )
    parser.add_argument("--checkpoint", default="checkpoints_multimodal/best_model.pth")
    parser.add_argument(
        "--baseline_checkpoint",
        default="checkpoints/best_model.pth",
        help="纯IMU模型checkpoint用于对比",
    )
    parser.add_argument("--output_dir", default="results/multimodal")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 加载多模态模型
    ckpt = torch.load(args.checkpoint, map_location=device)
    idx_to_label = ckpt["idx_to_label"]
    label_names = [idx_to_label[i] for i in range(len(idx_to_label))]

    model = MultimodalFusion(num_classes=5, pretrained_visual=False).to(device)
    model.load_state_dict(ckpt["model_state_dict"])

    # 测试集
    test_set = MultimodalDataset(f"{args.data_dir}/test", args.flight_img_root)
    test_loader = DataLoader(
        test_set, batch_size=16, shuffle=False, collate_fn=collate_fn, num_workers=4
    )

    # ── 多模态评估 ──────────────────────────────────────────────────
    print("=" * 60)
    print("多模态融合模型 (IMU + Vision)")
    print("=" * 60)
    preds, labels = evaluate(model, test_loader, device)
    report = classification_report(labels, preds, target_names=label_names, digits=4)
    print(report)
    acc_mm = (preds == labels).mean()

    with open(f"{args.output_dir}/report_multimodal.txt", "w") as f:
        f.write(report)

    # 混淆矩阵
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=label_names,
        yticklabels=label_names,
    )
    plt.title(f"Multimodal Fusion (acc={acc_mm:.4f})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(f"{args.output_dir}/confusion_multimodal.png", dpi=150)
    plt.close()

    # ── 降级模式：仅IMU ─────────────────────────────────────────────
    print("=" * 60)
    print("降级模式：仅IMU (多模态模型的IMU流)")
    print("=" * 60)
    preds_imu, _ = evaluate_imu_only(model, test_loader, device)
    report_imu = classification_report(
        labels, preds_imu, target_names=label_names, digits=4
    )
    print(report_imu)
    acc_imu = (preds_imu == labels).mean()

    # ── 基线模型对比（纯IMU训练的模型）────────────────────────────
    from src.uav_state_recognition.models import MultimodalStateClassifier
    from src.uav_state_recognition.dataset_simple import UAVMotionDataset

    baseline_acc = None
    if Path(args.baseline_checkpoint).exists():
        print("=" * 60)
        print("基线模型 (纯IMU训练)")
        print("=" * 60)
        bl_ckpt = torch.load(args.baseline_checkpoint, map_location=device)
        bl_model = MultimodalStateClassifier(
            telemetry_dim=6,
            telemetry_hidden_dim=128,
            fusion_hidden_dim=128,
            num_classes=5,
            use_video=False,
            use_lstm=True,
        ).to(device)
        bl_model.load_state_dict(bl_ckpt["model_state_dict"])
        bl_model.eval()

        bl_test = UAVMotionDataset("data/racing_prepared", "test")
        from torch.utils.data import DataLoader as DL

        bl_loader = DL(bl_test, batch_size=64, shuffle=False, num_workers=2)

        all_p, all_l = [], []
        with torch.no_grad():
            for batch in bl_loader:
                out = bl_model(None, batch["telemetry"].to(device))
                all_p.extend(out.argmax(1).cpu().numpy())
                all_l.extend(batch["label"].numpy())
        bl_preds, bl_labels = np.array(all_p), np.array(all_l)
        baseline_acc = (bl_preds == bl_labels).mean()
        bl_report = classification_report(
            bl_labels, bl_preds, target_names=label_names, digits=4
        )
        print(bl_report)

    # ── 汇总对比 ────────────────────────────────────────────────────
    print("=" * 60)
    print("性能对比汇总")
    print("=" * 60)
    print(f"{'模型':<30} {'测试准确率':>12}")
    print("-" * 44)
    if baseline_acc is not None:
        print(f"{'纯IMU训练 (基线)':<30} {baseline_acc:>12.4f}")
    print(f"{'多模态IMU流 (降级模式)':<30} {acc_imu:>12.4f}")
    print(f"{'多模态融合 (IMU+Vision)':<30} {acc_mm:>12.4f}")
    if baseline_acc is not None:
        lift = (acc_mm - baseline_acc) * 100
        print(f"\n相比基线提升: {lift:+.2f}%")

    # 保存对比结果
    comparison = {
        "baseline_imu_only": float(baseline_acc) if baseline_acc else None,
        "multimodal_imu_stream": float(acc_imu),
        "multimodal_full": float(acc_mm),
    }
    with open(f"{args.output_dir}/comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    print(f"\n结果已保存到: {args.output_dir}/")


if __name__ == "__main__":
    main()
