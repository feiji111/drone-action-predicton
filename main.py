import os
import sys
import json
import time
import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from collections import defaultdict
import torch.distributed as dist
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

from src.params import parser_gen
from src.logger import setup_logging
from src.model import MultimodalFusion, get_model
from src.train import train_epoch, evaluate
from src.scheduler import CosineAnnealingWarmupScheduler
from src.dataset import get_data

import swanlab


def main():
    args = parser_gen()

    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    os.makedirs(log_base_path, exist_ok=True)
    log_filename = "out.log"
    args.log_path = os.path.join(log_base_path, log_filename)
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)
    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    args.device = str(device)
    logging.info(f"Using device: {device}")

    # Init swanlab
    swanlab.init(
        project=args.swanlab_project_name,
        experiment_name=args.name,
    )

    # Datasets
    data, label_to_idx, idx_to_label = get_data(args)

    # Model
    # model = MultimodalFusion(num_classes=5, pretrained_visual=True).to(device)
    model = get_model(args)
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {total_params:,}")

    # Caculate total steps
    # total_steps = (data["train"].num_samples + args.batch_size - 1) // args.batch_size
    total_steps = data["train"].num_batches * args.epochs

    # Loss & Optimizer
    loss = nn.CrossEntropyLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # scheduler = CosineAnnealingWarmupScheduler(
    #     optimizer, warmup=args.warmup, T_max=total_steps
    # )
    # If some params are not passed, we use the default values based on model name.
    exclude = lambda n, p: (
        p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n
    )
    include = lambda n, p: not exclude(n, p)

    named_parameters = list(model.named_parameters())
    gain_or_bias_params = [
        p for n, p in named_parameters if exclude(n, p) and p.requires_grad
    ]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": args.wd},
        ],
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
    )

    scheduler = CosineAnnealingWarmupScheduler(
        optimizer, warmup=args.warmup, T_max=total_steps
    )

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info("Compiling model...")

        filter_prefixes = (
            "torch._dynamo",
            "torch._inductor",
            "torch._functorch",
            "torch._utils_internal",
            "torch.fx",
        )

        for name in logging.root.manager.loggerDict:
            if name.startswith(filter_prefixes):
                logging.getLogger(name).setLevel(logging.WARNING)

        model = torch.compile(original_model)

    # Checkpoints
    Path(args.ckpt_dir).mkdir(parents=True, exist_ok=True)

    # Training loop
    best_val_acc = 0.0

    logging.info("\n" + "=" * 60)
    logging.info("Training Multimodal Fusion Model (IMU + Optical Flow)")
    logging.info("=" * 60)

    for epoch in range(0, args.epochs):
        logging.info(f"Start epoch {epoch}")
        train_epoch(model, data, optimizer, scheduler, loss, args, epoch)
        val_loss, val_acc = evaluate(model, data, loss, epoch, args)

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": original_model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                    "label_to_idx": label_to_idx,
                    "idx_to_label": idx_to_label,
                },
                f"{args.ckpt_dir}/best_model.pth",
            )
            logging.info(f"  --> New best: {best_val_acc:.4f} (saved)")

    logging.info(f"\nBest validation accuracy: {best_val_acc:.4f}")
    logging.info(f"Checkpoint saved to: {args.ckpt_dir}/best_model.pth")


if __name__ == "__main__":
    main()
