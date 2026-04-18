import logging
import time
import torch
import swanlab
import torch.nn as nn
from src.meter import AverageMeter

import torch.profiler as profiler
from contextlib import nullcontext


def train_epoch(model, data, optimizer, scheduler, loss, args, epoch):
    device = torch.device(args.device)
    model.train()
    dataloader = data["train"]
    # num_batch_per_epoch = (len(dataloader) + args.batch_size - 1) // args.batch_size
    num_batches_per_epoch = dataloader.num_batches

    losses_m = {}
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    total_loss = 0.0
    correct = 0
    total = 0

    # Profile
    if args.profile:
        prof = profiler.profile(
            activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
            schedule=profiler.schedule(skip_first=2, wait=0, warmup=2, active=5),
            on_trace_ready=lambda p: p.export_chrome_trace(
                f"{args.logs}/{args.name}/trace.json"
            ),
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        )
    else:
        prof = nullcontext

    with prof:
        for i, batch in enumerate(dataloader):
            step = num_batches_per_epoch * epoch + i
            imu, frame, labels = batch
            imu = imu.to(device, non_blocking=True)
            frame = frame.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            data_time_m.update(time.time() - end)

            optimizer.zero_grad()
            logits = model(imu, frame)
            losses = loss(logits, labels)
            losses.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if args.profile:
                prof.step()

            batch_time_m.update(time.time() - end)
            end = time.time()

            total_loss += losses.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            # calculate metric
            samples_per_second = (
                args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
            )
            samples_per_second_per_gpu = (
                args.accum_freq * args.batch_size / batch_time_m.val
            )
            if i % args.log_every_n_steps == 0 or i == num_batches_per_epoch:
                logging.info(
                    f"Train Epoch: {epoch}LR: {optimizer.param_groups[0]['lr']:5f} "
                    # f"Loss: {loss.item()}"
                    # f"Acc: {correct}"
                )
                log_data = {
                    "data_time": data_time_m.val,
                    "batch_time": batch_time_m.val,
                    "samples_per_second": samples_per_second,
                    "samples_per_second_per_gpu": samples_per_second_per_gpu,
                    "lr": optimizer.param_groups[0]["lr"],
                    "loss": losses.item(),
                }
                log_data.update({name: val.val for name, val in losses_m.items()})
                log_data = {"train/" + name: val for name, val in log_data.items()}

                if args.swanlab:
                    log_data["step"] = step
                    swanlab.log(log_data, step=step)

                batch_time_m.reset()
                data_time_m.reset()


def evaluate(model, data, loss, epoch, args):
    log_data = {}
    device = torch.device(args.device)
    dataloader = data["val"]
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    # with torch.no_grad():
    with torch.inference_mode():
        for imu, frame, labels in dataloader:
            imu = imu.to(device)
            frame = frame.to(device)
            labels = labels.to(device)

            logits = model(imu, frame)
            losses = loss(logits, labels)

            total_loss += losses.item() * labels.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    log_data = {"eval/correct": correct / total, "eval/loss": total_loss / total}

    if args.swanlab:
        swanlab.log(log_data, step=epoch)

    return total_loss / total, correct / total
