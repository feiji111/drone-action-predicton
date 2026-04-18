import argparse
import os
import time

import numpy as np
import onnxruntime as ort
from torch.utils.data import DataLoader

from src.dataset import GdyDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark INT8 ONNX model latency and validation accuracy."
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=os.path.join("exports", "model.int8.onnx"),
        help="Path to exported ONNX model.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="gdy",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Validation batch size used for accuracy evaluation.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Execution provider used for accuracy evaluation.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup iterations for latency benchmark.",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=100,
        help="Measured iterations for latency benchmark.",
    )
    parser.add_argument(
        "--latency-sample-index",
        type=int,
        default=0,
        help="Validation sample index used for latency test.",
    )
    parser.add_argument(
        "--intra-op-threads",
        type=int,
        default=0,
        help="ONNX Runtime intra-op threads. 0 means runtime default.",
    )
    parser.add_argument(
        "--inter-op-threads",
        type=int,
        default=0,
        help="ONNX Runtime inter-op threads. 0 means runtime default.",
    )
    return parser.parse_args()


def build_session(model_path, device, intra_op_threads, inter_op_threads):
    if not os.path.exists(model_path):
        raise FileNotFoundError("Model file not found: {}".format(model_path))

    session_options = ort.SessionOptions()
    if intra_op_threads > 0:
        session_options.intra_op_num_threads = intra_op_threads
    if inter_op_threads > 0:
        session_options.inter_op_num_threads = inter_op_threads

    available = ort.get_available_providers()
    if device == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError(
                "CUDAExecutionProvider is not available. Available providers: {}".format(
                    available
                )
            )
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    else:
        providers = ["CPUExecutionProvider"]

    session = ort.InferenceSession(
        model_path,
        sess_options=session_options,
        providers=providers,
    )
    return session


def get_io_names(session):
    input_names = [item.name for item in session.get_inputs()]
    output_names = [item.name for item in session.get_outputs()]

    if len(input_names) != 2:
        raise RuntimeError(
            "Expected 2 model inputs (imu, image), got {}: {}".format(
                len(input_names), input_names
            )
        )
    if len(output_names) != 1:
        raise RuntimeError(
            "Expected 1 model output, got {}: {}".format(len(output_names), output_names)
        )

    return input_names, output_names[0]


def to_numpy_feed(input_names, imu, frame):
    return {
        input_names[0]: imu.detach().cpu().numpy().astype(np.float32, copy=False),
        input_names[1]: frame.detach().cpu().numpy().astype(np.float32, copy=False),
    }


def benchmark_latency(session, dataset, input_names, output_name, sample_index, warmup, runs):
    if len(dataset) == 0:
        raise RuntimeError("Validation dataset is empty")
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            "latency-sample-index {} out of range [0, {})".format(
                sample_index, len(dataset)
            )
        )

    imu, frame, _ = dataset[sample_index]
    imu = imu.unsqueeze(0)
    frame = frame.unsqueeze(0)
    feed = to_numpy_feed(input_names, imu, frame)

    for _ in range(warmup):
        session.run([output_name], feed)

    timings_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        session.run([output_name], feed)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        timings_ms.append(elapsed_ms)

    timings_ms = np.asarray(timings_ms, dtype=np.float64)
    return {
        "mean_ms": float(np.mean(timings_ms)),
        "std_ms": float(np.std(timings_ms)),
        "min_ms": float(np.min(timings_ms)),
        "max_ms": float(np.max(timings_ms)),
        "p50_ms": float(np.percentile(timings_ms, 50)),
        "p95_ms": float(np.percentile(timings_ms, 95)),
        "runs": runs,
        "warmup": warmup,
    }


def evaluate_accuracy(session, dataloader, input_names, output_name):
    correct = 0
    total = 0

    for imu, frame, labels in dataloader:
        outputs = session.run(
            [output_name],
            to_numpy_feed(input_names, imu, frame),
        )[0]
        preds = np.argmax(outputs, axis=1)
        labels_np = labels.detach().cpu().numpy()

        correct += int((preds == labels_np).sum())
        total += int(labels_np.shape[0])

    if total == 0:
        raise RuntimeError("Validation dataloader is empty")

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total,
    }


def build_val_loader(args):
    val_set = GdyDataset(args.data_dir, split="val")
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        drop_last=False,
        collate_fn=collate_fn,
    )
    return val_set, val_loader


def main():
    args = parse_args()

    val_set, val_loader = build_val_loader(args)
    session = build_session(
        args.model_path,
        args.device,
        args.intra_op_threads,
        args.inter_op_threads,
    )
    input_names, output_name = get_io_names(session)

    latency_stats = benchmark_latency(
        session,
        val_set,
        input_names,
        output_name,
        args.latency_sample_index,
        args.warmup,
        args.runs,
    )
    accuracy_stats = evaluate_accuracy(
        session,
        val_loader,
        input_names,
        output_name,
    )

    print("Model: {}".format(args.model_path))
    print("Providers: {}".format(session.get_providers()))
    print("Latency batch size: 1")
    print(
        "End-to-end latency (ms): mean={:.3f}, p50={:.3f}, p95={:.3f}, min={:.3f}, max={:.3f}, std={:.3f}".format(
            latency_stats["mean_ms"],
            latency_stats["p50_ms"],
            latency_stats["p95_ms"],
            latency_stats["min_ms"],
            latency_stats["max_ms"],
            latency_stats["std_ms"],
        )
    )
    print(
        "Validation accuracy: {:.4f} ({}/{})".format(
            accuracy_stats["accuracy"],
            accuracy_stats["correct"],
            accuracy_stats["total"],
        )
    )


if __name__ == "__main__":
    main()
