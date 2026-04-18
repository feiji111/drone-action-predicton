import argparse
import os
import time

import numpy as np
import tensorrt as trt
from cuda import cudart
from torchvision import transforms
from torch.utils.data import DataLoader

from src.dataset import GdyDataset, collate_fn


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build TensorRT engine from ONNX and benchmark latency/accuracy."
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=os.path.join("exports", "model.onnx"),
        help="Path to source ONNX model.",
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        default=None,
        help="Path to TensorRT engine. If omitted, derive from onnx path and precision.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="gdy",
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--precision",
        choices=("fp32", "fp16", "int8"),
        default="fp16",
        help="TensorRT precision flag used while building the engine. INT8 uses TensorRT native calibration.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Validation batch size used for accuracy evaluation.",
    )
    parser.add_argument(
        "--latency-batch-size",
        type=int,
        default=1,
        help="Batch size used for latency benchmark.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--latency-sample-index", type=int, default=0)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=150,
        help="Optimization profile max sequence length.",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=224,
        help="Optimization profile max image size.",
    )
    parser.add_argument(
        "--workspace-gb",
        type=float,
        default=4.0,
        help="TensorRT builder workspace size in GB.",
    )
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="Rebuild TensorRT engine even if an engine file already exists.",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Calibration dataset root. Defaults to --data-dir when precision=int8.",
    )
    parser.add_argument(
        "--calib-split",
        choices=("train", "val"),
        default="train",
        help="Dataset split used for TensorRT INT8 calibration.",
    )
    parser.add_argument(
        "--calib-batch-size",
        type=int,
        default=8,
        help="Calibration batch size for TensorRT INT8 calibration.",
    )
    parser.add_argument(
        "--calib-samples",
        type=int,
        default=256,
        help="Maximum number of calibration samples.",
    )
    parser.add_argument(
        "--calib-cache",
        type=str,
        default=None,
        help="TensorRT INT8 calibration cache file path.",
    )
    return parser.parse_args()


def cuda_check(result):
    err = result[0]
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError("CUDA runtime call failed: {}".format(err))
    if len(result) == 1:
        return None
    if len(result) == 2:
        return result[1]
    return result[1:]


def resolve_engine_path(args):
    if args.engine_path:
        return args.engine_path
    base, _ = os.path.splitext(args.onnx_path)
    return "{}.{}.engine".format(base, args.precision)


def resolve_calibration_cache_path(args):
    if args.calib_cache:
        return args.calib_cache
    base, _ = os.path.splitext(args.onnx_path)
    return "{}.int8.calib.cache".format(base)


def build_engine(args, engine_path):
    if not os.path.exists(args.onnx_path):
        raise FileNotFoundError("ONNX file not found: {}".format(args.onnx_path))

    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(args.onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errors = []
            for i in range(parser.num_errors):
                errors.append(str(parser.get_error(i)))
            raise RuntimeError("Failed to parse ONNX:\n{}".format("\n".join(errors)))

    config = builder.create_builder_config()
    workspace_size = int(args.workspace_gb * (1 << 30))
    if hasattr(config, "set_memory_pool_limit"):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_size)
    else:
        config.max_workspace_size = workspace_size

    input_names = [network.get_input(i).name for i in range(network.num_inputs)]

    if args.precision == "fp16":
        if not builder.platform_has_fast_fp16:
            raise RuntimeError("Current platform does not support fast FP16")
        config.set_flag(trt.BuilderFlag.FP16)
    elif args.precision == "int8":
        if not builder.platform_has_fast_int8:
            raise RuntimeError("Current platform does not support fast INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        calib_dir = args.calib_dir or args.data_dir
        config.int8_calibrator = EntropyCalibrator(
            input_names=input_names,
            data_dir=calib_dir,
            split=args.calib_split,
            image_size=args.image_size,
            batch_size=args.calib_batch_size,
            max_samples=args.calib_samples,
            cache_file=resolve_calibration_cache_path(args),
        )

    profile = builder.create_optimization_profile()
    for i in range(network.num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = list(tensor.shape)
        if len(shape) != 3 and len(shape) != 4:
            raise RuntimeError(
                "Unsupported input shape for {}: {}".format(name, tensor.shape)
            )

        if len(shape) == 3:
            seq_len = shape[1] if shape[1] > 0 else args.seq_len
            imu_dim = shape[2]
            min_shape = (1, seq_len, imu_dim)
            opt_shape = (max(1, min(args.batch_size, 8)), seq_len, imu_dim)
            max_shape = (
                max(args.batch_size, args.latency_batch_size),
                seq_len,
                imu_dim,
            )
        else:
            channels = shape[1]
            image_h = shape[2] if shape[2] > 0 else args.image_size
            image_w = shape[3] if shape[3] > 0 else args.image_size
            min_shape = (1, channels, image_h, image_w)
            opt_shape = (
                max(1, min(args.batch_size, 8)),
                channels,
                image_h,
                image_w,
            )
            max_shape = (
                max(args.batch_size, args.latency_batch_size),
                channels,
                image_h,
                image_w,
            )
        profile.set_shape(name, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine")

    os.makedirs(os.path.dirname(engine_path) or ".", exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    return engine_path


def load_engine(engine_path):
    if not os.path.exists(engine_path):
        raise FileNotFoundError("TensorRT engine not found: {}".format(engine_path))
    runtime = trt.Runtime(TRT_LOGGER)
    with open(engine_path, "rb") as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    if engine is None:
        raise RuntimeError("Failed to deserialize TensorRT engine: {}".format(engine_path))
    return engine


def get_tensor_mode(engine, tensor_name):
    if hasattr(engine, "get_tensor_mode"):
        return engine.get_tensor_mode(tensor_name)
    return engine.get_binding_mode(engine[tensor_name])


def is_input_tensor(engine, tensor_name):
    mode = get_tensor_mode(engine, tensor_name)
    if hasattr(trt, "TensorIOMode"):
        return mode == trt.TensorIOMode.INPUT
    return mode == trt.BindingMode.INPUT


def trt_dtype_to_numpy(dtype):
    return np.dtype(trt.nptype(dtype))


def tensor_name_at(engine, index):
    if hasattr(engine, "get_tensor_name"):
        return engine.get_tensor_name(index)
    return engine[index]


class CalibrationDataset:
    def __init__(self, data_dir, split, image_size):
        transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        self.dataset = GdyDataset(data_dir, split=split, transform=transform)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        input_names,
        data_dir,
        split,
        image_size,
        batch_size,
        max_samples,
        cache_file,
    ):
        super().__init__()
        self.input_names = input_names
        self.batch_size = batch_size
        self.cache_file = cache_file
        self.current_index = 0
        self.dataset = CalibrationDataset(data_dir, split, image_size)
        self.max_samples = min(max_samples, len(self.dataset))
        if self.max_samples == 0:
            raise RuntimeError("Calibration dataset is empty")
        self.device_allocations = {}

    def get_batch_size(self):
        return self.batch_size

    def _allocate(self, name, array):
        current = self.device_allocations.get(name)
        nbytes = array.nbytes
        if current is not None and current["nbytes"] >= nbytes:
            return current["ptr"]

        if current is not None:
            cuda_check(cudart.cudaFree(current["ptr"]))

        ptr = cuda_check(cudart.cudaMalloc(nbytes))
        self.device_allocations[name] = {"ptr": ptr, "nbytes": nbytes}
        return ptr

    def get_batch(self, names):
        if self.current_index >= self.max_samples:
            return None

        batch_items = []
        end_index = min(self.current_index + self.batch_size, self.max_samples)
        for idx in range(self.current_index, end_index):
            batch_items.append(self.dataset[idx])
        self.current_index = end_index

        imu_batch, frame_batch, _ = collate_fn(batch_items)
        feeds = {
            self.input_names[0]: imu_batch.detach().cpu().numpy().astype(np.float32, copy=False),
            self.input_names[1]: frame_batch.detach().cpu().numpy().astype(np.float32, copy=False),
        }

        device_ptrs = []
        for name in names:
            if name not in feeds:
                raise RuntimeError(
                    "Calibration input name {} not found in feeds {}".format(
                        name, list(feeds)
                    )
                )
            array = np.ascontiguousarray(feeds[name])
            ptr = self._allocate(name, array)
            cuda_check(
                cudart.cudaMemcpy(
                    ptr,
                    array.ctypes.data,
                    array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            device_ptrs.append(int(ptr))
        return device_ptrs

    def read_calibration_cache(self):
        if self.cache_file and os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        if not self.cache_file:
            return
        os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def __del__(self):
        try:
            for allocation in self.device_allocations.values():
                cuda_check(cudart.cudaFree(allocation["ptr"]))
        except Exception:
            pass


class TensorRTInfer:
    def __init__(self, engine_path):
        self.engine = load_engine(engine_path)
        self.context = self.engine.create_execution_context()
        if self.context is None:
            raise RuntimeError("Failed to create TensorRT execution context")

        self.num_io_tensors = (
            self.engine.num_io_tensors
            if hasattr(self.engine, "num_io_tensors")
            else self.engine.num_bindings
        )
        self.tensor_names = [
            tensor_name_at(self.engine, i) for i in range(self.num_io_tensors)
        ]
        self.input_names = [
            name for name in self.tensor_names if is_input_tensor(self.engine, name)
        ]
        self.output_names = [
            name for name in self.tensor_names if not is_input_tensor(self.engine, name)
        ]

        if len(self.input_names) != 2:
            raise RuntimeError(
                "Expected 2 TensorRT inputs, got {}: {}".format(
                    len(self.input_names), self.input_names
                )
            )
        if len(self.output_names) != 1:
            raise RuntimeError(
                "Expected 1 TensorRT output, got {}: {}".format(
                    len(self.output_names), self.output_names
                )
            )

        self.stream = cuda_check(cudart.cudaStreamCreate())
        self.allocations = {}
        self.host_outputs = {}
        self.output_dtypes = {}

    def __del__(self):
        try:
            for allocation in self.allocations.values():
                cuda_check(cudart.cudaFree(allocation["ptr"]))
            if hasattr(self, "stream") and self.stream is not None:
                cuda_check(cudart.cudaStreamDestroy(self.stream))
        except Exception:
            pass

    def _set_input_shapes(self, feed_dict):
        for name, array in feed_dict.items():
            shape = tuple(int(x) for x in array.shape)
            if hasattr(self.context, "set_input_shape"):
                if not self.context.set_input_shape(name, shape):
                    raise RuntimeError(
                        "Failed to set input shape for {} to {}".format(name, shape)
                    )
            else:
                binding_index = self.engine.get_binding_index(name)
                self.context.set_binding_shape(binding_index, shape)

    def _get_tensor_shape(self, name):
        if hasattr(self.context, "get_tensor_shape"):
            return tuple(self.context.get_tensor_shape(name))
        binding_index = self.engine.get_binding_index(name)
        return tuple(self.context.get_binding_shape(binding_index))

    def _allocate_tensor(self, name, shape, dtype):
        size = int(np.prod(shape))
        nbytes = size * np.dtype(dtype).itemsize
        current = self.allocations.get(name)
        if current is not None and current["nbytes"] >= nbytes:
            return

        if current is not None:
            cuda_check(cudart.cudaFree(current["ptr"]))

        ptr = cuda_check(cudart.cudaMalloc(nbytes))
        self.allocations[name] = {
            "ptr": ptr,
            "nbytes": nbytes,
            "shape": tuple(shape),
            "dtype": np.dtype(dtype),
        }

    def _prepare_bindings(self, feed_dict):
        self._set_input_shapes(feed_dict)

        for name in self.input_names:
            array = feed_dict[name]
            self._allocate_tensor(name, array.shape, array.dtype)
            if hasattr(self.context, "set_tensor_address"):
                self.context.set_tensor_address(name, self.allocations[name]["ptr"])

        for name in self.output_names:
            shape = self._get_tensor_shape(name)
            if any(dim < 0 for dim in shape):
                raise RuntimeError(
                    "Output {} still has dynamic shape after shape setup: {}".format(
                        name, shape
                    )
                )
            if hasattr(self.engine, "get_tensor_dtype"):
                dtype = trt_dtype_to_numpy(self.engine.get_tensor_dtype(name))
            else:
                binding_index = self.engine.get_binding_index(name)
                dtype = trt_dtype_to_numpy(self.engine.get_binding_dtype(binding_index))
            self._allocate_tensor(name, shape, dtype)
            self.host_outputs[name] = np.empty(shape, dtype=dtype)
            self.output_dtypes[name] = dtype
            if hasattr(self.context, "set_tensor_address"):
                self.context.set_tensor_address(name, self.allocations[name]["ptr"])

    def infer(self, feed_dict):
        self._prepare_bindings(feed_dict)

        for name in self.input_names:
            array = np.ascontiguousarray(feed_dict[name])
            cuda_check(
                cudart.cudaMemcpyAsync(
                    self.allocations[name]["ptr"],
                    array.ctypes.data,
                    array.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                    self.stream,
                )
            )

        if hasattr(self.context, "execute_async_v3"):
            ok = self.context.execute_async_v3(self.stream)
        else:
            bindings = [0] * self.num_io_tensors
            for i, name in enumerate(self.tensor_names):
                bindings[i] = self.allocations[name]["ptr"]
            ok = self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream)
        if not ok:
            raise RuntimeError("TensorRT execution failed")

        outputs = {}
        for name in self.output_names:
            host_output = self.host_outputs[name]
            cuda_check(
                cudart.cudaMemcpyAsync(
                    host_output.ctypes.data,
                    self.allocations[name]["ptr"],
                    host_output.nbytes,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                    self.stream,
                )
            )
            outputs[name] = host_output

        cuda_check(cudart.cudaStreamSynchronize(self.stream))
        return outputs


def to_numpy_feed(engine_runner, imu, frame):
    return {
        engine_runner.input_names[0]: imu.detach().cpu().numpy().astype(np.float32, copy=False),
        engine_runner.input_names[1]: frame.detach().cpu().numpy().astype(np.float32, copy=False),
    }


def benchmark_latency(engine_runner, dataset, sample_index, latency_batch_size, warmup, runs):
    if len(dataset) == 0:
        raise RuntimeError("Validation dataset is empty")
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(
            "latency-sample-index {} out of range [0, {})".format(
                sample_index, len(dataset)
            )
        )

    imu, frame, _ = dataset[sample_index]
    imu = imu.unsqueeze(0).repeat(latency_batch_size, 1, 1)
    frame = frame.unsqueeze(0).repeat(latency_batch_size, 1, 1, 1)
    feed = to_numpy_feed(engine_runner, imu, frame)

    for _ in range(warmup):
        engine_runner.infer(feed)

    timings_ms = []
    for _ in range(runs):
        start = time.perf_counter()
        engine_runner.infer(feed)
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
        "warmup": warmup,
        "runs": runs,
        "batch_size": latency_batch_size,
    }


def evaluate_accuracy(engine_runner, dataloader):
    correct = 0
    total = 0
    output_name = engine_runner.output_names[0]

    for imu, frame, labels in dataloader:
        outputs = engine_runner.infer(to_numpy_feed(engine_runner, imu, frame))
        preds = np.argmax(outputs[output_name], axis=1)
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
    engine_path = resolve_engine_path(args)

    if args.force_rebuild or not os.path.exists(engine_path):
        build_engine(args, engine_path)

    val_set, val_loader = build_val_loader(args)
    engine_runner = TensorRTInfer(engine_path)

    latency_stats = benchmark_latency(
        engine_runner,
        val_set,
        args.latency_sample_index,
        args.latency_batch_size,
        args.warmup,
        args.runs,
    )
    accuracy_stats = evaluate_accuracy(engine_runner, val_loader)

    print("ONNX: {}".format(args.onnx_path))
    print("TensorRT engine: {}".format(engine_path))
    print("Precision: {}".format(args.precision))
    print(
        "End-to-end latency (ms, batch={}): mean={:.3f}, p50={:.3f}, p95={:.3f}, min={:.3f}, max={:.3f}, std={:.3f}".format(
            latency_stats["batch_size"],
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
