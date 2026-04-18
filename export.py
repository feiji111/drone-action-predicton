import argparse
import os

import numpy as np

import torch
from torch.export.dynamic_shapes import Dim

import onnx
from onnx import shape_inference
from onnxsim import simplify
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_dynamic,
    quantize_static,
)

from src.dataset import GdyDataset
from src.model import get_model


IMAGE_SIZE = 224


class DatasetCalibrationReader(CalibrationDataReader):
    def __init__(
        self,
        data_dir,
        input_names,
        split="val",
        max_samples=128,
    ):
        super().__init__()
        self.dataset = GdyDataset(data_dir, split=split)
        self.input_names = input_names
        self.max_samples = min(max_samples, len(self.dataset))
        self._iterator = None

    def _build_iterator(self):
        for idx in range(self.max_samples):
            imu, frame, _ = self.dataset[idx]
            sample = {
                self.input_names[0]: imu.unsqueeze(0).cpu().numpy().astype(np.float32),
                self.input_names[1]: frame.unsqueeze(0)
                .cpu()
                .numpy()
                .astype(np.float32),
            }
            yield sample

    def get_next(self):
        if self._iterator is None:
            self._iterator = self._build_iterator()
        return next(self._iterator, None)

    def rewind(self):
        self._iterator = None


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export trained multimodal model to ONNX and INT8 ONNX."
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        required=True,
        help="Path to .pth checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exports",
        help="Directory used to save exported ONNX files.",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="multimodal_fusion",
        help="Base file name without suffix.",
    )
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=150)
    parser.add_argument("--imu-input-dim", type=int, default=22)
    parser.add_argument("--image-size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--backbone", type=str, default="resnet18")
    parser.add_argument("--imu-hidden-dim", type=int, default=256)
    parser.add_argument("--visual-hidden-dim", type=int, default=256)
    parser.add_argument("--fusion-hidden-dim", type=int, default=256)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Build visual backbone with torchvision pretrained weights.",
    )
    parser.add_argument(
        "--enable-simplify",
        action="store_true",
        help="Skip onnxsim graph simplification.",
    )
    parser.add_argument(
        "--enable-int8",
        action="store_true",
        help="Only export FP32 ONNX, do not generate INT8 model.",
    )
    parser.add_argument(
        "--quant-mode",
        choices=("static", "dynamic", "auto"),
        default="auto",
        help="auto: use static when calibration data is available, else dynamic.",
    )
    parser.add_argument(
        "--calib-dir",
        type=str,
        default=None,
        help="Dataset root used for static INT8 calibration. If omitted and mode=auto, dynamic INT8 is used.",
    )
    parser.add_argument(
        "--calib-split",
        choices=("train", "val"),
        default="val",
        help="Dataset split used for calibration.",
    )
    parser.add_argument("--calib-samples", type=int, default=128)
    parser.add_argument(
        "--calib-method",
        choices=["minmax", "entropy", "percentile", "distribution"],
        default="minmax",
        help="Calibration method",
    )
    return parser.parse_args()


def make_dummy_inputs(args, device):
    imu = torch.randn(
        args.batch_size,
        args.seq_len,
        args.imu_input_dim,
        device=device,
        dtype=torch.float32,
    )
    image = torch.randn(
        args.batch_size,
        3,
        args.image_size,
        args.image_size,
        device=device,
        dtype=torch.float32,
    )
    return imu, image


def validate_and_save_onnx(model, output_path):
    onnx.checker.check_model(model)
    inferred_model = shape_inference.infer_shapes(model)
    onnx.save(inferred_model, output_path)


def export_fp32_onnx(model, args, fp32_path):
    device = torch.device(args.device)
    dummy_inputs = make_dummy_inputs(args, device)
    input_names = ["imu", "frame"]
    output_names = ["logits"]

    torch.onnx.export(
        model,
        dummy_inputs,
        fp32_path,
        external_data=False,
        opset_version=None,
        input_names=input_names,
        output_names=output_names,
        dynamic_shapes={"imu": {0: Dim("batch")}, "frame": {0: Dim("batch")}},
    )

    onnx_model = onnx.load(fp32_path)
    validate_and_save_onnx(onnx_model, fp32_path)
    return fp32_path, input_names


def quantize_int8(args, fp32_path, int8_path, input_names):
    if args.quant_mode != "auto":
        quant_mode = args.quant_mode
    else:
        quant_mode = "static" if args.calib_dir is not None else "dynamic"
    if quant_mode == "static":
        reader = DatasetCalibrationReader(
            data_dir=args.calib_dir,
            input_names=input_names,
            split=args.calib_split,
            max_samples=args.calib_samples,
        )
        quantize_static(
            model_input=fp32_path,
            model_output=int8_path,
            calibration_data_reader=reader,
            activation_type=QuantType.QInt8,
            weight_type=QuantType.QUInt8,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            calibrate_method=CalibrationMethod.MinMax,
        )
    else:
        quantize_dynamic(
            model_input=fp32_path,
            model_output=int8_path,
            weight_type=QuantType.QInt8,
            per_channel=True,
        )

    quantized_model = onnx.load(int8_path)
    onnx.checker.check_model(quantized_model)
    onnx.save(quantized_model, int8_path)
    return quant_mode


def export_model(args):
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device)
    model = get_model(args)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.to(device).eval()

    fp32_path = os.path.join(args.output_dir, f"{args.output_name}.onnx")
    fp32_path, input_names = export_fp32_onnx(model, args, fp32_path)
    print(f"Exported FP32 ONNX: {fp32_path}")

    if args.disable_int8:
        return

    int8_path = os.path.join(args.output_dir, f"{args.output_name}.int8.onnx")
    quant_mode = quantize_int8(args, fp32_path, int8_path, input_names)
    print(f"Exported INT8 ONNX ({quant_mode}): {int8_path}")


def main():
    args = parse_args()
    export_model(args)


if __name__ == "__main__":
    main()
