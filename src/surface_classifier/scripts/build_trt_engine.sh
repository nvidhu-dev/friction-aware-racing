#!/usr/bin/env bash
# Build the FP16 TensorRT engine on the Jetson. Run after copying surface_mnv3.onnx over.
set -euo pipefail

ONNX="${1:-models/surface_mnv3.onnx}"
ENGINE="${2:-models/surface_mnv3_fp16.trt}"

trtexec \
    --onnx="${ONNX}" \
    --fp16 \
    --saveEngine="${ENGINE}" \
    --workspace=2048

echo "built ${ENGINE}"
