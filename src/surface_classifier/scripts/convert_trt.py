#!/usr/bin/env python3
"""Build a TensorRT engine from an ONNX file using the TensorRT Python API.

Mirrors the pattern used in lab-7-vision-lab-team3/python_scripts/convert_trt.py
so the engine is produced the same way as the existing detection model.

Run on the Jetson (after copying surface_mnv3.onnx over):
    python scripts/convert_trt.py --onnx models/surface_mnv3.onnx \
        --engine models/surface_mnv3_fp16.trt --fp16
"""

import argparse

import tensorrt as trt


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(onnx_path, engine_path, fp16=True, workspace_mb=256):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, TRT_LOGGER)

    print(f"[trt] parsing {onnx_path}")
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(parser.get_error(i))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * (1 << 20)
    )

    if fp16 and builder.platform_has_fast_fp16:
        print("[trt] FP16 enabled")
        config.set_flag(trt.BuilderFlag.FP16)
    elif fp16:
        print("[trt] FP16 requested but platform reports no fast FP16; falling back to FP32")

    print("[trt] building engine (this can take 1-3 min on Orin Nano)")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("engine build failed")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"[trt] saved {engine_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="models/surface_mnv3.onnx")
    p.add_argument("--engine", default="models/surface_mnv3_fp16.trt")
    p.add_argument("--fp16", action="store_true", default=True)
    p.add_argument("--no-fp16", dest="fp16", action="store_false")
    p.add_argument("--workspace-mb", type=int, default=256)
    args = p.parse_args()

    build_engine(args.onnx, args.engine, fp16=args.fp16,
                 workspace_mb=args.workspace_mb)


if __name__ == "__main__":
    main()
