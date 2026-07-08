#!/usr/bin/env python3
"""Export a trained MobileNetV3-Small checkpoint to ONNX with fixed batch=1 input."""

import argparse
import os

import torch
from torch import nn
from torchvision import models


def build_model(num_classes):
    m = models.mobilenet_v3_small(weights=None)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', default='models/surface_mnv3.pt')
    p.add_argument('--out', default='models/surface_mnv3.onnx')
    p.add_argument('--opset', type=int, default=16)
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location='cpu')
    classes = ckpt['classes']
    input_size = ckpt['input_size']
    print(f"[load] {args.ckpt} | classes={classes} | input_size={input_size}")

    model = build_model(len(classes))
    model.load_state_dict(ckpt['model'])
    model.eval()

    dummy = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=['input'], output_names=['logits'],
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"[export] {args.out}")


if __name__ == '__main__':
    main()
