#!/usr/bin/env python3
"""Fine-tune MobileNetV3-Small on collected surface patches.

Inputs:
    - manifest_train.csv, manifest_val.csv (from make_split.py): columns `path,label`
    - classes.txt: one class name per line, defines the label index order

Outputs:
    - models/surface_mnv3.pt (best val-acc checkpoint)
    - models/classes.txt (copy, kept next to the checkpoint)

Run on Colab T4 or a laptop GPU. Skip MINC pretraining for v1.
"""

import argparse
import csv
import os
import shutil
from collections import Counter

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms


def load_manifest(path):
    rows = []
    with open(path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append((r['path'], r['label']))
    return rows


class PatchDataset(Dataset):
    def __init__(self, rows, classes, transform):
        self.rows = rows
        self.cls_to_idx = {c: i for i, c in enumerate(classes)}
        self.transform = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        path, label = self.rows[i]
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = self.transform(img)
        return img, self.cls_to_idx[label]


def build_transforms(input_size):
    train_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.GaussianBlur(3, sigma=(0.1, 1.5)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(int(input_size * 1.15)),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def build_model(num_classes):
    m = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_features = m.classifier[-1].in_features
    m.classifier[-1] = nn.Linear(in_features, num_classes)
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--train-manifest', default='manifest_train.csv')
    p.add_argument('--val-manifest', default='manifest_val.csv')
    p.add_argument('--classes', default='classes.txt')
    p.add_argument('--out', default='models/surface_mnv3.pt')
    p.add_argument('--input-size', type=int, default=224)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--lr-head', type=float, default=1e-3)
    p.add_argument('--lr-backbone', type=float, default=1e-4)
    p.add_argument('--num-workers', type=int, default=4)
    args = p.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) is not None and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"[device] {device}")

    with open(args.classes) as f:
        classes = [c.strip() for c in f if c.strip()]
    print(f"[classes] {classes}")

    train_rows = load_manifest(args.train_manifest)
    val_rows = load_manifest(args.val_manifest)
    print(f"[data] train={len(train_rows)} val={len(val_rows)}")

    train_tf, val_tf = build_transforms(args.input_size)
    train_ds = PatchDataset(train_rows, classes, train_tf)
    val_ds = PatchDataset(val_rows, classes, val_tf)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                          num_workers=args.num_workers, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    model = build_model(len(classes)).to(device)
    head_params = list(model.classifier.parameters())
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW([
        {'params': head_params, 'lr': args.lr_head},
        {'params': backbone_params, 'lr': args.lr_backbone},
    ], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    counts = Counter(label for _, label in train_rows)
    weights = torch.tensor([1.0 / max(1, counts[c]) for c in classes], device=device)
    weights = weights * len(classes) / weights.sum()
    print(f"[class weights] {dict(zip(classes, weights.tolist()))}")
    criterion = nn.CrossEntropyLoss(weight=weights)

    os.makedirs(os.path.dirname(args.out) or '.', exist_ok=True)
    best_val = 0.0

    for epoch in range(args.epochs):
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for x, y in train_dl:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
            train_correct += (logits.argmax(1) == y).sum().item()
            train_total += x.size(0)
        scheduler.step()

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                logits = model(x)
                val_correct += (logits.argmax(1) == y).sum().item()
                val_total += x.size(0)
        val_acc = val_correct / max(1, val_total)
        train_acc = train_correct / max(1, train_total)
        print(f"epoch {epoch+1:02d}/{args.epochs} | train_loss={train_loss/train_total:.4f} "
              f"train_acc={train_acc:.3f} val_acc={val_acc:.3f}")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({'model': model.state_dict(), 'classes': classes,
                        'input_size': args.input_size}, args.out)
            shutil.copy(args.classes, os.path.join(os.path.dirname(args.out) or '.', 'classes.txt'))
            print(f"  ↑ saved {args.out} (val_acc={val_acc:.3f})")

    print(f"[done] best val_acc={best_val:.3f}")


if __name__ == '__main__':
    main()
