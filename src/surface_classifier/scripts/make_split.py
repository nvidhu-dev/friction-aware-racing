#!/usr/bin/env python3
"""Walk ~/surface_data/<label>/*.png and write stratified train/val manifests.

Usage:
    python scripts/make_split.py --root ~/surface_data --out-dir . --val-frac 0.2 --seed 0
"""

import argparse
import csv
import os
import random
from glob import glob


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', default='~/surface_data')
    p.add_argument('--out-dir', default='.')
    p.add_argument('--val-frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()

    root = os.path.expanduser(args.root)
    out_dir = os.path.expanduser(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)
    rng = random.Random(args.seed)

    labels = sorted(d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d)))
    train_rows, val_rows = [], []
    for label in labels:
        files = sorted(glob(os.path.join(root, label, '*.png')))
        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * args.val_frac))) if files else 0
        for f in files[:n_val]:
            val_rows.append((f, label))
        for f in files[n_val:]:
            train_rows.append((f, label))
        print(f"  {label:>10}: {len(files)} total, {n_val} val, {len(files) - n_val} train")

    rng.shuffle(train_rows)
    rng.shuffle(val_rows)

    for name, rows in (('manifest_train.csv', train_rows), ('manifest_val.csv', val_rows)):
        path = os.path.join(out_dir, name)
        with open(path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['path', 'label'])
            w.writerows(rows)
        print(f"wrote {path} ({len(rows)} rows)")

    classes_path = os.path.join(out_dir, 'classes.txt')
    with open(classes_path, 'w') as f:
        for label in labels:
            f.write(label + '\n')
    print(f"wrote {classes_path} ({len(labels)} classes)")


if __name__ == '__main__':
    main()
