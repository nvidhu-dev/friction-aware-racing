#!/usr/bin/env python3
"""Render the surface-classifier pipeline diagram for slides."""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


STAGES = [
    ("RGB Camera",         "640 x 480\n@ 30 Hz",                "#dbeafe", "#1e3a8a"),
    ("ROI Crop",           "200 x 160 patch\n(ground, ~30-50 cm)", "#dbeafe", "#1e3a8a"),
    ("MobileNetV3-Small",  "TensorRT FP16\n224 x 224 input",     "#fde68a", "#92400e"),
    ("Temporal Smoothing", "5-frame\nmajority vote",             "#dbeafe", "#1e3a8a"),
    ("Friction Map",       "material -> tier\n(low / med / high)", "#dcfce7", "#166534"),
    ("Planner /\nController", "uses friction\nas prior",         "#fee2e2", "#991b1b"),
]

BOX_W, BOX_H = 2.35, 1.35
GAP = 0.55
PAD = 0.6


def main():
    n = len(STAGES)
    fig_w = n * BOX_W + (n - 1) * GAP + 2 * PAD
    fig_h = BOX_H + 2 * PAD + 0.8

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=200)
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, fig_h)
    ax.set_aspect("equal")
    ax.axis("off")

    y = (fig_h - BOX_H) / 2
    centers = []
    for i, (title, sub, fill, edge) in enumerate(STAGES):
        x = PAD + i * (BOX_W + GAP)
        box = FancyBboxPatch(
            (x, y), BOX_W, BOX_H,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            linewidth=1.6, edgecolor=edge, facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x + BOX_W / 2, y + BOX_H * 0.68, title,
                ha="center", va="center", fontsize=11, fontweight="bold", color=edge)
        ax.text(x + BOX_W / 2, y + BOX_H * 0.30, sub,
                ha="center", va="center", fontsize=8.5, color="#374151")
        centers.append((x, x + BOX_W, y + BOX_H / 2))

    for i in range(n - 1):
        _, x_end, cy = centers[i]
        x_next, _, _ = centers[i + 1]
        arr = FancyArrowPatch(
            (x_end + 0.04, cy), (x_next - 0.04, cy),
            arrowstyle="-|>", mutation_scale=14,
            linewidth=1.4, color="#374151",
        )
        ax.add_patch(arr)

    ax.text(fig_w / 2, fig_h - 0.25,
            "Vision-Based Surface Classifier  |  ROS 2 node, 10 Hz on Jetson",
            ha="center", va="center", fontsize=12, fontweight="bold", color="#111827")

    out = "surface_classifier/pipeline_diagram.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
