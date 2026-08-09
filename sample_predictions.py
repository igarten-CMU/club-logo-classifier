"""Render a grid of test-set predictions, including every misclassification.

Produces assets_grouped/sample_predictions.png for the README.
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data import IMAGENET_MEAN, IMAGENET_STD, build_datasets, build_loaders, get_device
from evaluate import collect_predictions
from train import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best_grouped.pt")
    parser.add_argument("--assets", default="assets_grouped")
    parser.add_argument("--n-correct", type=int, default=5)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]
    grouped = ckpt.get("grouped", False)

    model = build_model(len(classes), unfreeze=ckpt.get("unfreeze", False)).to(device)
    model.load_state_dict(ckpt["model_state"])

    _, _, test_loader, _ = build_loaders(batch_size=64, num_workers=4,
                                         grouped=grouped, threshold=ckpt.get("threshold", 0.02))
    probs, labels = collect_predictions(model, test_loader, device)
    preds = probs.argmax(dim=1)

    wrong = (preds != labels).nonzero(as_tuple=True)[0].tolist()
    # Lowest-confidence correct predictions are the interesting ones -- a grid of
    # 99.9%-confident hits says nothing.
    correct = (preds == labels).nonzero(as_tuple=True)[0]
    conf = probs[correct, labels[correct]]
    hardest = correct[conf.argsort()[:args.n_correct]].tolist()

    show = wrong + hardest
    _, _, test_ds, _ = build_datasets(grouped=grouped, threshold=ckpt.get("threshold", 0.02))

    mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    n = len(show)
    fig, axes = plt.subplots(2, n, figsize=(2.6 * n, 5.6),
                             gridspec_kw={"height_ratios": [1.15, 1]})
    if n == 1:
        axes = axes.reshape(2, 1)

    for col, i in enumerate(show):
        img, true_label = test_ds[i]
        img = (img * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()

        ax_img = axes[0, col]
        ax_img.imshow(img)
        ax_img.axis("off")
        is_wrong = i in wrong
        ax_img.set_title(f"true: {classes[true_label]}", fontsize=9,
                         color="#c0392b" if is_wrong else "#222")

        top3 = probs[i].topk(3)
        names = [classes[j] for j in top3.indices]
        vals = top3.values.numpy()

        ax_bar = axes[1, col]
        colors = ["#c0392b" if names[k] != classes[true_label] else "#2980b9"
                  for k in range(3)]
        ax_bar.barh(np.arange(3)[::-1], vals, color=colors)
        ax_bar.set_yticks(np.arange(3)[::-1])
        ax_bar.set_yticklabels(names, fontsize=8)
        ax_bar.set_xlim(0, 1)
        ax_bar.set_xlabel("confidence", fontsize=8)
        ax_bar.tick_params(axis="x", labelsize=7)
        for spine in ("top", "right"):
            ax_bar.spines[spine].set_visible(False)

    fig.suptitle(
        f"Test-set predictions — all {len(wrong)} misclassification(s) "
        f"plus the {len(hardest)} lowest-confidence correct predictions",
        fontsize=11,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(args.assets, exist_ok=True)
    out = os.path.join(args.assets, "sample_predictions.png")
    plt.savefig(out, dpi=130, bbox_inches="tight")
    print(f"wrote {out}  ({len(wrong)} wrong, {len(hardest)} hardest-correct)")


if __name__ == "__main__":
    main()
