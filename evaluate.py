"""Phase 3: evaluation, confusion matrix and error analysis.

Produces (into assets/):
  training_curves.png   loss + accuracy across epochs
  confusion_matrix.png  normalised 20x20 confusion matrix on the test split
  per_class_accuracy.png
and prints top-1 / top-3 accuracy plus the worst confusions.
"""

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import classification_report, confusion_matrix

from data import build_loaders, get_device
from train import build_model


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    all_probs, all_labels = [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images)
        all_probs.append(torch.softmax(logits, dim=1).cpu())
        all_labels.append(labels)

    return torch.cat(all_probs), torch.cat(all_labels)


def plot_training_curves(history_path, out_path):
    with open(history_path) as f:
        history = json.load(f)

    epochs = [h["epoch"] for h in history]
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax_loss.plot(epochs, [h["train_loss"] for h in history], marker="o", label="train")
    ax_loss.plot(epochs, [h["val_loss"] for h in history], marker="o", label="val")
    ax_loss.set_xlabel("epoch"); ax_loss.set_ylabel("cross-entropy loss")
    ax_loss.set_title("Loss"); ax_loss.legend(); ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, [h["train_acc"] for h in history], marker="o", label="train top-1")
    ax_acc.plot(epochs, [h["val_acc"] for h in history], marker="o", label="val top-1")
    ax_acc.plot(epochs, [h["val_top3"] for h in history], marker="o", ls="--", label="val top-3")
    ax_acc.set_xlabel("epoch"); ax_acc.set_ylabel("accuracy")
    ax_acc.set_title("Accuracy"); ax_acc.legend(); ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_confusion(cm, classes, out_path):
    # Row-normalise so each row reads "of all true X, what fraction went where".
    cm_norm = cm.astype(np.float64) / cm.sum(axis=1, keepdims=True).clip(min=1)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_norm, annot=False, cmap="viridis", vmin=0, vmax=1,
                xticklabels=classes, yticklabels=classes, square=True,
                cbar_kws={"label": "fraction of true class"})
    plt.xlabel("predicted"); plt.ylabel("true")
    plt.title("Confusion matrix (row-normalised) — test split")
    plt.xticks(rotation=90); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def plot_per_class(cm, classes, out_path):
    per_class = cm.diagonal() / cm.sum(axis=1).clip(min=1)
    order = np.argsort(per_class)

    plt.figure(figsize=(10, 6))
    colors = ["#c0392b" if per_class[i] < 0.95 else "#2980b9" for i in order]
    plt.barh([classes[i] for i in order], per_class[order], color=colors)
    plt.axvline(0.95, ls="--", c="k", lw=1, label="95% target")
    plt.xlabel("top-1 accuracy"); plt.xlim(0, 1.02)
    plt.title("Per-class accuracy — test split"); plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--history", default="checkpoints/history.json")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--assets", default="assets")
    args = parser.parse_args()

    os.makedirs(args.assets, exist_ok=True)
    device = get_device()

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = ckpt["classes"]

    model = build_model(len(classes), unfreeze=ckpt.get("unfreeze", False)).to(device)
    model.load_state_dict(ckpt["model_state"])

    # The split must match the one the checkpoint was trained under, otherwise
    # the "test" set would contain images the model trained on.
    grouped = ckpt.get("grouped", False)
    print(f"Split: {'group-aware' if grouped else 'random'}")

    _, _, test_loader, _ = build_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        grouped=grouped, threshold=ckpt.get("threshold", 0.02),
    )
    probs, labels = collect_predictions(model, test_loader, device)

    preds = probs.argmax(dim=1)
    top1 = (preds == labels).float().mean().item()
    top3 = (probs.topk(3, dim=1).indices == labels.unsqueeze(1)).any(dim=1).float().mean().item()

    print(f"Test top-1 accuracy: {top1:.4f}")
    print(f"Test top-3 accuracy: {top3:.4f}")
    print()
    print(classification_report(labels.numpy(), preds.numpy(), target_names=classes, digits=3))

    cm = confusion_matrix(labels.numpy(), preds.numpy(), labels=list(range(len(classes))))

    plot_training_curves(args.history, os.path.join(args.assets, "training_curves.png"))
    plot_confusion(cm, classes, os.path.join(args.assets, "confusion_matrix.png"))
    plot_per_class(cm, classes, os.path.join(args.assets, "per_class_accuracy.png"))

    # Worst off-diagonal cells: the pairs the model actually mixes up.
    off = cm.copy()
    np.fill_diagonal(off, 0)
    pairs = [(off[i, j], classes[i], classes[j])
             for i in range(len(classes)) for j in range(len(classes)) if off[i, j] > 0]
    pairs.sort(reverse=True)

    if pairs:
        print("Top confusions (true -> predicted, count):")
        for count, true_c, pred_c in pairs[:10]:
            print(f"  {true_c:20s} -> {pred_c:20s} {count}")
    else:
        print("No misclassifications on the test split.")

    print(f"\nPlots written to {args.assets}/")


if __name__ == "__main__":
    main()
