"""Transfer-learning training loop for the EPL crest classifier.

Default run is a linear probe: the ImageNet-pretrained EfficientNet-B0 backbone
is frozen and only the new 20-way head is trained. Pass --unfreeze to fine-tune
the whole network at a lower LR once the head has converged.
"""

import argparse
import json
import time

import torch
import torch.nn as nn
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

from data import build_loaders, get_device


def build_model(num_classes, unfreeze=False):
    model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)

    if not unfreeze:
        for param in model.parameters():
            param.requires_grad = False

    # Replacing the layer creates fresh params with requires_grad=True, so the
    # head stays trainable even though the freeze loop ran above it.
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)

    return model


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, top3_correct, seen = 0.0, 0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)

        total_loss += loss.item() * labels.size(0)
        seen += labels.size(0)

        correct += (logits.argmax(dim=1) == labels).sum().item()
        top3 = logits.topk(3, dim=1).indices
        top3_correct += (top3 == labels.unsqueeze(1)).any(dim=1).sum().item()

    return total_loss / seen, correct / seen, top3_correct / seen


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, seen = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        seen += labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()

    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--unfreeze", action="store_true",
                        help="Fine-tune the whole backbone instead of a linear probe.")
    parser.add_argument("--grouped", action="store_true",
                        help="Use the leakage-free group-aware split.")
    parser.add_argument("--threshold", type=float, default=0.02,
                        help="Near-duplicate clustering threshold for --grouped.")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--history", default="checkpoints/history.json")
    args = parser.parse_args()

    device = get_device()
    print(f"Device: {device}")

    train_loader, val_loader, _, classes = build_loaders(
        batch_size=args.batch_size, num_workers=args.num_workers,
        grouped=args.grouped, threshold=args.threshold,
    )
    split_name = "group-aware" if args.grouped else "random"
    print(f"Split: {split_name} | classes {len(classes)} | "
          f"train {len(train_loader.dataset)} | val {len(val_loader.dataset)}")

    model = build_model(len(classes), unfreeze=args.unfreeze).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable:,} / {total:,}")

    criterion = nn.CrossEntropyLoss()
    # Fine-tuning pretrained features at 1e-3 would wreck them, so drop the LR.
    lr = args.lr * 0.1 if args.unfreeze else args.lr
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    import os
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    history = []
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_top3 = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - start

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc,
            "val_loss": val_loss, "val_acc": val_acc, "val_top3": val_top3,
            "lr": scheduler.get_last_lr()[0], "seconds": elapsed,
        })

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"model_state": model.state_dict(),
                        "classes": classes,
                        "unfreeze": args.unfreeze,
                        "grouped": args.grouped,
                        "threshold": args.threshold,
                        "val_acc": val_acc}, args.checkpoint)
            marker = "  <- saved"

        print(f"epoch {epoch:2d}/{args.epochs} | "
              f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
              f"val loss {val_loss:.4f} acc {val_acc:.4f} top3 {val_top3:.4f} | "
              f"{elapsed:.0f}s{marker}")

        with open(args.history, "w") as f:
            json.dump(history, f, indent=2)

    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {args.checkpoint}")


if __name__ == "__main__":
    main()
