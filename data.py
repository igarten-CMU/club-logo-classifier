"""Data pipeline for the EPL crest classifier.

The source images are 140x140 RGBA PNGs with transparent backgrounds. Everything
interesting about this file comes from that: a crest on a *constant* background
is trivially learnable, so the model would hit 99% val accuracy and then fall
apart on a real jersey photo. We composite each crest onto a random background
during training so the model is forced to learn the crest itself.
"""

import random

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder

DATA_ROOT = "data/raw/epl-logos-big/epl-logos-big"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

SEED = 42


def rgba_loader(path):
    """Load as RGBA, keeping the alpha channel.

    torchvision's default loader does .convert("RGB"), which discards alpha and
    exposes whatever junk RGB sits under the transparent pixels. We need alpha
    intact so CompositeBackground can decide what goes behind the crest.
    """
    with open(path, "rb") as f:
        return Image.open(f).convert("RGBA")


class CompositeBackground:
    """Flatten an RGBA crest onto an opaque background.

    Training uses a random solid colour so the network cannot use the backdrop
    as a shortcut feature. Eval uses a fixed white background so validation and
    test numbers are deterministic and reproducible across runs.
    """

    def __init__(self, randomize):
        self.randomize = randomize

    def __call__(self, img):
        # User uploads in the Gradio app arrive as RGB. Converting gives alpha=255
        # everywhere, so the paste below becomes a no-op and the photo passes
        # through untouched — same code path, no branch needed at inference.
        if img.mode != "RGBA":
            img = img.convert("RGBA")

        if self.randomize:
            bg = tuple(random.randint(0, 255) for _ in range(3))
        else:
            bg = (255, 255, 255)
        canvas = Image.new("RGB", img.size, bg)
        # Third arg uses img's alpha as the paste mask.
        canvas.paste(img, (0, 0), img)
        return canvas


def build_transforms():
    """Return (train, eval) transform pipelines.

    Note the ordering: composite first (needs a PIL RGBA image), then upscale to
    256 *before* RandomResizedCrop. Source crests are only 140px, so cropping at
    native resolution would slice off the crest's outer ring.
    """
    train_transforms = T.Compose([
        CompositeBackground(randomize=True),
        T.Resize((256, 256)),
        T.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        # No RandomHorizontalFlip: crests carry text and asymmetric heraldry.
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    eval_transforms = T.Compose([
        CompositeBackground(randomize=False),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    return train_transforms, eval_transforms


def grouped_split_indices(samples, threshold, seed):
    """Split indices 70/15/15 so near-duplicate images never straddle splits.

    Groups are assigned whole, largest-first, to whichever split is currently
    furthest below its quota (greedy bin-packing). Necessary because group sizes
    are wildly uneven -- the biggest holds ~550 images, so round-robin assignment
    would blow the quotas apart.

    Balancing is done per class, so all three splits stay class-balanced.
    """
    import collections

    from fingerprint import cluster_within_class, load_or_compute

    paths, vecs, fp_labels = load_or_compute()
    group_ids, _ = cluster_within_class(vecs, fp_labels, threshold)
    path_to_group = dict(zip(paths, group_ids))

    by_class = collections.defaultdict(list)
    for idx, (path, class_idx) in enumerate(samples):
        by_class[class_idx].append((idx, path_to_group[path]))

    rng = random.Random(seed)
    splits = {"train": [], "val": [], "test": []}
    quotas = {"train": 0.70, "val": 0.15, "test": 0.15}

    for class_idx in sorted(by_class):
        groups = collections.defaultdict(list)
        for idx, gid in by_class[class_idx]:
            groups[gid].append(idx)

        # Shuffle before the stable sort so equal-sized groups don't always land
        # in the same split across classes.
        ordered = list(groups.values())
        rng.shuffle(ordered)
        ordered.sort(key=len, reverse=True)

        n_class = len(by_class[class_idx])
        filled = {k: 0 for k in splits}

        for members in ordered:
            deficit = {k: quotas[k] * n_class - filled[k] for k in splits}
            target = max(deficit, key=deficit.get)
            splits[target].extend(members)
            filled[target] += len(members)

    for k in splits:
        rng.shuffle(splits[k])

    return splits["train"], splits["val"], splits["test"]


def build_datasets(data_root=DATA_ROOT, seed=SEED, grouped=False, threshold=0.02):
    """70/15/15 split into train/val/test.

    Two separate ImageFolder objects are built over the same root, then sliced
    with a shared permutation. This is deliberate: random_split hands back
    Subsets that all point at ONE underlying dataset, so setting
    `val_subset.dataset.transform` would also silently strip augmentation from
    the training split.
    """
    train_transforms, eval_transforms = build_transforms()

    train_base = ImageFolder(root=data_root, transform=train_transforms, loader=rgba_loader)
    eval_base = ImageFolder(root=data_root, transform=eval_transforms, loader=rgba_loader)

    if grouped:
        train_idx, val_idx, test_idx = grouped_split_indices(train_base.samples, threshold, seed)
    else:
        generator = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(train_base), generator=generator).tolist()

        n_total = len(perm)
        n_train = int(0.70 * n_total)
        n_val = int(0.15 * n_total)

        train_idx = perm[:n_train]
        val_idx = perm[n_train:n_train + n_val]
        test_idx = perm[n_train + n_val:]

    train_dataset = Subset(train_base, train_idx)
    val_dataset = Subset(eval_base, val_idx)
    test_dataset = Subset(eval_base, test_idx)

    return train_dataset, val_dataset, test_dataset, train_base.classes


def build_loaders(batch_size=32, num_workers=4, data_root=DATA_ROOT, seed=SEED,
                  grouped=False, threshold=0.02):
    train_dataset, val_dataset, test_dataset, classes = build_datasets(
        data_root, seed, grouped=grouped, threshold=threshold
    )

    def loader(dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            # Workers are forked processes; without this they respawn every epoch.
            persistent_workers=num_workers > 0,
        )

    return (
        loader(train_dataset, shuffle=True),
        loader(val_dataset, shuffle=False),
        loader(test_dataset, shuffle=False),
        classes,
    )


def get_device():
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


if __name__ == "__main__":
    print(f"Using device: {get_device()}")

    train_loader, val_loader, test_loader, classes = build_loaders()

    print(f"Classes ({len(classes)}): {classes}")
    print(f"Train: {len(train_loader.dataset)}")
    print(f"Val:   {len(val_loader.dataset)}")
    print(f"Test:  {len(test_loader.dataset)}")

    images, labels = next(iter(train_loader))
    print("Batch image tensor shape:", images.shape)
    print("Batch label tensor shape:", labels.shape)
    print(f"Pixel range after normalize: [{images.min():.2f}, {images.max():.2f}]")
