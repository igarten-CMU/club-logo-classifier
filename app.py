"""Phase 4: Gradio demo.

Drag in a crest, a jersey photo or a broadcast screenshot and get the top-3
clubs with confidences.
"""

import argparse
import glob
import os

import gradio as gr
import torch

from data import build_transforms, get_device
from train import build_model

# Defaults to the group-aware model: same accuracy, trained on the split with
# near-duplicate leakage removed. Override with CHECKPOINT=... python app.py
CHECKPOINT = os.environ.get("CHECKPOINT", "checkpoints/best_grouped.pt")

device = get_device()
_, eval_transforms = build_transforms()

ckpt = torch.load(CHECKPOINT, map_location=device, weights_only=False)
CLASSES = ckpt["classes"]

model = build_model(len(CLASSES), unfreeze=ckpt.get("unfreeze", False)).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# "manchester-united" -> "Manchester United"
PRETTY = {c: c.replace("-", " ").title() for c in CLASSES}


@torch.no_grad()
def predict(image):
    if image is None:
        return {}
    tensor = eval_transforms(image).unsqueeze(0).to(device)
    probs = torch.softmax(model(tensor), dim=1)[0].cpu()
    return {PRETTY[c]: float(p) for c, p in zip(CLASSES, probs)}


demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Crest, jersey photo or screenshot"),
    outputs=gr.Label(num_top_classes=3, label="Predicted club"),
    title="Premier League Crest Classifier",
    description=(
        "EfficientNet-B0 fine-tuned on 20,000 crest images across 20 Premier "
        "League clubs (2021/22 season). Trained on transparent-background crest "
        "renders composited onto random backgrounds, so it is most confident on "
        "clean crest images."
    ),
    examples=sorted(glob.glob("assets/samples/*.png")),
    flagging_mode="never",
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    args = parser.parse_args()
    demo.launch(share=args.share)
