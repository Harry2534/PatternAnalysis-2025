import os
import re
import numpy as np
import torch
from PIL import Image
# from modules import ImprovedUNet3D
import importlib.util, pathlib
_here = pathlib.Path(__file__).parent
_spec_mod = importlib.util.spec_from_file_location("modules3d", _here / "3Dmodules.py")
_modules3d = importlib.util.module_from_spec(_spec_mod); _spec_mod.loader.exec_module(_modules3d)
ImprovedUNet3D = _modules3d.ImprovedUNet3D

device = "cuda" if torch.cuda.is_available() else "cpu"

TEST_IMG_DIR = "OASIS/keras_png_slices_test"
TEST_MSK_DIR = "OASIS/keras_png_slices_seg_test"
CKPT_PATH = "saved_models/improved_unet3d.pth"
NUM_CLASSES = 4

CASE_RE = re.compile(r"^case_(\d+)_slice_(\d+)(?:\.nii)?\.png$")
SEG_RE  = re.compile(r"^seg_(\d+)_slice_(\d+)(?:\.nii)?\.png$")


def group_cases(img_dir: str, msk_dir: str):
    imgs = {}
    for f in os.listdir(img_dir):
        if not f.endswith(".png"): continue
        m = CASE_RE.match(f)
        if not m: continue
        cid, s = m.group(1), int(m.group(2))
        imgs.setdefault(cid, []).append((s, os.path.join(img_dir, f)))

    msks = {}
    for f in os.listdir(msk_dir):
        if not f.endswith(".png"): continue
        m = SEG_RE.match(f)
        if not m: continue
        cid, s = m.group(1), int(m.group(2))
        msks.setdefault(cid, []).append((s, os.path.join(msk_dir, f)))

    common = sorted(set(imgs.keys()) & set(msks.keys()), key=lambda x: int(x))
    cases = []
    for cid in common:
        si = sorted(imgs[cid], key=lambda t: t[0])
        sm = sorted(msks[cid], key=lambda t: t[0])
        idx = sorted(set([i for i,_ in si]) & set([i for i,_ in sm]))
        if not idx: continue
        imap = {i:p for i,p in si}
        mmap = {i:p for i,p in sm}
        cases.append((cid, [imap[i] for i in idx], [mmap[i] for i in idx]))
    return cases


def load_volume(paths):
    return np.stack([np.array(Image.open(p).convert('L'), dtype=np.float32) for p in paths], axis=0)  # [D,H,W]


@torch.no_grad()
def macro_dice(pred: np.ndarray, target: np.ndarray, num_classes: int, eps: float = 1e-6) -> float:
    s = 0.0
    for c in range(num_classes):
        p = (pred == c)
        t = (target == c)
        inter = np.logical_and(p, t).sum()
        denom = p.sum() + t.sum()
        s += (2.0 * inter + eps) / (denom + eps)
    return s / num_classes


def main():
    # Load model
    ckpt = torch.load(CKPT_PATH, map_location=device)
    state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    model = ImprovedUNet3D(n_channels=1, n_classes=NUM_CLASSES, base_ch=16, trilinear=True).to(device)
    model.load_state_dict(state)
    model.eval()

    cases = group_cases(TEST_IMG_DIR, TEST_MSK_DIR)
    if len(cases) == 0:
        print("No test cases found."); return

    dice_scores = []
    for cid, ipaths, mpaths in cases:
        img = load_volume(ipaths).astype(np.float32) / 255.0
        img = (img - 0.5) / 0.5
        msk = load_volume(mpaths).astype(np.int64)

        # optional remap to 0..K-1
        vals = np.unique(msk)
        if not np.array_equal(vals, np.arange(vals.size)):
            remap = {v: i for i, v in enumerate(vals)}
            msk = np.vectorize(remap.get)(msk).astype(np.int64)

        inp = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(device)  # [1,1,D,H,W]
        logits = model(inp)
        pred = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()      # [D,H,W]

        d = macro_dice(pred, msk, NUM_CLASSES)
        dice_scores.append(d)
        print(f"Case {cid}: Dice={d:.4f}")

    print(f"Evaluated {len(dice_scores)} cases.")
    print(f"Mean Dice: {np.mean(dice_scores):.4f}")


if __name__ == "__main__":
    main()