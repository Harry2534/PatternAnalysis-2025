import os
import re
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from typing import Dict, List, Tuple

CASE_RE = re.compile(r"^case_(\d+)_slice_(\d+)(?:\.nii)?\.png$")
SEG_RE = re.compile(r"^seg_(\d+)_slice_(\d+)(?:\.nii)?\.png$")


def _group_slices_by_case(image_dir: str, mask_dir: str) -> List[Tuple[str, List[str], List[str]]]:
    img_cases: Dict[str, List[Tuple[int, str]]] = {}
    for f in os.listdir(image_dir):
        if not f.endswith(".png"):
            continue
        m = CASE_RE.match(f)
        if not m:
            continue
        cid, s = m.group(1), int(m.group(2))
        img_cases.setdefault(cid, []).append((s, os.path.join(image_dir, f)))

    mask_cases: Dict[str, List[Tuple[int, str]]] = {}
    for f in os.listdir(mask_dir):
        if not f.endswith(".png"):
            continue
        m = SEG_RE.match(f)
        if not m:
            continue
        cid, s = m.group(1), int(m.group(2))
        mask_cases.setdefault(cid, []).append((s, os.path.join(mask_dir, f)))

    common = sorted(set(img_cases.keys()) & set(mask_cases.keys()), key=lambda x: int(x))
    grouped: List[Tuple[str, List[str], List[str]]] = []
    for cid in common:
        imgs = sorted(img_cases[cid], key=lambda t: t[0])
        msks = sorted(mask_cases[cid], key=lambda t: t[0])
        img_idx = [i for i, _ in imgs]
        msk_idx = [i for i, _ in msks]
        idx = sorted(set(img_idx) & set(msk_idx))
        if not idx:
            continue
        imap = {i: p for i, p in imgs}
        mmap = {i: p for i, p in msks}
        img_paths = [imap[i] for i in idx]
        msk_paths = [mmap[i] for i in idx]
        grouped.append((cid, img_paths, msk_paths))
    return grouped


class OASIS3DDataset(Dataset):
    """
    Returns:
      image: FloatTensor [1, D, H, W] in approx [-1,1]
      mask:  LongTensor  [D, H, W] with class indices
      case_id: str
    """
    def __init__(self, image_dir: str, mask_dir: str):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.cases = _group_slices_by_case(image_dir, mask_dir)
        if len(self.cases) == 0:
            raise RuntimeError(f"No matched cases in\n images: {image_dir}\n masks:  {mask_dir}")

    def __len__(self):
        return len(self.cases)

    def _load_stack(self, paths: List[str]) -> np.ndarray:
        vol = np.stack([np.array(Image.open(p).convert('L'), dtype=np.float32) for p in paths], axis=0)  # [D,H,W]
        return vol

    def _normalize(self, vol: np.ndarray) -> np.ndarray:
        vol = vol / 255.0
        vol = (vol - 0.5) / 0.5
        return vol

    def _remap_mask(self, vol: np.ndarray) -> np.ndarray:
        vals = np.unique(vol)
        if not np.array_equal(vals, np.arange(vals.size)):
            remap = {v: i for i, v in enumerate(vals)}
            vol = np.vectorize(remap.get)(vol)
        return vol.astype(np.int64)

    def __getitem__(self, idx: int):
        case_id, img_paths, msk_paths = self.cases[idx]
        img_vol = self._load_stack(img_paths)                    # [D,H,W], float32
        msk_vol = self._load_stack(msk_paths).astype(np.int64)   # [D,H,W]
        msk_vol = self._remap_mask(msk_vol)
        img_vol = self._normalize(img_vol)

        img = torch.from_numpy(img_vol).unsqueeze(0).float()     # [1,D,H,W]
        msk = torch.from_numpy(msk_vol).long()                    # [D,H,W]
        return {"image": img, "mask": msk, "case_id": case_id}


def get_loaders_3d(
    train_img_dir: str,
    train_mask_dir: str,
    val_img_dir: str,
    val_mask_dir: str,
    batch_size: int = 1,
    num_workers: int = 0,
):
    train_ds = OASIS3DDataset(train_img_dir, train_mask_dir)
    val_ds = OASIS3DDataset(val_img_dir, val_mask_dir)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader