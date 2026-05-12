import json
import pathlib
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip  # openai/clip


# ── Dataset ───────────────────────────────────────────────────────────────────

class COCOCaptionDataset(Dataset):
    """Each item is one (image_path, caption) pair from COCO annotations."""

    def __init__(self, ann_file: pathlib.Path, img_dir: pathlib.Path, transform=None, max_samples: Optional[int] = None):
        with open(ann_file) as f:
            data = json.load(f)

        # build id → filename map
        id2file = {img["id"]: img["file_name"] for img in data["images"]}

        pairs = []
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            if img_id not in id2file:
                continue
            img_path = img_dir / id2file[img_id]
            if not img_path.exists():   # skip images not yet downloaded
                continue
            pairs.append((img_path, ann["caption"]))
            if max_samples and len(pairs) >= max_samples:
                break

        self.pairs = pairs
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, caption = self.pairs[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, caption


class EmbeddingDataset(Dataset):
    """Wraps pre-computed (img_emb, txt_emb) tensors for fast DataLoader iteration."""

    def __init__(self, img_emb: torch.Tensor, txt_emb: torch.Tensor):
        assert img_emb.size(0) == txt_emb.size(0)
        self.img_emb = img_emb
        self.txt_emb = txt_emb

    def __len__(self):
        return self.img_emb.size(0)

    def __getitem__(self, idx):
        return self.img_emb[idx], self.txt_emb[idx]


# ── Embedding cache ───────────────────────────────────────────────────────────

def cache_or_compute_embeddings(clip_model, preprocess, split: str, CONFIG: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Load cached embeddings if available, otherwise extract and save them.

    Args:
        clip_model:  CLIP model (already on device, in eval mode)
        preprocess:  CLIP image preprocessing transform
        split:       "train" or "val"
        CONFIG:      project CONFIG dict

    Returns:
        (img_emb, txt_emb) on CPU, shape (N, embed_dim)
    """
    cache_dir = CONFIG["cache_dir"]
    img_path = cache_dir / f"{split}_img_emb.pt"
    txt_path = cache_dir / f"{split}_txt_emb.pt"

    if img_path.exists() and txt_path.exists():
        print(f"[data_utils] Loading cached {split} embeddings …")
        return torch.load(img_path), torch.load(txt_path)

    print(f"[data_utils] Computing {split} embeddings …")
    device = CONFIG["device"]

    ann_file = CONFIG[f"coco_{split}_ann"]
    img_dir = CONFIG[f"coco_{split}_images"]
    max_samples = CONFIG.get(f"{split}_size")

    dataset = COCOCaptionDataset(ann_file, img_dir, transform=preprocess, max_samples=max_samples)
    loader = DataLoader(dataset, batch_size=256, num_workers=4, pin_memory=True, collate_fn=_collate)

    all_img, all_txt = [], []
    with torch.no_grad():
        for images, captions in loader:
            images = images.to(device)
            tokens = clip.tokenize(captions, truncate=True).to(device)
            img_emb = clip_model.encode_image(images).float()
            txt_emb = clip_model.encode_text(tokens).float()
            all_img.append(img_emb.cpu())
            all_txt.append(txt_emb.cpu())

    img_emb = torch.cat(all_img)
    txt_emb = torch.cat(all_txt)

    torch.save(img_emb, img_path)
    torch.save(txt_emb, txt_path)
    print(f"[data_utils] Saved to {img_path} and {txt_path}")
    return img_emb, txt_emb


def _collate(batch):
    images = torch.stack([b[0] for b in batch])
    captions = [b[1] for b in batch]
    return images, captions


def make_loader(img_emb: torch.Tensor, txt_emb: torch.Tensor, batch_size: int, shuffle: bool = True) -> DataLoader:
    ds = EmbeddingDataset(img_emb, txt_emb)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False)


# ── CIFAR-100 zero-shot helpers ───────────────────────────────────────────────

_CIFAR100_TEMPLATES = [
    'a photo of a {}.',
    'a blurry photo of a {}.',
    'a photo of the {}.',
    'a rendering of a {}.',
    'itap of a {}.',
    'a photo of a small {}.',
    'a photo of a large {}.',
]


def load_or_compute_cifar100_embs(
    clip_model,
    preprocess,
    cache_dir: pathlib.Path,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (img_embs, cls_embs, labels) for CIFAR-100 test split.

    img_embs : (10000, E)  L2-normalised CLIP image embeddings
    cls_embs : (100,   E)  L2-normalised template-ensemble class embeddings
    labels   : (10000,)    integer ground-truth class indices

    Results are cached under cache_dir so subsequent calls are instant.
    """
    import importlib
    torchvision = importlib.import_module('torchvision')
    import torch.nn.functional as _F

    img_path = cache_dir / 'cifar100_img_emb.pt'
    cls_path = cache_dir / 'cifar100_cls_emb.pt'
    lab_path = cache_dir / 'cifar100_labels.pt'

    if img_path.exists() and cls_path.exists() and lab_path.exists():
        print('[data_utils] Loading cached CIFAR-100 embeddings...')
        return torch.load(img_path), torch.load(cls_path), torch.load(lab_path)

    print('[data_utils] Computing CIFAR-100 CLIP embeddings (first time)...')
    cifar_root = '/tmp/cifar100'

    ds = torchvision.datasets.CIFAR100(cifar_root, train=False, download=True, transform=preprocess)
    loader = DataLoader(ds, batch_size=256, shuffle=False, num_workers=2)

    all_img, all_labels = [], []
    clip_model.eval()
    with torch.no_grad():
        for imgs, labs in loader:
            emb = clip_model.encode_image(imgs.to(device)).float()
            all_img.append(_F.normalize(emb, dim=-1).cpu())
            all_labels.append(labs)
    img_embs = torch.cat(all_img)
    labels   = torch.cat(all_labels)

    class_names = torchvision.datasets.CIFAR100(cifar_root, train=False, download=False).classes
    all_cls = []
    with torch.no_grad():
        for name in class_names:
            tokens = clip.tokenize([t.format(name) for t in _CIFAR100_TEMPLATES]).to(device)
            emb = clip_model.encode_text(tokens).float()
            all_cls.append(_F.normalize(emb.mean(0, keepdim=True), dim=-1).cpu())
    cls_embs = torch.cat(all_cls)

    torch.save(img_embs, img_path)
    torch.save(cls_embs, cls_path)
    torch.save(labels,   lab_path)
    print(f'[data_utils] Cached to {cache_dir}')
    return img_embs, cls_embs, labels
