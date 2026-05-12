import math
import os
import pathlib

# ── Colab detection & Drive mount ────────────────────────────────────────────
_in_colab = "COLAB_RELEASE_TAG" in os.environ or os.path.exists("/content")

if _in_colab:
    from google.colab import drive
    drive.mount("/content/drive", force_remount=False)
    _base = pathlib.Path("/content/drive/MyDrive/sparse_clip")
else:
    _base = pathlib.Path(__file__).resolve().parents[1]  # project root

_cache_dir = _base / "cache"
_results_dir = _base / "results"
_cache_dir.mkdir(parents=True, exist_ok=True)
_results_dir.mkdir(parents=True, exist_ok=True)
(_results_dir / "variation1").mkdir(exist_ok=True)
(_results_dir / "variation2").mkdir(exist_ok=True)

# ── COCO paths ────────────────────────────────────────────────────────────────
# In Colab the dataset lives on Drive; locally set COCO_ROOT or drop it under
# the project root.  Drive path: MyDrive/sparse_clip/coco/
_default_coco = _base / "coco"
_coco_root = pathlib.Path(os.environ.get("COCO_ROOT", str(_default_coco)))

CONFIG = {
    # ── Environment ────────────────────────────────────────────────────────
    "in_colab": _in_colab,
    "cache_dir": _cache_dir,
    "results_dir": _results_dir,

    # ── CLIP backbone ───────────────────────────────────────────────────────
    "clip_model": "ViT-B/32",
    "embed_dim": 512,

    # ── SparseHead architecture ─────────────────────────────────────────────
    "sparse_dim": 16384,   # 32× expansion from 512

    # ── COCO dataset ────────────────────────────────────────────────────────
    "coco_root": _coco_root,
    "coco_train_images": _coco_root / "train2017",
    "coco_val_images": _coco_root / "val2017",
    "coco_train_ann": _coco_root / "annotations" / "captions_train2017.json",
    "coco_val_ann": _coco_root / "annotations" / "captions_val2017.json",
    "train_size": None,   # None = use all 118K pairs
    "val_size": 5000,

    # ── Training ────────────────────────────────────────────────────────────
    "batch_size": 256,
    "lr": 1e-3,
    "epochs": 15,
    "weight_decay": 1e-4,
    # Learnable logit scale: log(1/τ), init equivalent to τ=0.07
    "log_scale_init": math.log(1 / 0.07),
    "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",

    # ── Metrics ─────────────────────────────────────────────────────────────
    "retrieval_k": 1,       # Recall@K used in evaluate()
    "clarity_tau": 0.001,   # τ threshold for feature activation (paper: 0.001)
    "clarity_n_min": 2,     # min activating images per feature (paper: n_min=2)
}

if __name__ == "__main__":
    print(f"[config] base      : {_base}")
    print(f"[config] cache_dir : {CONFIG['cache_dir']}")
    print(f"[config] coco_root : {CONFIG['coco_root']}")
    print(f"[config] results   : {CONFIG['results_dir']}")
    print(f"[config] device    : {CONFIG['device']}")
