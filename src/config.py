"""
Project configuration — central place for all paths, model names, and hyperparams.
"""
from pathlib import Path
import torch

# ── Paths ──────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CACHE_DIR = DATA_DIR / "hf_cache"          # HuggingFace download cache
OUTPUT_DIR = PROJECT_ROOT / "outputs"
DATA_DIR.mkdir(exist_ok=True)
CACHE_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Dataset ────────────────────────────────────────────
# Using yerevann/coco-karpathy — parquet-native, no script issues
DATASET_NAME = "yerevann/coco-karpathy"
DATASET_SPLIT = "train"
SUBSET_SIZE = 2500                          # Number of image-caption pairs
RANDOM_SEED = 42

# ── Models ─────────────────────────────────────────────
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
GPT2_TOKENIZER_NAME = "gpt2"
MAX_CAPTION_LENGTH = 64                     # Max tokens per caption

# ── Device (MPS for M-series Macs, fallback to CPU) ───
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

DEVICE = get_device()