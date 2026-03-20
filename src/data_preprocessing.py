"""
Dataset download, cleaning, and subset creation.
Uses yerevann/coco-karpathy (parquet-native, no script issues).
Images are downloaded from COCO URLs on the fly.
"""
import re
import random
import io
import urllib.request
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from config import DATASET_NAME, DATASET_SPLIT, SUBSET_SIZE, RANDOM_SEED, CACHE_DIR


# ── Text cleaning ──────────────────────────────────────
def clean_caption(caption: str) -> str:
    caption = caption.lower().strip()
    caption = re.sub(r'[^a-zA-Z0-9\s.,!?\'\-]', '', caption)
    caption = re.sub(r'\s+', ' ', caption)
    return caption


def is_valid_caption(caption: str, min_words: int = 4, max_words: int = 60) -> bool:
    word_count = len(caption.split())
    return min_words <= word_count <= max_words


# ── Image downloading ──────────────────────────────────
def download_image(url: str, timeout: int = 10) -> Image.Image | None:
    """Download an image from a URL and return as PIL Image."""
    try:
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (dataset-download)"}
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            image_data = response.read()
        img = Image.open(io.BytesIO(image_data))
        if img.mode != "RGB":
            img = img.convert("RGB")
        return img
    except Exception as e:
        return None


# ── Main loader ────────────────────────────────────────
def load_and_preprocess(subset_size: int = SUBSET_SIZE) -> list[dict]:
    """
    Load COCO Karpathy split from HuggingFace (parquet),
    download images, clean captions, return processed pairs.
    """
    print(f"Loading {DATASET_NAME} ({DATASET_SPLIT} split)...")
    dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, cache_dir=str(CACHE_DIR))
    print(f"Full dataset size: {len(dataset)}")

    # Reproducible random subset — sample extra to account for download failures
    random.seed(RANDOM_SEED)
    pool_size = min(subset_size * 2, len(dataset))
    indices = random.sample(range(len(dataset)), pool_size)

    processed = []
    skipped = 0

    print(f"Downloading images and preprocessing (target: {subset_size} pairs)...")
    for idx in tqdm(indices, desc="Processing"):
        if len(processed) >= subset_size:
            break

        item = dataset[idx]

        # --- Captions ---
        # Field is 'sentences' — a list of caption strings
        raw_captions = item.get("sentences", [])
        if isinstance(raw_captions, str):
            raw_captions = [raw_captions]

        cleaned = [clean_caption(c) for c in raw_captions if isinstance(c, str)]
        cleaned = [c for c in cleaned if is_valid_caption(c)]

        if not cleaned:
            skipped += 1
            continue

        # --- Image (download from URL) ---
        url = item.get("url", "")
        if not url:
            skipped += 1
            continue

        image = download_image(url)
        if image is None:
            skipped += 1
            continue

        # Basic size check
        w, h = image.size
        if w < 64 or h < 64:
            skipped += 1
            continue

        processed.append({
            "image": image,
            "captions": cleaned,
            "cocoid": item.get("cocoid", None),
            "url": url,
        })

    print(f"\nDone — {len(processed)} valid pairs ({skipped} skipped)")
    return processed


# ── CLI quick-test ─────────────────────────────────────
if __name__ == "__main__":
    data = load_and_preprocess(subset_size=10)  # small test
    for i, item in enumerate(data[:3]):
        print(f"\nSample {i+1}:")
        print(f"  Image size : {item['image'].size}")
        print(f"  COCO ID    : {item['cocoid']}")
        print(f"  Captions   : {item['captions'][:2]}")