"""
Milestone 1 deliverable: Run 5 image-caption test samples.
Generates a visual grid + printed summary for submission.
"""
import matplotlib.pyplot as plt

from config import DEVICE, OUTPUT_DIR, SUBSET_SIZE
from data_preprocessing import load_and_preprocess
from embedding_pipeline import ImageEmbedder, CaptionTokenizer


def run_test_samples(n_samples: int = 5):
    """Run the full pipeline on n_samples and save results."""

    print(f"Device: {DEVICE}")
    print("=" * 60)

    # 1. Load data
    data = load_and_preprocess(subset_size=SUBSET_SIZE)

    # 2. Initialize models
    embedder = ImageEmbedder()
    tokenizer = CaptionTokenizer()

    # 3. Process samples
    samples = data[:n_samples]
    results = []

    print("\n" + "=" * 60)
    print(f"PROCESSING {n_samples} TEST SAMPLES")
    print("=" * 60)

    for i, item in enumerate(samples):
        image = item["image"]
        caption = item["captions"][0]

        # Image → CLIP embedding
        embedding = embedder.embed(image)

        # Caption → GPT-2 tokens
        tokens = tokenizer.tokenize(caption)
        token_count = tokens["attention_mask"].sum().item()
        decoded = tokenizer.decode(tokens["input_ids"])

        result = {
            "index": i + 1,
            "image": image,
            "caption": caption,
            "cocoid": item.get("cocoid", "N/A"),
            "embedding_shape": tuple(embedding.shape),
            "embedding_norm": embedding.norm().item(),
            "token_count": int(token_count),
            "decoded": decoded,
        }
        results.append(result)

        # Print summary
        print(f"\n--- Sample {i + 1} (COCO ID: {item.get('cocoid', 'N/A')}) ---")
        print(f"  Caption         : {caption}")
        print(f"  Embedding shape : {embedding.shape}")
        print(f"  Embedding norm  : {embedding.norm():.4f}")
        print(f"  Token count     : {token_count}")
        print(f"  Roundtrip check : {'PASS' if decoded.strip() == caption.strip() else 'DIFF'}")

    # 4. Save visual grid
    save_sample_grid(results)

    return results


def save_sample_grid(results: list[dict], filename: str = "sample_test_runs.png"):
    """Create a 1×5 image grid with captions — good for the submission."""
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5))

    if n == 1:
        axes = [axes]

    for ax, res in zip(axes, results):
        ax.imshow(res["image"])
        ax.set_title(
            f"Sample {res['index']}\n"
            f"Emb: {res['embedding_shape']} | Tokens: {res['token_count']}",
            fontsize=9
        )
        # Show truncated caption below
        cap_display = res["caption"][:60] + ("..." if len(res["caption"]) > 60 else "")
        ax.set_xlabel(cap_display, fontsize=8, wrap=True)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout()
    out_path = OUTPUT_DIR / filename
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved sample grid → {out_path}")


if __name__ == "__main__":
    run_test_samples()