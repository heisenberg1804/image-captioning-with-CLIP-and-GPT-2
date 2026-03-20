"""
CLIP image embeddings + GPT-2 caption tokenization.
Core pipeline that Milestone 1 asks you to demonstrate.
"""
import torch
from transformers import CLIPModel, CLIPProcessor, GPT2Tokenizer
from PIL import Image
from config import CLIP_MODEL_NAME, GPT2_TOKENIZER_NAME, MAX_CAPTION_LENGTH, DEVICE


class ImageEmbedder:
    """Wraps CLIP to produce image embedding vectors."""

    def __init__(self, model_name: str = CLIP_MODEL_NAME, device: torch.device = DEVICE):
        self.device = device
        print(f"Loading CLIP model on {device}...")
        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()  # Inference mode — saves memory

    @torch.no_grad()
    def embed(self, image: Image.Image) -> torch.Tensor:
        """Single image → embedding vector (shape: [512])."""
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        # Use the full model's vision encoder to get the pooled [CLS] embedding
        vision_outputs = self.model.vision_model(**inputs)
        pooled = self.model.visual_projection(vision_outputs.pooler_output)
        return pooled.squeeze().cpu()  # Shape: [512]

    @torch.no_grad()
    def embed_batch(self, images: list[Image.Image], batch_size: int = 16) -> torch.Tensor:
        """List of images → stacked embeddings (shape: [N, 512])."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            vision_outputs = self.model.vision_model(**inputs)
            pooled = self.model.visual_projection(vision_outputs.pooler_output)
            all_embeddings.append(pooled.cpu())
        return torch.cat(all_embeddings, dim=0)


class CaptionTokenizer:
    """Wraps GPT-2 tokenizer for caption encoding/decoding."""

    def __init__(self, model_name: str = GPT2_TOKENIZER_NAME, max_length: int = MAX_CAPTION_LENGTH):
        print("Loading GPT-2 tokenizer...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # GPT-2 has no pad token — set it to eos to avoid warnings
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_length = max_length

    def tokenize(self, caption: str) -> dict:
        """Caption string → token IDs + attention mask."""
        return self.tokenizer(
            caption,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def tokenize_batch(self, captions: list[str]) -> dict:
        """List of captions → batched token IDs + attention masks."""
        return self.tokenizer(
            captions,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

    def decode(self, token_ids: torch.Tensor) -> str:
        """Token IDs back to readable string."""
        return self.tokenizer.decode(token_ids.squeeze(), skip_special_tokens=True)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.vocab_size

    @property
    def eos_token_id(self) -> int:
        return self.tokenizer.eos_token_id


# ── CLI quick-test ─────────────────────────────────────
if __name__ == "__main__":
    from data_preprocessing import load_and_preprocess

    data = load_and_preprocess(subset_size=5)

    embedder = ImageEmbedder()
    tokenizer = CaptionTokenizer()

    for i, item in enumerate(data):
        emb = embedder.embed(item["image"])
        tokens = tokenizer.tokenize(item["captions"][0])
        decoded = tokenizer.decode(tokens["input_ids"])

        print(f"\n--- Sample {i+1} ---")
        print(f"  Caption      : {item['captions'][0]}")
        print(f"  Embedding    : shape={emb.shape}, norm={emb.norm():.4f}")
        print(f"  Token count  : {tokens['attention_mask'].sum().item()}")
        print(f"  Decoded back : {decoded}")