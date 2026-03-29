"""
Training script for the image captioning model.
Handles: data loading, CLIP pre-computation, training loop,
validation, checkpointing, and sample generation per epoch.
"""
import os
import sys
import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import (
    CLIPModel, CLIPProcessor,
    GPT2LMHeadModel, GPT2Tokenizer,
    get_linear_schedule_with_warmup,
)
from peft import LoraConfig, get_peft_model, TaskType
from tqdm.auto import tqdm

# Add src/ to path when running directly
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    DEVICE, CLIP_MODEL_NAME, GPT2_TOKENIZER_NAME,
    DATASET_NAME, SUBSET_SIZE, RANDOM_SEED,
    MAX_CAPTION_LENGTH, OUTPUT_DIR,
)
from data_preprocessing import load_and_preprocess
from model import MappingNetwork, CaptioningModel
from dataset import CaptionDataset


# ── Training config ────────────────────────────────────
class TrainConfig:
    # Data
    train_size = 2500
    val_size = 500

    # Architecture
    clip_dim = 512
    gpt2_dim = 768
    prefix_length = 10
    mapping_hidden_dim = 512

    # LoRA
    lora_r = 8
    lora_alpha = 16
    lora_dropout = 0.05

    # Training
    epochs = 5
    batch_size = 16
    learning_rate = 2e-4
    weight_decay = 0.01
    warmup_steps = 100
    grad_accum_steps = 2
    max_grad_norm = 1.0

    # Paths
    checkpoint_dir = OUTPUT_DIR / "checkpoints"
    log_path = OUTPUT_DIR / "training_log.json"


cfg = TrainConfig()
cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)


# ── CLIP embedding pre-computation ─────────────────────
def precompute_clip_embeddings(data, clip_model, clip_processor, device):
    """Encode all images through CLIP once upfront."""
    clip_model.eval()
    embeddings = []

    print("Pre-computing CLIP embeddings...")
    for item in tqdm(data, desc="CLIP encoding"):
        inputs = clip_processor(images=item["image"], return_tensors="pt").to(device)
        with torch.no_grad():
            vision_out = clip_model.vision_model(**inputs)
            emb = clip_model.visual_projection(vision_out.pooler_output)
        embeddings.append(emb.squeeze().cpu())

    return torch.stack(embeddings)


# ── Main training function ─────────────────────────────
def train():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    print(f"Device: {DEVICE}")
    print("=" * 60)

    # ── Load CLIP (frozen, just for embedding pre-computation) ──
    print("Loading CLIP...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME).to(DEVICE)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    # ── Load GPT-2 with LoRA ──
    print("Loading GPT-2 with LoRA adapters...")
    gpt2_model = GPT2LMHeadModel.from_pretrained(GPT2_TOKENIZER_NAME)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=["c_attn", "c_proj"],
    )
    gpt2_model = get_peft_model(gpt2_model, lora_config)
    gpt2_model.print_trainable_parameters()

    # ── Tokenizer ──
    tokenizer = GPT2Tokenizer.from_pretrained(GPT2_TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token

    # ── Mapping network ──
    mapping_net = MappingNetwork(
        clip_dim=cfg.clip_dim,
        gpt2_dim=cfg.gpt2_dim,
        prefix_length=cfg.prefix_length,
        hidden_dim=cfg.mapping_hidden_dim,
    )

    # ── Full model ──
    model = CaptioningModel(
        gpt2_model, mapping_net, cfg.prefix_length, tokenizer.eos_token_id
    ).to(DEVICE)

    # ── Load data ──
    print("\nLoading training data...")
    train_data = load_and_preprocess(subset_size=cfg.train_size)

    # Load validation from the validation split
    from datasets import load_dataset as hf_load
    print("Loading validation data...")
    val_raw = hf_load(DATASET_NAME, split="validation")
    random.seed(RANDOM_SEED + 1)
    val_indices = random.sample(range(len(val_raw)), min(cfg.val_size * 2, len(val_raw)))

    from data_preprocessing import download_image, clean_caption, is_valid_caption
    val_data = []
    for idx in tqdm(val_indices, desc="Val preprocessing"):
        if len(val_data) >= cfg.val_size:
            break
        item = val_raw[idx]
        captions = item.get("sentences", [])
        url = item.get("url", "")
        if not captions or not url:
            continue
        cleaned = [clean_caption(c) for c in captions if isinstance(c, str)]
        cleaned = [c for c in cleaned if is_valid_caption(c)]
        if not cleaned:
            continue
        image = download_image(url)
        if image is None:
            continue
        val_data.append({"image": image, "captions": cleaned})

    print(f"Train: {len(train_data)} | Val: {len(val_data)}")

    # ── Pre-compute CLIP embeddings ──
    train_clip_embs = precompute_clip_embeddings(train_data, clip_model, clip_processor, DEVICE)
    val_clip_embs = precompute_clip_embeddings(val_data, clip_model, clip_processor, DEVICE)

    # Free CLIP from GPU memory — we don't need it during training
    del clip_model
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # ── Datasets and loaders ──
    train_captions = [item["captions"] for item in train_data]
    val_captions = [item["captions"] for item in val_data]

    train_dataset = CaptionDataset(train_clip_embs, train_captions, tokenizer, MAX_CAPTION_LENGTH)
    val_dataset = CaptionDataset(val_clip_embs, val_captions, tokenizer, MAX_CAPTION_LENGTH)

    train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    # ── Optimizer and scheduler ──
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )

    total_steps = (len(train_loader) // cfg.grad_accum_steps) * cfg.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, cfg.warmup_steps, total_steps)

    # ── Training loop ──
    training_log = []
    best_val_loss = float("inf")

    print(f"\nTraining for {cfg.epochs} epochs ({total_steps} optimizer steps)")
    print(f"Effective batch size: {cfg.batch_size * cfg.grad_accum_steps}")
    print("=" * 60)

    for epoch in range(cfg.epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{cfg.epochs} [Train]")
        for step, (clip_emb, input_ids, attn_mask) in enumerate(pbar):
            clip_emb = clip_emb.to(DEVICE)
            input_ids = input_ids.to(DEVICE)
            attn_mask = attn_mask.to(DEVICE)

            loss, _ = model(clip_emb, input_ids, attn_mask)
            loss = loss / cfg.grad_accum_steps
            loss.backward()

            if (step + 1) % cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            epoch_loss += loss.item() * cfg.grad_accum_steps
            pbar.set_postfix(loss=f"{loss.item() * cfg.grad_accum_steps:.4f}")

        avg_train_loss = epoch_loss / len(train_loader)

        # --- Validate ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for clip_emb, input_ids, attn_mask in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]"):
                clip_emb = clip_emb.to(DEVICE)
                input_ids = input_ids.to(DEVICE)
                attn_mask = attn_mask.to(DEVICE)
                loss, _ = model(clip_emb, input_ids, attn_mask)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        # --- Sample captions (3 strategies on first val image) ---
        sample_emb = val_clip_embs[0].to(DEVICE)
        greedy_cap = model.generate_greedy(sample_emb, tokenizer)
        beam_cap = model.generate_beam(sample_emb, tokenizer, beam_width=5)
        nucleus_cap = model.generate_nucleus(sample_emb, tokenizer, temperature=0.8, top_p=0.9)

        # --- Log ---
        epoch_log = {
            "epoch": epoch + 1,
            "train_loss": round(avg_train_loss, 4),
            "val_loss": round(avg_val_loss, 4),
            "lr": scheduler.get_last_lr()[0],
            "sample_gt": val_captions[0][0],
            "sample_greedy": greedy_cap,
            "sample_beam": beam_cap,
            "sample_nucleus": nucleus_cap,
        }
        training_log.append(epoch_log)

        print(f"\nEpoch {epoch+1}: Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f}")
        print(f"  GT:      \"{val_captions[0][0]}\"")
        print(f"  Greedy:  \"{greedy_cap}\"")
        print(f"  Beam:    \"{beam_cap}\"")
        print(f"  Nucleus: \"{nucleus_cap}\"")

        # --- Checkpoint ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_loss": avg_val_loss,
            }, cfg.checkpoint_dir / "best_model.pt")
            print(f"  ✓ Saved best model (val_loss={avg_val_loss:.4f})")

    # Save training log
    with open(cfg.log_path, "w") as f:
        json.dump(training_log, f, indent=2)
    print(f"\n✓ Training log saved → {cfg.log_path}")

    return model, tokenizer, val_data, val_clip_embs, val_captions, training_log


if __name__ == "__main__":
    train()