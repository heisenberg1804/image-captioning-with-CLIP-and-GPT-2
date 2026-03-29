"""
PyTorch Dataset for training: pairs pre-computed CLIP embeddings with tokenized captions.
"""
import random
import torch
from torch.utils.data import Dataset


class CaptionDataset(Dataset):
    """
    Each sample returns (clip_embedding, input_ids, attention_mask).
    Randomly picks one of the multiple captions per image each time
    for data augmentation.
    """
    def __init__(self, clip_embeddings, captions_list, tokenizer, max_length=64):
        """
        Args:
            clip_embeddings: Tensor [N, 512] — pre-computed CLIP embeddings
            captions_list: list of lists — multiple captions per image
            tokenizer: GPT2Tokenizer instance
            max_length: max token sequence length
        """
        self.clip_embeddings = clip_embeddings
        self.captions_list = captions_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.clip_embeddings)

    def __getitem__(self, idx):
        emb = self.clip_embeddings[idx]

        # Randomly select one of the captions (different each epoch = augmentation)
        captions = self.captions_list[idx]
        caption = random.choice(captions)

        # Tokenize: <caption> <eos>
        tokens = self.tokenizer(
            caption + self.tokenizer.eos_token,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        input_ids = tokens["input_ids"].squeeze()           # [max_length]
        attention_mask = tokens["attention_mask"].squeeze()  # [max_length]

        return emb, input_ids, attention_mask