# Image Captioning — Generative AI Project

## Overview
Image captioning pipeline using **CLIP** (image encoder) and **GPT-2** (caption tokenizer/decoder), built on the **COCO Captions** dataset.

## Setup

### Prerequisites
- Python 3.10+
- macOS with Apple Silicon (M1/M2/M3) — uses MPS acceleration
- ~4 GB disk space for model weights and dataset cache

### Installation
```bash
# Clone the repo
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Project Structure
```
├── README.md
├── requirements.txt
├── src/
│   ├── config.py                 # Paths, model names, device setup
│   ├── data_preprocessing.py     # Dataset loading and cleaning
│   ├── embedding_pipeline.py     # CLIP embedder + GPT-2 tokenizer
│   └── run_samples.py            # Generate 5 sample test runs
├── data/
│   └── hf_cache/                 # HuggingFace download cache (auto-created)
├── outputs/
│   └── sample_test_runs.png      # Visual grid of test samples
└── docs/
    └── proposal.md               # Project proposal document
```

## Usage

### Quick test (5 samples)
```bash
cd src
python run_samples.py
```
This will:
1. Download and preprocess a 2500-pair subset of COCO Captions
2. Generate CLIP embeddings for 5 sample images
3. Tokenize captions using GPT-2
4. Save a visual summary grid to `outputs/sample_test_runs.png`

### Test individual components
```bash
# Test data preprocessing only
python data_preprocessing.py

# Test embedding pipeline only
python embedding_pipeline.py
```

## Tech Stack
| Component | Purpose |
|-----------|---------|
| CLIP (`clip-vit-base-patch32`) | Image → 512-dim embedding |
| GPT-2 Tokenizer | Caption → token IDs |
| COCO Captions (2500 subset) | Image-caption pair dataset |
| PyTorch (MPS backend) | Tensor computation on Apple Silicon |
| HuggingFace Transformers | Model loading and inference |

## Team
- *[Add team members and roles here]*

## License
MIT
