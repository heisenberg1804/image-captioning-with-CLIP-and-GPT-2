# Project Proposal — Image Captioning with Generative AI

## 1. Objective
Build an image captioning pipeline that leverages **CLIP** for visual feature extraction and **GPT-2** for caption generation/tokenization, trained and evaluated on COCO Captions.

## 2. Problem Statement
Automatic image captioning bridges computer vision and natural language processing. Given an image, the system should produce a concise, accurate natural-language description. This project focuses on building the encoding and tokenization pipeline as a foundation for future caption generation.

## 3. Methodology
1. **Image Encoding** — Use a pre-trained CLIP model (`clip-vit-base-patch32`) to extract a 512-dimensional embedding from each image.
2. **Caption Tokenization** — Use a GPT-2 tokenizer to convert ground-truth captions into token IDs suitable for language model training.
3. **Dataset** — Use a 2,500-pair subset of COCO Captions loaded via HuggingFace Datasets.
4. **Hardware** — Leverage Apple Silicon MPS acceleration for efficient local inference.

## 4. Expected Deliverables
- Cleaned and preprocessed COCO Captions subset
- CLIP embedding generation pipeline
- GPT-2 caption tokenization pipeline
- Visual summary grid of sample test runs
- Project documentation and reproducible setup instructions

## 5. Timeline
| Week | Milestone |
|------|-----------|
| 1 | Project setup, environment configuration, data loading |
| 2 | CLIP embedding pipeline implementation |
| 3 | GPT-2 tokenization and integration |
| 4 | Sample test runs, visualization, documentation |

## 6. Team
- *[Add team members and roles here]*

## 7. References
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)
- [GPT-2: Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [COCO Captions Dataset](https://cocodataset.org/)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers/)
