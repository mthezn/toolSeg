# toolSeg — Surgical Tool Segmentation via Knowledge Distillation from SAM

This repository contains the code developed for a thesis project at **Politecnico di Milano** focused on **binary segmentation of surgical instruments** in laparoscopic video frames. The core idea is to distill the capabilities of the large **Segment Anything Model (SAM ViT-H)** into a compact, efficient student model suitable for real-time inference.

---

## Overview

The project follows a **three-stage training pipeline**:

### Stage 1 — Decoupled Encoder Distillation (`mainDecoupled.py`)
The student model's image encoder (based on a lightweight CMT architecture) is trained to mimic the feature representations produced by the SAM ViT-H encoder. At this stage only the encoder is trained, while the decoder is kept frozen. The loss is **MSE** between teacher and student encoder outputs. This is called *decoupled* distillation because encoder and decoder are optimised separately.

### Stage 2 — Automatic Decoder Distillation (`mainAuto.py`)
The student's UNet-based mask decoder is trained using **automatic self-distillation**: prompts (bounding boxes and centroid points) are derived automatically from ground truth masks, fed to the frozen SAM teacher, and the resulting high-quality masks are used as soft targets. The student decoder learns to replicate SAM's segmentation output without any manual annotation effort beyond the original GT masks. The loss used is **BCEWithLogitsLoss**.

### Stage 3 — End-to-End Fine-tuning (`mainFine.py`)
The full student model (encoder + decoder from the previous stages) is fine-tuned end-to-end on the MICCAI 2017 Robotic Instrument Segmentation dataset combined with a filtered subset of CholecSeg8k. The loss is **Dice Loss**, which directly optimises the overlap metric. A `ReduceLROnPlateau` scheduler and early stopping with patience are used to prevent overfitting.

---

## Architecture

The student model (`autoSamUnet`) replaces SAM's heavy ViT-H encoder with a **lightweight CMT encoder** and substitutes the original SAM mask decoder with a **UNet-style decoder**, drastically reducing the number of parameters while retaining segmentation quality on the surgical domain.

---

## Datasets

- **MICCAI 2017 Robotic Instrument Segmentation** — laparoscopic frames with binary and type segmentation masks for 8 instrument datasets (instruments 1–8).
- **CholecSeg8k** (via HuggingFace `minwoosun/CholecSeg8k`) — cholecystectomy frames, filtered to keep only frames containing surgical instruments.

---

## Repository Structure

```
toolSeg/
├── modeling/           # SAM model registry and architecture definitions
├── Encoder/            # Lightweight CMT encoder
├── EncoderRed/         # Reduced encoder variant
├── DecoderAutoSam/     # UNet-style mask decoder
├── repvit_sam/         # SamPredictor and related utilities
├── Dataset.py          # ImageMaskDataset, CholecDataset, LeedsDataset
├── engine.py           # Training and validation loop functions
├── losses.py           # Dice loss, distillation loss
├── utility.py          # IoU, dice, sensitivity, specificity, prompt generation
├── utils.py            # Misc helpers
├── mainDecoupled.py    # Stage 1: encoder distillation
├── mainAuto.py         # Stage 2: automatic decoder distillation
├── mainFine.py         # Stage 3: end-to-end fine-tuning
├── testSeg.py          # Unified inference and evaluation script
├── testAutoSam.py      # Legacy CPU test script
├── testAutoSamGPU.py   # Legacy GPU test script
├── videoSeg.py         # Inference on video files
├── videoSegArthrex.py  # Inference on Arthrex surgical videos
└── requirements.txt
```

---

## Installation

```bash
git clone https://github.com/mthezn/toolSeg.git
cd toolSeg
pip install -r requirements.txt
```

---

## Testing with `testSeg.py`

`testSeg.py` is the unified evaluation script. It runs inference with the trained student model on a set of images and computes IoU, Dice, Sensitivity and Specificity. Device (CPU or GPU) and all paths are configurable via command-line arguments.

### Basic usage

```bash
# Auto device (uses CUDA if available, otherwise CPU)
python testSeg.py

# Force CPU
python testSeg.py --device cpu

# Force GPU
python testSeg.py --device cuda
```

### With ground truth masks (full evaluation)

```bash
python testSeg.py \
    --device cuda \
    --image_dirs MICCAI/instrument_1_4_testing/instrument_dataset_1/left_frames \
                 MICCAI/instrument_1_4_testing/instrument_dataset_2/left_frames \
    --mask_dirs  MICCAI/instrument_2017_test/instrument_dataset_1/gt/BinarySegmentation \
                 MICCAI/instrument_2017_test/instrument_dataset_2/gt/BinarySegmentation
```

### Inference only (no ground truth)

```bash
python testSeg.py \
    --device cuda \
    --image_dirs path/to/your/images
```

### Custom checkpoint and output paths

```bash
python testSeg.py \
    --device cuda \
    --checkpoint checkpoints/checkpointsLight/autoSamFineUnetk57VL.pth \
    --model_type autoSamUnet \
    --image_dirs path/to/images \
    --mask_dirs  path/to/masks \
    --batch_size 2 \
    --output_dir results/ \
    --csv_out RISULTATI/metrics.csv
```

### Headless / no display (server environments)

```bash
python testSeg.py \
    --device cuda \
    --image_dirs path/to/images \
    --no_display
```

### All arguments

| Argument | Default | Description |
|---|---|---|
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--checkpoint` | `checkpoints/checkpointsLight/autoSamFineUnetk57VL.pth` | Path to model checkpoint |
| `--model_type` | `autoSamUnet` | Model key in the SAM registry |
| `--image_dirs` | MICCAI dataset 1 | One or more image folders |
| `--mask_dirs` | `None` | One or more GT mask folders (optional) |
| `--batch_size` | `2` | DataLoader batch size |
| `--shuffle` | `False` | Shuffle the dataset |
| `--output_dir` | `results_seg` | Where to save output images |
| `--csv_out` | `RISULTATI/TimeDfAutoSam.csv` | Path for the metrics CSV |
| `--no_display` | `False` | Disable `plt.show()` for headless environments |
