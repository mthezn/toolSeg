"""
test_autosam.py — Script unificato per il test di AutoSAM
=========================================================
Esempi di utilizzo:
  # GPU automatico (default)
  python test_autosam.py

  # Forza CPU
  python test_autosam.py --device cpu

  # Forza CUDA
  python test_autosam.py --device cuda

  # Dataset con mask (autosampuetest)
  python test_autosam.py --image_dirs path/to/imgs --mask_dirs path/to/masks

  # Dataset senza mask (autosam)
  python test_autosam.py --image_dirs path/to/imgs

  # Parametri completi
  python test_autosam.py \
      --device cuda \
      --checkpoint checkpoints/autoSamFineUnetk57VL.pth \
      --model_type autoSamUnet \
      --image_dirs MICCAI/inst1/left_frames MICCAI/inst2/left_frames \
      --mask_dirs  MICCAI/test/inst1/gt    MICCAI/test/inst2/gt \
      --batch_size 2 \
      --output_dir results_seg \
      --csv_out RISULTATI/TimeDf.csv \
      --no_display
"""

import argparse
import os
import sys
import time

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from Dataset import ImageMaskDataset, CholecDataset, LeedsDataset
from modeling.build_sam import sam_model_registry
from utility import dice_coefficient, sensitivity, specificity, refining, calculate_iou,show_mask




# ─────────────────────────────────────────────
# Argparse
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Test AutoSAM segmentation model")

    # Device
    parser.add_argument(
        "--device", type=str, default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device: 'auto' selects CUDA se disponibile, 'cpu' o 'cuda' forzano la scelta (default: auto)"
    )

    # Model
    parser.add_argument("--checkpoint",  type=str,
                        default="checkpoints/checkpointsLight/autoSamFineUnetk57VL.pth",
                        help="Percorso al checkpoint del modello")
    parser.add_argument("--model_type", type=str, default="autoSamUnet",
                        help="Tipo di modello nel registry SAM (default: autoSamUnet)")

    # Dataset
    parser.add_argument("--image_dirs", nargs="+", required=False,
                        default=["MICCAI/instrument_1_4_testing/instrument_dataset_1/left_frames"],
                        help="Una o più cartelle con le immagini")
    parser.add_argument("--mask_dirs", nargs="+", default=None,
                        help="Una o più cartelle con le mask (opzionale). Se omesso usa ImageMaskDataset senza GT)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size del DataLoader (default: 2)")
    parser.add_argument("--shuffle", action="store_true",
                        help="Shuffle del DataLoader")

    # Output
    parser.add_argument("--output_dir", type=str, default="results_seg",
                        help="Cartella dove salvare le immagini risultato (default: results_seg)")
    parser.add_argument("--csv_out", type=str, default="RISULTATI/TimeDfAutoSam.csv",
                        help="Percorso CSV per le metriche (default: RISULTATI/TimeDfAutoSam.csv)")
    parser.add_argument("--no_display", action="store_true",
                        help="Non mostrare le immagini con plt.show() (utile in ambienti headless)")

    return parser.parse_args()


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    args = parse_args()

    # ── Device ──────────────────────────────
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            print("[WARNING] CUDA richiesta ma non disponibile, uso CPU.")
            device = "cpu"
    print(f"[INFO] Dispositivo selezionato: {device}")

    # ── Transform ────────────────────────────
    transform = A.Compose([
        A.Resize(1024, 1024),
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])

    # ── Dataset ──────────────────────────────
    dataset = ImageMaskDataset(
        image_dirs=args.image_dirs,
        mask_dirs=args.mask_dirs,
        transform=transform,
    )
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=args.shuffle)
    print(f"[INFO] Dataset: {len(dataset)} campioni | mask_dirs: {'sì' if args.mask_dirs else 'no'}")

    # ── Model ────────────────────────────────
    model = sam_model_registry[args.model_type](checkpoint=None)
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.image_encoder.parameters() if p.requires_grad) / 1e6
    print(f"[INFO] Parametri encoder: {total_params:.2f} M")

    # ── Output dirs ──────────────────────────
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)

    # ── Inference loop ───────────────────────
    timeDf = pd.DataFrame(columns=["time", "index", "iou", "dice", "sensitivity", "specificity"])
    n = 0

    for images, labels in dataloader:
        images = images.to(device)
        labels = labels.to(device)
        print(f"[BATCH] images: {images.shape} | labels: {labels.shape}")

        for image, label in zip(images, labels):
            label_np = label.detach().cpu().numpy()
            label_np = (label_np > 0).astype(np.uint8)

            image = image.to(device).float().unsqueeze(0)  # (1, C, H, W)

            # ── Forward pass ─────────────────
            start = time.time()
            with torch.no_grad():
                image_embedding = model.image_encoder(image)
                low_res = model.mask_decoder(image_embedding)
                low_res = model.postprocess_masks(low_res, (1024, 1024), (1024, 1024))
            end = time.time()

            mask = (low_res > 0).detach().cpu().numpy()
            mask = refining(mask)

            # ── Metrics ──────────────────────
            iou  = calculate_iou(mask, label_np)
            dice = dice_coefficient(mask, label_np)
            sens = sensitivity(mask, label_np)
            spec = specificity(mask, label_np)
            latency_ms = (end - start) * 1000
            print(f"  [{n}] IoU={iou:.4f} | Dice={dice:.4f} | Sens={sens:.4f} | Spec={spec:.4f} | {latency_ms:.1f} ms")
            timeDf.loc[len(timeDf)] = [latency_ms, n, iou, dice, sens, spec]

            # ── Visualizzazione ──────────────
            img_vis = image.squeeze(0).cpu().permute(1, 2, 0).numpy()
            img_vis = (img_vis - img_vis.min()) / (img_vis.max() - img_vis.min() + 1e-8)

            fig, axs = plt.subplots(1, 3 if args.mask_dirs else 2, figsize=(14, 4))
            axs[0].imshow(img_vis);       axs[0].set_title("Image");      axs[0].axis("off")
            axs[1].imshow(mask, cmap="gray"); axs[1].set_title("Prediction"); axs[1].axis("off")
            if args.mask_dirs:
                axs[2].imshow(label_np.squeeze(), cmap="gray"); axs[2].set_title("Ground Truth"); axs[2].axis("off")

            save_path = os.path.join(args.output_dir, f"result_{n}.png")
            plt.savefig(save_path, bbox_inches="tight")
            if not args.no_display:
                plt.show()
            plt.close(fig)
            n += 1

    # ── Salva CSV ────────────────────────────
    timeDf.to_csv(args.csv_out, index=False)
    pd.set_option("display.max_rows", None)
    print("\n[RISULTATI]")
    print(timeDf)
    print(f"\n[INFO] CSV salvato in: {args.csv_out}")


if __name__ == "__main__":
    main()