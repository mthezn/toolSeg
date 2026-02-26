import cv2
import torch
import numpy as np
from torchvision import transforms
import torch.nn.functional as F

from modeling.build_sam import sam_model_registry

# ============================================================================
# PREPROCESSING ARTROSCOPICO
# ============================================================================

def preprocess_arthroscopy_frame(frame_bgr):
    """
    Preprocessing specifico per ridurre gli artefatti artroscopici
    PRIMA di passare al modello.

    Problemi artroscopia:
      - Vignette circolare nera ai bordi → confusa con strumento scuro
      - Riflessioni speculari del liquido sinoviale → confuse col metal
      - Basso contrasto strumento/tessuto → CLAHE aiuta
    """
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    # 1. Sostituisce il vignette circolare nero con grigio neutro
    #    → il modello non confonde i bordi neri con lo strumento
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    _, bright = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
    kernel_big = np.ones((20, 20), np.uint8)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, kernel_big)
    result = frame_rgb.copy()
    result[bright == 0] = [127, 127, 127]  # grigio neutro nei bordi neri

    # 2. Rimuove riflessioni speculari con inpainting
    #    → pixel >240 sono riflessioni, non strumento metallico
    gray2 = cv2.cvtColor(result, cv2.COLOR_RGB2GRAY)
    spec_mask = (gray2 > 240).astype(np.uint8) * 255
    if spec_mask.sum() > 0:
        kernel_spec = np.ones((5, 5), np.uint8)
        spec_mask = cv2.dilate(spec_mask, kernel_spec, iterations=2)
        result = cv2.inpaint(result, spec_mask, 5, cv2.INPAINT_TELEA)

    # 3. CLAHE sul canale L (luminanza) per migliorare contrasto locale
    #    → distingue meglio bordi strumento da tessuto circostante
    lab = cv2.cvtColor(result, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    result = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)

    return result  # RGB uint8


# ============================================================================
# POST-PROCESSING MASCHERA
# ============================================================================

class MaskPostProcessor:
    """
    Post-processing con smoothing temporale tra frame.
    Mantiene uno storico delle maschere per ridurre il flickering.
    """

    def __init__(self,
                 min_area=800,           # filtra blob troppo piccoli
                 temporal_window=5,      # quanti frame passati usare
                 temporal_alpha=0.7,     # peso frame corrente vs storia
                 morph_kernel_size=5):   # dimensione kernel morfologia

        self.min_area = min_area
        self.temporal_window = temporal_window
        self.temporal_alpha = temporal_alpha
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
        )
        self.history = []  # buffer maschere precedenti

    def process(self, raw_mask_float, frame_rgb):
        """
        Args:
            raw_mask_float : maschera grezza [0,1] float, shape (H,W)
            frame_rgb      : frame RGB uint8 per analisi speculare e vignette
        Returns:
            refined_mask   : maschera raffinata [0,1] float, shape (H,W)
        """
        mask = raw_mask_float.copy()

        # Step 1 – Threshold Otsu adattivo (meglio di soglia fissa)
        mask = self._otsu_threshold(mask)

        # Step 2 – Escludi i bordi del vignette circolare
        mask = self._remove_vignette_region(mask, frame_rgb)

        # Step 3 – Attenua zone speculari (non sono strumento)
        mask = self._attenuate_specular(mask, frame_rgb)

        # Step 4 – Morfologia: chiude buchi, rimuove rumore
        mask = self._morphology(mask)

        # Step 5 – Filtra componenti troppo piccole
        mask = self._filter_small_blobs(mask)

        # Step 6 – Smoothing temporale
        mask = self._temporal_smooth(mask)

        # Aggiorna storia
        self.history.append(mask.copy())
        if len(self.history) > self.temporal_window:
            self.history.pop(0)

        return mask

    def _otsu_threshold(self, mask_float):
        uint8 = (mask_float * 255).astype(np.uint8)
        _, binary = cv2.threshold(uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Fallback se Otsu dà risultato degenere
        if binary.mean() > 250 or binary.mean() < 5:
            binary = (mask_float > 0.5).astype(np.uint8) * 255
        return binary.astype(np.float32) / 255.0

    def _remove_vignette_region(self, mask, frame_rgb):
        """La maschera non può stare nei bordi neri del canale artroscopico"""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        _, fov = cv2.threshold(gray, 15, 255, cv2.THRESH_BINARY)
        kernel = np.ones((20, 20), np.uint8)
        fov = cv2.morphologyEx(fov, cv2.MORPH_CLOSE, kernel)
        fov_float = fov.astype(np.float32) / 255.0
        return mask * fov_float

    def _attenuate_specular(self, mask, frame_rgb):
        """Le riflessioni speculari non sono parte dello strumento"""
        gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        spec = (gray > 240).astype(np.uint8)
        kernel = np.ones((7, 7), np.uint8)
        spec = cv2.dilate(spec, kernel, iterations=1).astype(np.float32)
        return mask * (1.0 - spec * 0.8)  # attenua, non annulla del tutto

    def _morphology(self, mask):
        uint8 = (mask * 255).astype(np.uint8)
        closed = cv2.morphologyEx(uint8, cv2.MORPH_CLOSE, self.kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, self.kernel)
        smoothed = cv2.GaussianBlur(opened, (5, 5), 0)
        return smoothed.astype(np.float32) / 255.0

    def _filter_small_blobs(self, mask):
        uint8 = (mask > 0.5).astype(np.uint8) * 255
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            uint8, connectivity=8
        )
        out = np.zeros_like(mask)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= self.min_area:
                out[labels == lbl] = mask[labels == lbl]
        return out

    def _temporal_smooth(self, current):
        if len(self.history) == 0:
            return current
        temporal_mean = np.mean(self.history, axis=0)
        smoothed = self.temporal_alpha * current + (1 - self.temporal_alpha) * temporal_mean
        return (smoothed > 0.5).astype(np.float32)

    def reset(self):
        self.history = []


# ============================================================================
# SELF-SUPERVISED ADAPTATION (opzionale, eseguita UNA VOLTA sul video)
# ============================================================================

class SelfSupervisedAdapter:
    """
    Adatta il decoder del modello al video artroscopico senza annotazioni.

    Strategia:
    - Congela image_encoder (CMT-Ti) → preserva conoscenza laparoscopica
    - Finetuna solo mask_decoder (UNet) → adatta all'artroscopia
    - Loss non supervisionate:
        1. Consistency: flip(pred(x)) == pred(flip(x))
        2. Entropy minimization: forza predizioni sicure (no ambiguità)
        3. Self-training: pseudo-label con predizioni ad alta confidence
    """

    def __init__(self, model, device='cuda', lr=1e-5):
        self.model = model
        self.device = device

        # Congela encoder, sblocca solo decoder
        for name, param in model.named_parameters():
            param.requires_grad = ('image_encoder' not in name)

        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in model.parameters())
        print(f"Trainable params: {n_train:,} / {n_total:,} "
              f"({100 * n_train / n_total:.1f}%) — solo il decoder")

        self.opt = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=1e-4
        )

    def _forward(self, x):
        """Replica esatta del forward loop del tuo codice"""
        emb = self.model.image_encoder(x)
        pred = self.model.mask_decoder(emb)
        mask = self.model.postprocess_masks(pred, (1024, 1024), (1024, 1024))
        return mask  # [B,1,H,W] logits

    def _consistency_loss(self, x):
        """pred(flip(x)) deve essere flip(pred(x))"""
        with torch.no_grad():
            pred_orig = self._forward(x)
        x_flip = torch.flip(x, dims=[-1])
        pred_flip = self._forward(x_flip)
        target = torch.flip(torch.sigmoid(pred_orig), dims=[-1]).detach()
        return F.binary_cross_entropy_with_logits(pred_flip, target)

    def _entropy_loss(self, x):
        """Minimizza entropia → il modello deve essere sicuro"""
        pred = self._forward(x)
        p = torch.sigmoid(pred)
        entropy = -(p * torch.log(p + 1e-8) + (1 - p) * torch.log(1 - p + 1e-8))
        return entropy.mean()

    def _self_training_loss(self, x, conf_thresh=0.85):
        """Pseudo-label dai pixel ad alta confidence"""
        with torch.no_grad():
            prob = torch.sigmoid(self._forward(x))
        high_conf = (prob > conf_thresh) | (prob < (1 - conf_thresh))
        if high_conf.float().mean() < 0.05:
            return torch.tensor(0.0, requires_grad=True).to(self.device)
        pseudo = (prob > 0.5).float()
        pred_new = self._forward(x)
        return F.binary_cross_entropy_with_logits(
            pred_new[high_conf], pseudo[high_conf]
        )

    def adapt(self, frames_tensor_list, n_epochs=3, batch_size=4):
        """
        Args:
            frames_tensor_list : lista di tensor (C,H,W) già normalizzati
            n_epochs           : 3 epoche sono sufficienti (TTT leggero)
        """
        print(f"\nAdaptation: {len(frames_tensor_list)} frames × {n_epochs} epochs")
        self.model.train()

        for epoch in range(n_epochs):
            idx = torch.randperm(len(frames_tensor_list))
            total_loss = 0.0
            n_batches = 0

            for i in range(0, len(frames_tensor_list), batch_size):
                batch = torch.stack(
                    [frames_tensor_list[j] for j in idx[i:i + batch_size]]
                ).to(self.device)

                self.opt.zero_grad()
                loss = (1.0 * self._consistency_loss(batch)
                        + 0.3 * self._entropy_loss(batch)
                        + 0.5 * self._self_training_loss(batch))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                total_loss += loss.item()
                n_batches += 1

            print(f"  Epoch {epoch + 1}/{n_epochs}  loss={total_loss / n_batches:.4f}")

        self.model.eval()
        print("✓ Adaptation done\n")


# ============================================================================
# CONFIG
# ============================================================================

video_input  = "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/arthrex/acl_short2.mp4"
video_output = "C:/Users/User/OneDrive - Politecnico di Milano/Documenti/arthrex/acl_short_seg2.mp4"

autosam_checkpoint = "checkpoints/checkpoints_finali/autoSamFineUnetMUcH0.pth"
model_type = "autoSamUnet"

DO_ADAPTATION   = False   # ← metti False per saltare il fine-tuning
ADAPTATION_FPS  = 5      # campiona 1 frame ogni N per l'adaptation
N_EPOCHS        = 3      # epoche adaptation (3 bastano)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================================
# TRANSFORM (identica alla tua originale)
# ============================================================================

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((1024, 1024)),
    transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)
])

# ============================================================================
# CARICA MODELLO (identico al tuo)
# ============================================================================

checkpoint = torch.load(autosam_checkpoint, map_location='cpu')
model = sam_model_registry[model_type](checkpoint=None)
model.load_state_dict(checkpoint)
model.to(device)
model.eval()

# ============================================================================
# APRI VIDEO
# ============================================================================

cap = cv2.VideoCapture(video_input)
fps       = cap.get(cv2.CAP_PROP_FPS)
frame_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Video: {frame_w}×{frame_h} @ {fps:.1f}fps  —  {total_frames} frames")

out = cv2.VideoWriter(
    video_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_w, frame_h)
)

# ============================================================================
# STEP 1 — SELF-SUPERVISED ADAPTATION (opzionale)
# ============================================================================

if DO_ADAPTATION:
    print("Raccolta frame per adaptation...")
    frames_for_adapt = []
    frame_counter = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Campiona 1 frame ogni ADAPTATION_FPS per non usare tutto il video
        if frame_counter % ADAPTATION_FPS == 0:
            preprocessed_rgb = preprocess_arthroscopy_frame(frame)
            tensor = transform(preprocessed_rgb).to(device)
            frames_for_adapt.append(tensor.cpu())  # tieni su CPU finché non serve
        frame_counter += 1
        if frame_counter > total_frames * 0.3:    # usa solo il primo 30% del video
            break

    # Torna all'inizio
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    adapter = SelfSupervisedAdapter(model, device=device, lr=1e-5)
    adapter.adapt(frames_for_adapt, n_epochs=N_EPOCHS, batch_size=4)

# ============================================================================
# STEP 2 — INFERENZA + POST-PROCESSING FRAME-BY-FRAME
# ============================================================================

postprocessor = MaskPostProcessor(
    min_area=800,
    temporal_window=5,
    temporal_alpha=0.7,
    morph_kernel_size=5
)

frame_idx = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    original = frame.copy()

    # --- Preprocessing artroscopico ---
    preprocessed_rgb = preprocess_arthroscopy_frame(frame)

    # --- Trasformazione per il modello ---
    img = transform(preprocessed_rgb).unsqueeze(0).to(device)

    # --- Inferenza (identica al tuo codice) ---
    with torch.no_grad():
        image_embedding = model.image_encoder(img)
        pred = model.mask_decoder(image_embedding)

    # --- Post-process interno del modello (identico al tuo) ---
    mask = model.postprocess_masks(pred, (1024, 1024), (1024, 1024)).cpu().numpy()
    mask = (mask > 0).astype(np.uint8)
    mask = mask.squeeze()  # (1024, 1024)

    # --- Resize alla dimensione originale PRIMA del post-processing ---
    mask_resized = cv2.resize(
        mask.astype(np.float32),
        (frame_w, frame_h),
        interpolation=cv2.INTER_LINEAR
    )

    # --- Post-processing artroscopico avanzato ---
    #     (usa il frame RGB preprocessato per analisi speculare/vignette)
    preprocessed_rgb_full = cv2.resize(preprocessed_rgb, (frame_w, frame_h))
    refined_mask = postprocessor.process(mask_resized, preprocessed_rgb_full)

    # --- Overlay sul frame originale ---
    colored_mask = np.zeros_like(original)
    colored_mask[:, :, 1] = (refined_mask * 255).astype(np.uint8)  # verde
    blended = cv2.addWeighted(original, 0.8, colored_mask, 0.4, 0)

    # Aggiungi contorno per visibilità
    binary_u8 = (refined_mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(binary_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(blended, contours, -1, (0, 255, 0), 2)

    out.write(blended)

    frame_idx += 1
    if frame_idx % 30 == 0:
        print(f"  Frame {frame_idx}/{total_frames}")

# ============================================================================
# CHIUDI
# ============================================================================

cap.release()
out.release()
print(f"\n✓ Video salvato in {video_output}")