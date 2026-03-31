# FreqMAE — Frequency-Aware Masked Autoencoder for Texture Anomaly Detection

> Self-supervised anomaly detection in industrial textures using frequency-domain masking.  
> Evaluated on the MVTec-AD benchmark.

---

## Method

We modify the Masked Autoencoder (MAE) training strategy by replacing random patch masking with a **frequency-aware masking** scheme. Patches are ranked by their high-frequency energy (computed via 2D FFT), and the top-k highest-frequency patches are preferentially masked during training.

**Hypothesis:** Forcing the decoder to reconstruct high-frequency texture details makes the model more sensitive to subtle industrial defects, which predominantly appear as local disruptions in texture regularity.

### Masking Modes (for ablation)

| Mode | Description |
|---|---|
| `freq` | Top-k high-frequency patches masked **(proposed)** |
| `random` | Uniform random masking — standard MAE baseline |
| `low` | Top-k low-frequency patches masked — inverse ablation |

---

## Repository Structure

```
freq-mae-anomaly/
├── configs/
│   └── default.yaml        # all hyperparameters
├── data/
│   └── .gitkeep            # MVTec-AD goes here (gitignored)
├── src/
│   ├── masking.py          # frequency-aware masking (core novelty)
│   ├── model.py            # FreqMAE encoder-decoder
│   ├── dataset.py          # MVTec-AD dataloader
│   ├── train.py            # self-supervised training loop
│   └── evaluate.py         # AUROC + PRO evaluation
├── notebooks/
│   └── colab_train.ipynb   # end-to-end Colab T4 notebook
├── requirements.txt
└── README.md
```

---

## Dataset

**MVTec-AD** — 15 industrial texture/object categories, ~5GB.  
Free for academic research. Download at:  
https://www.mvtec.com/company/research/datasets/mvtec-ad

Place extracted data at `data/mvtec/` (or set `data_root` in `configs/default.yaml`).

---

## Quickstart

### Local (CPU, for development)

```bash
# Install dependencies
pip install -r requirements.txt

# Verify modules
python src/masking.py
python src/model.py
python src/dataset.py
```

### Training on Colab T4

1. Open `notebooks/colab_train.ipynb` in Google Colab
2. Set runtime to **T4 GPU**
3. Fill in GitHub credentials in Cell 3
4. Upload `mvtec_anomaly_detection.tar.xz` to `MyDrive/freqmae/`
5. Run all cells

### Training via CLI

```bash
# Single category
python src/train.py --category bottle

# All 15 categories
python src/train.py --all

# Quick 50-epoch test
python src/train.py --category bottle --epochs 50
```

### Evaluation

```bash
# Single category
python src/evaluate.py --category bottle --checkpoint checkpoints/bottle/best.pth

# Full benchmark table
python src/evaluate.py --all --checkpoint_dir checkpoints/

# Ablation table (all 3 masking modes)
python src/evaluate.py --all --mask_mode freq   --output results/freq.txt
python src/evaluate.py --all --mask_mode random --output results/random.txt
python src/evaluate.py --all --mask_mode low    --output results/low.txt
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Image AUROC** | Separates normal vs anomaly images |
| **Pixel AUROC** | Anomaly map alignment with GT defect pixels |
| **PRO Score** | Per-Region Overlap — rewards detecting small defects |

---

## Configuration

All hyperparameters are in `configs/default.yaml`. Key settings:

```yaml
mask_mode  : "freq"    # core novelty — change for ablations
mask_ratio : 0.75      # fraction of patches masked
epochs     : 200
batch_size : 32        # safe for T4 16GB
lr         : 1.5e-4
```

---

## Citation

> (manuscript in preparation)