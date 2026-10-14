# Architecture and design decisions

## 1. Data flow

Everything starts from one **manifest** (`artifacts/manifest.csv`): one row per image with `label`, `split`
(train/val/test, stratified per class), `lesion_frac` and a proxy `severity`. Every training script builds its tables
from `data.datasets.load_frames`, which is the only place where rules are applied:

- rare classes are down-sampled **in train only** (`data.rare_classes`, `data.rare_keep_fraction`);
- holdout classes are removed from train/val and flagged `is_unknown` in test (open-set experiment);
- synthetic images are appended to **train only**, and only when requested.

Validation and test therefore always consist of untouched real images, and the diffusion model is trained on exactly
the (reduced) training set the classifier sees. A unit test asserts that train/val/test never overlap.

## 2. Components

### Classifier (`models/classifier.py`, `training/train_classifier.py`)
ResNet18 backbone (ImageNet-pretrained by default; a small custom CNN is available) with two linear heads. Disease loss
is cross-entropy with label smoothing; severity loss is masked so healthy and unlabeled samples do not contribute.
Training uses AdamW + one-cycle LR, mixed precision, gradient clipping, early stopping on **validation macro-F1**
(accuracy is a poor model-selection metric on imbalanced classes). After training, a single **temperature** is fitted
on validation logits (Guo et al., 2017) and stored in the checkpoint, so `confidence` in the API is a calibrated
probability, which matters because the advisory thresholds on it. ECE is reported before and after.

An augmentation (`A.Downscale`) deliberately blurs a fraction of *all* training images. Diffusion samples are generated
at 64 px and up-sampled to 128 px, so without this the classifier could learn "blurry = rare class" as a shortcut.

### Diffusion (`models/ddpm.py`, `training/train_ddpm.py`, `training/generate_synthetic.py`)
A from-scratch epsilon-prediction DDPM: cosine noise schedule, a ~22 M parameter UNet (ResBlocks, self-attention at
low resolutions, sinusoidal time embedding), a class embedding with a learned *null* token so that **classifier-free
guidance** can be used at sampling time, EMA weights, and **DDIM** sampling (50 steps instead of 1000).
Generation is gated twice: a *teacher* classifier trained on real data only must agree with the intended class, and
images whose feature-space cosine similarity to any real training image is above a threshold are dropped, which guards
against memorisation. The same lesion-area severity proxy is computed for each synthetic image.

### VAE (`models/vae.py`, `training/train_vae.py`, `evaluation/eval_vae.py`)
Convolutional VAE trained on healthy leaves only. The anomaly score is the **mean of the worst 10% pixel errors**
(`vae.score: topk`), which reacts to small local lesions that a global mean-squared error averages away. The alert
threshold is the 95th percentile of healthy *validation* scores (about a 5% false-alarm rate by construction). Evaluation
reports AUROC and per-disease detection rate, plus an open-set test (see README). VAE reconstructions are blurry by
nature and the score is sensitive to lighting/background shifts, so treat it as a *screen*, not a classifier.

### SNN (`models/snn.py`, `training/train_snn.py`, `edge/export_snn.py`)
Input: 24 hourly readings x 3 features. **Encoding**: per feature a *population* of 16 neurons with Gaussian tuning
curves (resolves the absolute level) plus UP/DOWN neurons driven by the hour-to-hour change (resolves rain/irrigation
onset and flat-lined or jumping probes); spikes are Bernoulli draws, 4 sub-steps per hour (96 time steps).
**Network**: Linear - LIF - Linear - LIF - Linear - non-spiking leaky-integrator readout whose membrane is averaged over the
last 25% of the spike train (the decision concerns the present state). Training uses a fast-sigmoid surrogate gradient.
The LIF update is `mem_t = beta*mem_{t-1} + I_t - thr*H(mem_{t-1}-thr)`, `spk_t = H(mem_t-thr)`, identical in snnTorch,
in the dependency-free native fallback, and in the NumPy edge runtime; equality of logits is unit-tested.
Inference averages logits over 5 seeded spike-train draws so the decision is reproducible.

A GRU baseline on the same windows is trained in the same script. Cost is compared with an op-count proxy: spike-driven
accumulates (0.9 pJ) plus neuron-state MACs (4.6 pJ) for the SNN versus MACs for the GRU (45 nm numbers from Horowitz,
ISSCC 2014). That is an argument, not a measurement.

### Fusion (`pipeline/advisory.py`)
Disease evidence becomes a **status**: `known_disease`, `healthy`, `healthy_flagged`, `uncertain`, or
`possible_unknown_disease` (low classifier confidence **and** VAE alarm). Each maps to a risk. The irrigation action adds
urgency, and the two interact: a foliar disease turns "irrigate" into "irrigate at the base, early morning, avoid wetting
leaves"; mites turn it into "keep moisture steady"; high 24 h humidity adds an infection-pressure warning only for
humidity-driven diseases; a sensor `alert` tells the user to inspect the field before trusting the automation. Pure
Python, covered by 10 scenario tests.

## 3. Checkpoint formats

| File | Keys |
|---|---|
| classifier `best.pt` | `model_state, class_names, backbone, image_size, temperature, severity_names` |
| VAE `best.pt` | `model_state, image_size, latent_dim, base, threshold, score_mode, topk_frac` |
| SNN `best.pt` | `model_state (fc weights), hyper, encoder, ranges, actions, window_hours` |
| DDPM `best.pt` | `ema_state, model_state, class_names, unet (arch), timesteps, schedule` |

`CropAdvisor` loads each model independently: a missing checkpoint degrades the advisory (API returns 503 for that
branch only) instead of crashing the service.

## 4. Failure modes to know about

- Domain shift from lab to field images (largest risk for the classifier and the VAE).
- Diffusion samples can be off-class or memorised; the two gates reduce, not eliminate, that. Inspect the sample grids.
- The VAE flags background/lighting changes as "disease". Calibrate the threshold on images captured like deployment data.
- Sensor faults the simulator does not model (drift, partial dropouts) will not be recognised.
- All thresholds in the advisory are heuristics for decision support and need agronomist review before real use.
