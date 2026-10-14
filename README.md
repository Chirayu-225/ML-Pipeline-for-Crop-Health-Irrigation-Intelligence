# Crop Health & Irrigation Intelligence

A multimodal agricultural decision-support pipeline for tomato farming: it screens a **leaf photo** for disease
(type + severity, plus an *unknown-disease* alarm) and reads **soil/weather sensor time-series** to recommend an
**irrigation action**, then fuses both into a single advisory that knows how the two interact (for example,
*how* to irrigate when a foliar disease is present).

| Branch | Model | Job |
|---|---|---|
| Disease classification | ResNet18 (or custom CNN), two heads | disease type + severity (mild / moderate / severe) |
| Rare-class augmentation | Class-conditional DDPM (+ classifier-free guidance, DDIM sampling, EMA) | synthetic images for under-represented diseases, **train-time only** |
| Unknown-disease screen | Convolutional VAE trained on healthy leaves | reconstruction error flags leaves that look abnormal |
| Irrigation | Spiking neural network (snnTorch, LIF) on spike-encoded sensor windows | `wait` / `irrigate` / `alert`, edge-deployable (NumPy runtime) |
| Fusion | Rule-based advisory layer | one risk level + prioritised, interaction-aware recommendations |

## Architecture

```
TRAINING TIME                                          INFERENCE TIME
-------------                                          --------------
real TRAIN images (rare classes reduced)                     leaf photo                 sensor window (24 h x 3)
        |                                                       |                              |
        +--> DDPM --> synthetic rare-class images               +------------+---------+      |
        |              (teacher + near-duplicate gates)         |            |         |      |
        |                       |                               v            v         |      v
        +---------+-------------+                          ResNet18      VAE          |   spike encoder
                  v                                       (disease +    (healthy-    |   (population + delta)
         classifier training                               severity,     only; error |      |
        (real [+ synthetic], val/test                      temperature-  = anomaly)   |      v
         always real only)                                  scaled)         |         |     SNN
                                                                |           |         |      |
healthy-only images --> VAE training (+ threshold)              +-----+-----+         |      |
                                                                      v               |      v
sensor simulator / CSV --> SNN training (+ GRU baseline)       disease status         |  wait / irrigate / alert
                                                              (known / uncertain /     |      |
                                                               possible unknown)       |      |
                                                                      +---------------+------+
                                                                                      v
                                                                  fused advisory: risk level + recommendations
                                                                          (FastAPI  /  Streamlit dashboard)
```

Two things are deliberately different from a naive reading of the pipeline: the **diffusion model never runs at
inference time** (it only creates training data), and the **VAE runs next to the classifier**, not before it.

## Tech stack

Python 3.10+, PyTorch, torchvision, snnTorch, Pandas / NumPy / scikit-learn, OpenCV + Albumentations, FastAPI,
Streamlit, Docker / Docker Compose, MLflow or Weights & Biases (optional, `tracking.backend` in the config), Git.

## Repository layout

```
configs/            default.yaml (4 GB-GPU settings) and smoke.yaml (CPU, tiny)
src/cropiq/
  data/             manifest + splits, lesion-area severity proxy, datasets/augmentation, toy leaves, sensor simulator
  models/           classifier.py  ddpm.py  vae.py  snn.py  loading.py
  training/         train_classifier  train_ddpm  generate_synthetic  train_vae  train_snn
  evaluation/       ablation (diffusion vs. baselines)  eval_vae (AUROC, open-set)  metrics (ECE, temperature scaling)
  pipeline/         advisory.py (fusion rules)  inference.py (CropAdvisor)  knowledge.py
  edge/             export_snn.py (NumPy-only SNN runtime)
  api/              main.py (FastAPI)        dashboard/app.py (Streamlit)
tests/              60 tests: data leakage, models, SNN/edge parity, advisory rules, API
docs/               ARCHITECTURE  MODEL_CARD  INTERVIEW_GUIDE  ROADMAP
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate
# NVIDIA GPU: install the CUDA build of PyTorch first, e.g.
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e ".[serve,dev]"

make test      # unit + integration tests
make smoke     # full pipeline on toy data, CPU only, a few minutes (proves the plumbing, not model quality)
```

## Getting the data

**Leaf images - PlantVillage.** Download it yourself (Kaggle, GitHub `spMohanty/PlantVillage-Dataset`, or TensorFlow
Datasets `plant_village` - check the licence/terms of whichever source you use) and unpack under
`data/plantvillage/`. Any folder nesting works; the loader finds class folders recursively, understands the common
naming variants (`Tomato___Early_blight`, `Tomato_Early_blight`, ...), and ignores `grayscale/` and `segmented/`
variants unless you set `data.variant`. By default only **tomato** classes are used (`data.crop: tomato`, 10 classes);
set `data.crop: all` for the full 38-class dataset.

**Sensor data.** With `paths.sensor_csv: null` the built-in simulator generates hourly soil-moisture / temperature /
humidity for many fields (monsoon rain, heat waves, a noisy farmer irrigating, sensor faults). To use real or
AgriSense-style data, point `paths.sensor_csv` at a CSV with columns `soil_moisture, temperature, humidity`
(common aliases such as `moisture`, `temp`, `rh` are accepted) and optionally `timestamp`, `field_id`, `label`
(`wait|irrigate|alert`). Without a `label` column, an observed-values-only rule labeller is used.

## Full pipeline (4 GB GPU)

```bash
make data          # scan PlantVillage -> artifacts/manifest.csv (stratified splits + proxy severity labels)
make clf-real      # baseline classifier on real data only
make clf-balanced  # baseline + inverse-frequency sampling   (the strong classical baseline)
make ddpm          # train the class-conditional diffusion model  (the long step; resumable: --resume)
make synth         # generate + quality-gate synthetic images for the rare classes
make clf-synth     # classifier on real + synthetic
make vae           # healthy-only VAE, threshold calibration, AUROC evaluation
make snn           # SNN + GRU baseline on sensor windows
make ablation      # -> artifacts/runs/ablation.md
make edge          # NumPy export of the SNN
make api           # http://localhost:8000/docs
make dashboard     # http://localhost:8501
```

The ResNet18 ImageNet weights are downloaded by torchvision on first use (internet needed once; set
`classifier.pretrained: false` to avoid it). I could not test on a GPU in my build environment (CPU only), so the VRAM
settings are conservative estimates, not measurements.

Rough time budget on an RTX 3050 Ti 4 GB (estimates, not measurements - replace with your own): classifier runs tens of
minutes each; DDPM several hours (it checkpoints every epoch); VAE and SNN minutes. Mixed precision is on by default.
If you hit CUDA OOM: lower `diffusion.batch_size` and raise `diffusion.grad_accum`, reduce `diffusion.base_channels`,
or lower `data.batch_size`.

## What to report (and how it is measured)

Three experiments make this more than "I trained four models". Fill the tables from **your** runs.

1. **Does diffusion augmentation help?** `data.rare_classes` keeps only 15% of the training images of three diseases
   (a controlled low-data setting). Compare `clf_real`, `clf_balanced` and `clf_synthetic` on the same untouched real
   test set, focusing on rare-class F1. The balanced-sampler baseline matters: if diffusion does not beat plain
   oversampling, say so - that is still a valid, honest finding.
2. **Can the VAE catch unseen diseases?** Healthy-vs-diseased AUROC and per-disease detection rate at a threshold fixed
   from healthy *validation* images. Add the open-set test: `data.holdout_classes: [target_spot]` removes a disease from
   classifier training, then `python -m cropiq.evaluation.eval_vae` reports how often (VAE alarm OR low classifier
   confidence) catches it, against the false-alarm rate on known classes.
3. **SNN vs. a conventional GRU** on identical sensor windows: macro-F1, per-class recall (especially `alert`),
   robustness to extra sensor noise, and an op-count energy *proxy*.

| Experiment | Metric | Your result |
|---|---|---|
| Classifier (real only) | test macro-F1 / rare-class F1 | _fill in_ |
| Classifier (+ oversampling) | test macro-F1 / rare-class F1 | _fill in_ |
| Classifier (+ diffusion) | test macro-F1 / rare-class F1 | _fill in_ |
| Calibration | ECE before / after temperature scaling | _fill in_ |
| VAE screen | AUROC; detection @ 5% false alarms | _fill in_ |
| Open-set | unseen-disease detection vs. false alarms | _fill in_ |
| SNN vs GRU | macro-F1, alert recall, energy proxy | _fill in_ |

Sandbox verification (not a benchmark): every stage was run end-to-end on procedurally generated toy leaves and
simulated sensors on CPU, and the test suite passes. With the default config on the *simulated* sensor task the SNN
reached 0.80 test macro-F1 versus 0.83 for a GRU (`alert` recall 0.70 vs 0.75), at an estimated ~3.6x lower op-count
energy per decision (about 103 nJ vs 371 nJ by the proxy in `docs/ARCHITECTURE.md`). That is a statement about
simulated data and a counting model, not about field performance or measured power; see `docs/MODEL_CARD.md`.

## Serving

```bash
curl -F "file=@leaf.jpg" http://localhost:8000/predict/image
curl -X POST http://localhost:8000/predict/irrigation -H "Content-Type: application/json" \
     -d '{"readings": [{"soil_moisture": 27.5, "temperature": 31.0, "humidity": 55.0}, ... 24 hourly rows ...]}'
curl -F "file=@leaf.jpg" -F 'readings={"readings": [...]}' http://localhost:8000/advisory
```

Docker: `docker compose up --build` serves the API (8000) and dashboard (8501) from the checkpoints in `./artifacts`.

## Known limitations (read before putting numbers on a resume)

- **PlantVillage is lab imagery** (single leaf, uniform background). Accuracy on field photos will be much lower, and
  the dataset is known to contain near-duplicate images of the same leaf that can inflate random-split scores. A
  field-photo or cross-dataset check (e.g. PlantDoc, or your own phone photos from a local farm) is the most valuable
  next experiment.
- **Severity labels are a proxy** (lesion-area fraction from colour segmentation), not expert grades. It works for
  lesion-type diseases and poorly for viruses and mites.
- **Sensor labels in the simulator come from a rule policy.** The SNN learns to imitate that policy under noise and
  faults; this is not field-validated agronomy. Real irrigation data would change the story.
- **The energy numbers are an op-count proxy**, not a power measurement on hardware.
- **The advisory is decision support**, not a diagnosis, and deliberately avoids product/dose recommendations.

## Licence

Code: MIT (add a LICENSE file when you publish). Respect the licence of whichever dataset you download.
