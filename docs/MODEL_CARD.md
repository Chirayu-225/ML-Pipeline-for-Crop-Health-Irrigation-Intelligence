# Model card

## Intended use
A portfolio / research prototype of decision support for tomato growers: screen a leaf photo, read recent sensor data,
and suggest next steps. **Not** a certified diagnostic tool and not a substitute for an agronomist. The advisory never
recommends specific chemicals or doses.

## Models

| Model | Input | Output | Training data |
|---|---|---|---|
| Classifier (ResNet18, 2 heads) | 128x128 RGB leaf | disease class (calibrated probs), severity | PlantVillage tomato train split (+ synthetic rare classes in one variant) |
| DDPM (UNet ~22 M, class-conditional) | noise + class | 64x64 synthetic leaf | PlantVillage tomato train split only |
| VAE | 128x128 RGB leaf | anomaly score + alert flag | healthy tomato leaves, train split |
| SNN (~8 k parameters) | 24 h x 3 sensor readings | wait / irrigate / alert | simulated fields (or your CSV) |

## Data
- **PlantVillage**: ~18 k tomato images in 10 classes, lab conditions, plain backgrounds. Class imbalance is natural
  (e.g. mosaic virus is ~300 images, yellow leaf curl ~3 k). Splits are stratified per class (70/15/15) from a manifest.
- **Severity**: not annotated in PlantVillage. Proxy = non-green fraction of the segmented leaf, thresholds 8% / 25%.
- **Sensors**: simulated hourly data with a Karnataka-like climate (monsoon rain, heat waves), a water-balance soil model,
  a noisy farmer irrigating, and soil-probe faults (flat-line, dropout, spike). Labels come from an agronomic rule policy
  evaluated on the hidden, noise-free state; the model only sees noisy/faulty observations.

## Evaluation protocol
Test images are real, held out, and never seen by the classifier, the diffusion model or the synthetic-image filters.
Model selection uses validation macro-F1. Reported: accuracy, macro-F1, per-class F1, rare-class macro-F1, ECE
(before/after temperature scaling), VAE AUROC and detection rate, open-set detection, SNN/GRU macro-F1 and per-class recall.
Report numbers from your own runs in README.md; do not reuse numbers from a different configuration.

## Observed behaviour in the build environment
The full pipeline was executed on toy data on CPU to verify correctness (tests + smoke run). On the **simulated** sensor
task with the default config (28 training fields, 30 epochs) the SNN scored 0.805 test macro-F1 versus 0.827 for a GRU.
Per class (precision / recall / F1): wait 0.98 / 0.90 / 0.94, irrigate 0.86 / 0.96 / 0.91, alert 0.48 / 0.70 / 0.57 for the SNN,
against alert 0.49 / 0.75 / 0.60 for the GRU. With extra sensor noise added at test time the SNN's macro-F1 fell from 0.80 to
0.76 (x1), 0.73 (x2) and 0.69 (x3). The op-count energy proxy was about 103 nJ per decision for the SNN versus 371 nJ for the
GRU (~3.6x). Getting there needed three specific fixes: a population-coded input with a high-gain change channel, a
readout over the most recent part of the spike train, and full inverse-frequency class weights; before them the SNN plateaued
near 0.63 macro-F1 while a gradient-boosting check reached 0.81. These results are on simulated data with rule-generated
labels and say nothing about field performance; the precision of `alert` (about 0.5) means many false alarms, which is the
safer error here but still a cost.

## Limitations and risks
- Lab-to-field domain shift; near-duplicate images in PlantVillage can inflate random-split metrics.
- Proxy severity labels are weak for viral and mite problems.
- The VAE can flag lighting/background changes; its false-alarm rate is only calibrated for deployment-like healthy images.
- Rule-generated sensor labels mean the SNN benchmark measures imitation of a policy, not agronomic benefit.
- Energy figures are op-count proxies, not hardware measurements.
- A wrong "healthy" or "wait" can cost a crop. Keep a human in the loop and surface uncertainty (the advisory does).

## Ethical considerations
Farmers may over-trust automated advice. The system therefore reports calibrated confidence, escalates unknown/uncertain
cases to an agronomist, and carries a disclaimer on every advisory. Avoid advertising accuracy figures without stating the
dataset, split protocol and lab-vs-field caveat.
