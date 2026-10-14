# Interview guide

## 30-second pitch
"I built a tomato crop-health and irrigation advisor. A ResNet classifies leaf disease and estimates severity; a diffusion
model generates extra images for rare diseases, and I tested whether that beats simple oversampling on a leak-free test set;
a VAE trained on healthy leaves raises an alarm for diseases the classifier has never seen; and a spiking neural network
turns soil-moisture, temperature and humidity into an irrigation decision that can run on a cheap edge device. A fusion layer
combines both into one advisory, for example it tells you to water at the base rather than overhead when a fungal disease is
present. It's served through FastAPI and Streamlit, and I'm explicit about what the numbers do and don't show."

## Resume bullets (replace every [X] with a number from YOUR runs; delete any bullet you cannot back up)
- Built a multimodal crop-health pipeline (PyTorch, snnTorch, FastAPI, Docker): disease/severity CNN, class-conditional DDPM
  augmentation, healthy-only VAE anomaly screen, and a spiking-network irrigation controller fused into one advisory.
- Designed a leak-free evaluation (single manifest, synthetic data train-only, real held-out test) and showed diffusion
  augmentation changed rare-class F1 from [X] to [Y] versus [Z] for inverse-frequency oversampling.
- Calibrated the classifier with temperature scaling (ECE [X] -> [Y]) and added an unknown-disease alarm that caught [X]% of an
  unseen disease class at [Y]% false alarms.
- Trained an SNN with population/delta spike encoding for irrigation decisions on simulated sensor data: [X] macro-F1 vs [Y]
  for a GRU, with an estimated [Z]x lower op-count energy; exported to a NumPy runtime with unit-tested numerical parity.

## Questions you should expect (with the honest answer shape)

**Why a diffusion model instead of just classic augmentation?** Classic augmentation re-uses existing pixels; a generative
model can in principle add new variation. But it is not free: it needs training time, can produce off-class or memorised
images, and often does not beat oversampling. That is why I built the oversampling baseline and two quality gates, and why
I report the comparison either way.

**How do you know the synthetic images don't leak into your test results?** The diffusion model trains only on the train
split (after rare-class reduction), synthetic images are added to train only, and val/test are real images from a single
manifest; a test asserts the splits are disjoint.

**Why does the VAE score use the worst 10% of pixels?** Lesions are small and local; a global MSE averages them away. Top-k
pooling makes the score sensitive to local abnormality. The threshold comes from healthy validation images, so the false-alarm
rate is controlled by construction.

**Where does the VAE fail?** Lighting and background shifts, blurry reconstructions, and diseases with subtle colour change.
It's a screen, so it is combined with classifier confidence rather than used alone.

**Why is your classifier confidence trustworthy?** I fit one temperature on validation logits and report ECE before and after.
The advisory thresholds on that calibrated confidence.

**Your severity labels - where do they come from?** PlantVillage has none. I use lesion-area fraction from colour segmentation as
a proxy and state that. It works for spots and blights, poorly for viruses and mites. With expert grades I would replace it.

**Why an SNN? Isn't a GRU better?** Slightly, yes: 0.83 vs 0.81 macro-F1 on my simulated task (alert recall 0.75 vs 0.70). The
SNN's argument is sparse, event-driven compute (accumulates instead of MACs) for battery-powered field nodes: about 3.6x lower
by an op-count proxy, which I do not claim as measured power. The honest summary: close on accuracy, noticeably cheaper per
decision on neuromorphic-style hardware, and still unproven on real sensors.

**What did it take to make the SNN train?** Three things: a population-coded input so the network can resolve a 2% change in
soil moisture; a high-gain UP/DOWN change channel so faults and rain onsets produce distinctive spikes; and a readout averaged
over the end of the spike train, because the decision concerns the present. Full inverse-frequency class weights then fixed
recall on the rare alert class. Before those fixes it plateaued at ~0.63 macro-F1 while a gradient-boosting check scored 0.81 (it now reaches ~0.80),
which told me the task was learnable and the encoding was the bottleneck.

**Your sensor labels come from a rule. Isn't that circular?** Partly, and I say so. The model sees noisy, faulty observations,
not the hidden state the rule uses, so it must infer the state and detect probe faults; but the benchmark measures policy
imitation, not agronomic benefit. The loader accepts real CSVs so the same pipeline can train on logged data.

**How would this behave on real field photos?** Worse. PlantVillage is lab imagery. I would test on PlantDoc or my own phone
photos, fine-tune with field images and stronger augmentation, and re-calibrate the VAE threshold on deployment-like healthy leaves.

**How would you deploy the SNN on a microcontroller?** The NumPy export proves the dynamics are portable and tested for parity.
Next: int8 quantisation of weights, fixed-point membrane arithmetic, then a C implementation of the 96-step loop; the model has
about 8 k parameters, so memory is not the constraint.

**What would you do with another month?** Field-photo evaluation, conformal prediction for set-valued disease outputs, FID/KID
for the synthetic images, real irrigation logs, active learning from farmer corrections, and measured energy on a Pi/MCU.

## Demo script (3 minutes)
1. Open the dashboard, upload a late-blight leaf, choose a simulated field with high humidity: show the fused advisory
   (critical risk, "irrigate at the base").
2. Upload a healthy leaf with a dry field: "irrigate now".
3. Upload a leaf of a class that was held out in training: show `possible unknown disease` and escalation advice.
4. Show `artifacts/runs/ablation.md`, the diffusion sample grid and the confusion matrix, and say what the numbers do and don't prove.
