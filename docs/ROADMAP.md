# Build roadmap (about 4-5 weeks part-time)

| Week | Goal | Done when |
|---|---|---|
| 0 | Environment + smoke test | `make test` and `make smoke` pass; PlantVillage unpacked in `data/plantvillage` |
| 1 | Data + baselines | `make data`; look at the manifest summary and a few severity labels by eye; `make clf-real` and `make clf-balanced` give rare-class F1 numbers and confusion matrices |
| 2 | Diffusion | `make ddpm` (leave it running, `--resume` if interrupted); inspect `samples_epoch*.png`; `make synth` and read `generation_report.json`; `make clf-synth`; `make ablation` |
| 3 | VAE + open-set | `make vae`; set `data.holdout_classes` to one disease and re-run classifier + VAE evaluation; tune `vae.topk_frac` / `threshold_percentile` on validation only |
| 4 | SNN + edge | `make snn`; compare with the GRU; try your own sensor CSV; `make edge`; run `EdgeSNN` on a Raspberry Pi if you have one |
| 5 | Serve + polish | API, dashboard, Docker; fill README results; record a 2-minute demo; write the blog post / portfolio page |

## Tips for a 4 GB GPU
- Run classifier trainings first; they are cheap and give you the teacher model the diffusion step needs.
- Diffusion is the bottleneck: keep `image_size: 64`, `batch_size: 16`, `grad_accum: 2`, AMP on. If it is still too slow,
  reduce `base_channels` to 48 or `epochs` to 80 and compare sample grids; or run that one step on a free Colab/Kaggle GPU
  (copy `artifacts/manifest.csv` and the data, run `make ddpm`, download `artifacts/runs/ddpm/best.pt`).
- Watch VRAM with `nvidia-smi`; close other GPU apps (browsers use VRAM too).

## Stretch goals, ordered by value to a reviewer
1. **Field-image evaluation**: test on PlantDoc or your own phone photos from a farm; report the drop and one mitigation.
2. **Real sensor data**: a soil-moisture/weather logger CSV for even one field; compare with the simulator results.
3. **Uncertainty**: conformal prediction sets for the disease class; show coverage on the test split.
4. **Synthetic-image quality**: FID/KID (e.g. `torch-fidelity`) and a nearest-neighbour memorisation plot.
5. **Edge**: int8 quantisation of the SNN, a C port of `forward_spikes`, measured latency/power on a board.
6. **Classifier export**: ONNX / TorchScript, latency on CPU; Grad-CAM overlays in the dashboard.
7. **MLOps polish**: GitHub Actions running `pytest`, MLflow runs for all experiments, model registry for the checkpoints.
8. **Localisation**: crop calendar and advisory text in Kannada/Hindi for Indian growers; ask a local agronomist (or KVK
   officer) to review the advisory rules and quote them in the write-up.
