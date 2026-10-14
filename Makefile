CFG ?= configs/default.yaml
SMOKE = configs/smoke.yaml
PY ?= python

.PHONY: install test smoke data clf-real clf-balanced ddpm synth clf-synth vae snn ablation edge api dashboard all-train

install:
	pip install -e ".[serve,dev]"

test:
	pytest -q

# End-to-end CPU smoke run on toy data (minutes). Proves the plumbing, NOT model quality.
smoke:
	bash scripts/run_smoke_test.sh

# ---- Real pipeline (needs PlantVillage under data/plantvillage; see README) ----
data:
	$(PY) -m cropiq.data.manifest --config $(CFG)

clf-real:
	$(PY) -m cropiq.training.train_classifier --config $(CFG) --run-name clf_real

clf-balanced:
	$(PY) -m cropiq.training.train_classifier --config $(CFG) --run-name clf_balanced --balanced-sampler

ddpm:
	$(PY) -m cropiq.training.train_ddpm --config $(CFG)

synth:
	$(PY) -m cropiq.training.generate_synthetic --config $(CFG) --teacher artifacts/runs/clf_real/best.pt

clf-synth:
	$(PY) -m cropiq.training.train_classifier --config $(CFG) --run-name clf_synthetic --use-synthetic --balanced-sampler

vae:
	$(PY) -m cropiq.training.train_vae --config $(CFG)

snn:
	$(PY) -m cropiq.training.train_snn --config $(CFG)

ablation:
	$(PY) -m cropiq.evaluation.ablation --config $(CFG)

edge:
	$(PY) -m cropiq.edge.export_snn --ckpt artifacts/runs/snn/best.pt --out artifacts/edge

all-train: data clf-real clf-balanced ddpm synth clf-synth vae snn ablation edge

api:
	CROPIQ_CONFIG=$(CFG) uvicorn cropiq.api.main:app --host 0.0.0.0 --port 8000

dashboard:
	CROPIQ_CONFIG=$(CFG) streamlit run src/cropiq/dashboard/app.py
