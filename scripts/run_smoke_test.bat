@echo off
rem End-to-end CPU smoke test on toy leaves + simulated sensors (Windows cmd version of run_smoke_test.sh).
rem Verifies every stage runs and hands artifacts to the next one. Metrics here are meaningless.
setlocal
cd /d "%~dp0.."
set CFG=configs/smoke.yaml
if "%PY%"=="" set PY=python

if exist artifacts_smoke rmdir /s /q artifacts_smoke
%PY% -m cropiq.data.toy --out artifacts_smoke/toy_plantvillage || goto :fail
%PY% -m cropiq.data.manifest --config %CFG% || goto :fail
%PY% -m cropiq.training.train_classifier --config %CFG% --run-name clf_real || goto :fail
%PY% -m cropiq.training.train_classifier --config %CFG% --run-name clf_balanced --balanced-sampler || goto :fail
%PY% -m cropiq.training.train_ddpm --config %CFG% || goto :fail
%PY% -m cropiq.training.generate_synthetic --config %CFG% --teacher artifacts_smoke/runs/clf_real/best.pt || goto :fail
%PY% -m cropiq.training.train_classifier --config %CFG% --run-name clf_synthetic --use-synthetic --balanced-sampler || goto :fail
%PY% -m cropiq.training.train_vae --config %CFG% || goto :fail
%PY% -m cropiq.training.train_snn --config %CFG% || goto :fail
%PY% -m cropiq.evaluation.ablation --config %CFG% || goto :fail
%PY% -m cropiq.edge.export_snn --ckpt artifacts_smoke/runs/snn/best.pt --out artifacts_smoke/edge || goto :fail

rem Fused advisory from the command line
%PY% -c "from cropiq.config import load_config; from cropiq.data.sensors import simulate_field, FEATURES; cfg = load_config('configs/smoke.yaml'); simulate_field(3, cfg).iloc[200:224][FEATURES].to_csv('artifacts_smoke/sample_sensors.csv', index=False)" || goto :fail
for %%f in (artifacts_smoke\toy_plantvillage\color\Tomato___Late_blight\*.jpg) do (
  set "LEAF=%%f"
  goto :gotleaf
)
:gotleaf
%PY% -m cropiq.pipeline.inference --config %CFG% --image "%LEAF%" --sensors artifacts_smoke/sample_sensors.csv || goto :fail
echo SMOKE TEST PASSED
exit /b 0

:fail
echo SMOKE TEST FAILED (see the error above)
exit /b 1
