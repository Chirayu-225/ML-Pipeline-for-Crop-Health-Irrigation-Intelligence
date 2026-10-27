@echo off
rem Seeded diffusion-augmentation ablation at a low rare-class keep-fraction (Windows cmd version).
rem
rem   scripts\run_ablation_seeds.bat                         defaults: KEEP=0.05, SEEDS=0 1 2
rem   set KEEP=0.02
rem   set "SEEDS=0 1 2 3 4"
rem   set GATE_ABLATION=1                                    also train on synthetic images WITHOUT the teacher gate
rem   scripts\run_ablation_seeds.bat
rem
rem Fixed across seeds: the manifest split and the rare-class subset (data.rare_seed). Varies per seed: classifier
rem init + data order. The DDPM and the synthetic set are built ONCE per keep-fraction. Needs artifacts\manifest.csv.
setlocal
cd /d "%~dp0.."
if "%CFG%"=="" set CFG=configs/default.yaml
if "%PY%"=="" set PY=python
if "%KEEP%"=="" set KEEP=0.05
if "%SEEDS%"=="" set "SEEDS=0 1 2"
if "%GATE_ABLATION%"=="" set GATE_ABLATION=0

if not exist "artifacts\manifest.csv" (
  echo No artifacts\manifest.csv yet. First put PlantVillage under data\plantvillage, then run:
  echo   %PY% -m cropiq.data.manifest --config %CFG%
  exit /b 1
)
set RUNS=artifacts/runs_keep%KEEP%
set SYN=artifacts/synthetic_keep%KEEP%
set SYN_NOGATE=artifacts/synthetic_keep%KEEP%_nogate
set BASE=paths.runs_dir=%RUNS% paths.synthetic_dir=%SYN% data.rare_keep_fraction=%KEEP%
set FIRST=
for %%s in (%SEEDS%) do if not defined FIRST set FIRST=%%s

echo ^>^>^> keep-fraction %KEEP% ^| seeds: %SEEDS% ^| runs -^> %RUNS%
for %%s in (%SEEDS%) do (
  %PY% -m cropiq.training.train_classifier --config %CFG% --run-name clf_real_s%%s --override %BASE% seed=%%s || goto :fail
  %PY% -m cropiq.training.train_classifier --config %CFG% --run-name clf_balanced_s%%s --balanced-sampler --override %BASE% seed=%%s || goto :fail
)

rem Diffusion: trained once on the same (fixed) rare subset. Resumable if interrupted.
if not exist "%RUNS%/ddpm/best.pt" (
  %PY% -m cropiq.training.train_ddpm --config %CFG% --override %BASE% --resume || goto :fail
)
%PY% -m cropiq.training.generate_synthetic --config %CFG% --teacher %RUNS%/clf_real_s%FIRST%/best.pt --override %BASE% || goto :fail
set ARMS=clf_real clf_balanced clf_synthetic
if "%GATE_ABLATION%"=="1" (
  %PY% -m cropiq.training.generate_synthetic --config %CFG% --ddpm %RUNS%/ddpm/best.pt --override %BASE% paths.synthetic_dir=%SYN_NOGATE% || goto :fail
)
if "%GATE_ABLATION%"=="1" set "ARMS=%ARMS% clf_synthetic_nogate"

for %%s in (%SEEDS%) do (
  %PY% -m cropiq.training.train_classifier --config %CFG% --run-name clf_synthetic_s%%s --use-synthetic --balanced-sampler --override %BASE% seed=%%s || goto :fail
  if "%GATE_ABLATION%"=="1" (
    %PY% -m cropiq.training.train_classifier --config %CFG% --run-name clf_synthetic_nogate_s%%s --use-synthetic --balanced-sampler --override %BASE% paths.synthetic_dir=%SYN_NOGATE% seed=%%s || goto :fail
  )
)

%PY% -m cropiq.evaluation.ablation --config %CFG% --override %BASE% --arms %ARMS% --seeds %SEEDS% || goto :fail
echo ABLATION DONE: %RUNS%/ablation_seeds.md
exit /b 0

:fail
echo ABLATION FAILED (see the error above)
exit /b 1
