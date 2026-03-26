#!/bin/bash
#SBATCH -J my_gpu_job
#SBATCH -A MPHIL-DIS-SL2-GPU
#SBATCH -p ampere
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/home/zyw26/Machine-Learning-Enhanced-Inference-of-the-21-cm-Power-Spectrum-at-Cosmic-Dawn/logs/%j.log

PROJECT=/home/zyw26/Machine-Learning-Enhanced-Inference-of-the-21-cm-Power-Spectrum-at-Cosmic-Dawn

mkdir -p "$PROJECT/logs"
cd "$PROJECT" || exit 1

source "$PROJECT/VenvA1Cw/bin/activate"

echo "Job ID: $SLURM_JOB_ID"
echo "Host: $(hostname)"
which python
python --version
nvidia-smi -l 60 &
GPU_MONITOR_PID=$!


python -u optuna_optimize.py \
--device cuda \
--data-dir "$PROJECT/simulations" \
--output-dir "$PROJECT/optuna_outputs" \
--study-name emulator_optuna-recon \
--storage "sqlite:///$PROJECT/optuna_outputs/emulator_optuna_recon.db?timeout=60" \
--n-trials 100 \
--n-comp 9 \
--epochs 1500 \
--batch-size 512 \
--patience 500 \
--seed 1701 \
--loss-mode reconstruction \
--log-power

kill $GPU_MONITOR_PID