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
nvidia-smi

python -u optuna_optimize.py \
  --device cuda \
  --data-dir "$PROJECT/simulations" \
  --output-dir "$PROJECT/optuna_outputs"