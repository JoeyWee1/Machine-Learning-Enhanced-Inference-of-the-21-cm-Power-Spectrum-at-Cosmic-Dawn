#!/bin/bash
#SBATCH -J train_fixed
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

python -u train_fixed.py \
--device cuda \
--data-dir "$PROJECT/simulations" \
--output-dir "$PROJECT/optuna_outputs" \
--n-comp 9 \
--num-layers 4 \
--hidden-dim 512 \
--lr 0.0016729307114133733 \
--weight-decay 4.125123892293413e-06 \
--epochs 10000 \
--batch-size 512 \
--patience 2500 \
--seed 1701

kill $GPU_MONITOR_PID