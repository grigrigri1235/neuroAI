#!/bin/bash
#SBATCH --cpus-per-task=8 # number of cores (threads)
#SBATCH --gres=gpu:L40:1 # Request 1 gpu type L40 (Newton's name for the L40S 47GB)
#SBATCH --mem=32G # two GPT-2s
#SBATCH --time=02:00:00 # expected ~15-30 min
#SBATCH --mail-user=eran.b@campus.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="simcot_run"
#SBATCH -o ./logs/out_job%j.txt # stdout goes to logs/out_job<id>.txt (relative to where you run sbatch)
#SBATCH -e ./logs/err_job%j.txt # stderr goes to logs/err_job<id>.txt

# SIM-CoT GPT-2: gate (G0-G2) + the minimal 2-round interleaving test (L, T2, T2-gold).
# Plan: docs/planning/simcot_interleaved_plan.md. Submit from the repo root, after the checkpoint is downloaded:
#   cd /home/eran.b/neuroAI && sbatch coconut/experiments/simcot_run.slurm

module purge # clean active modules list
source /home/eran.b/miniconda3/etc/profile.d/conda.sh # needed for "conda activate" in a batch job
conda activate neuroAI_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export USE_TF=0 # transformers: never import TensorFlow
export HF_HUB_OFFLINE=1 # checkpoint + GPT-2 are already in the HF cache (setup step)
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR/coconut" || exit 1
[ -f coconut.py ] || { echo "FAILED: run sbatch from the repo root (/home/eran.b/neuroAI)"; exit 1; }

echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FAILED: torch sees no GPU"; exit 1; }

echo "[$(date +%T)] start simcot_run"
if python experiments/simcot_run.py > "$SLURM_SUBMIT_DIR/logs/simcot_run_${SLURM_JOB_ID}.log" 2>&1; then
    echo "[$(date +%T)] done  simcot_run"
else
    echo "[$(date +%T)] FAILED: simcot_run, see logs/simcot_run_${SLURM_JOB_ID}.log"
    exit 1
fi
