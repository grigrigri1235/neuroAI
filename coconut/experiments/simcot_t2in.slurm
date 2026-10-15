#!/bin/bash
#SBATCH --cpus-per-task=8 # number of cores (threads)
#SBATCH --gres=gpu:L40:1 # Request 1 gpu type L40 (Newton's name for the L40S 47GB)
#SBATCH --mem=32G # two GPT-2s
#SBATCH --time=01:00:00 # expected ~10 min
#SBATCH --mail-user=eran.b@campus.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="simcot_t2in"
#SBATCH -o ./logs/out_job%j.txt # stdout goes to logs/out_job<id>.txt (relative to where you run sbatch)
#SBATCH -e ./logs/err_job%j.txt # stderr goes to logs/err_job<id>.txt

# SIM-CoT GPT-2: step 1 inserted inside the question (L, T2-in, T2-in-gold).
# Plan: docs/planning/simcot_t2in_plan.md. Needs coconut/results/simcot/simcot_run.json (job 1449771).
#   cd /home/eran.b/neuroAI && sbatch coconut/experiments/simcot_t2in.slurm

module purge # clean active modules list
source /home/eran.b/miniconda3/etc/profile.d/conda.sh # needed for "conda activate" in a batch job
conda activate neuroAI_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export USE_TF=0 # transformers: never import TensorFlow
export HF_HUB_OFFLINE=1 # checkpoint + GPT-2 are already in the HF cache
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR/coconut" || exit 1
[ -f coconut.py ] || { echo "FAILED: run sbatch from the repo root (/home/eran.b/neuroAI)"; exit 1; }
[ -f results/simcot/simcot_run.json ] || { echo "FAILED: results/simcot/simcot_run.json missing (job 1449771)"; exit 1; }

echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FAILED: torch sees no GPU"; exit 1; }

echo "[$(date +%T)] start simcot_t2in"
if python experiments/simcot_t2in.py > "$SLURM_SUBMIT_DIR/logs/simcot_t2in_${SLURM_JOB_ID}.log" 2>&1; then
    echo "[$(date +%T)] done  simcot_t2in"
else
    echo "[$(date +%T)] FAILED: simcot_t2in, see logs/simcot_t2in_${SLURM_JOB_ID}.log"
    exit 1
fi
