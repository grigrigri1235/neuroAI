#!/bin/bash
#SBATCH --cpus-per-task=8 # number of cores (threads)
#SBATCH --gres=gpu:L40:1 # Request 1 gpu type L40 (Newton's name for the L40S 47GB)
#SBATCH --mem=48G # 4B model, ~8GB in bf16
#SBATCH --time=02:00:00 # expected ~30-45 min
#SBATCH --mail-user=eran.b@campus.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="qwen3_check"
#SBATCH -o ./logs/out_job%j.txt # stdout goes to logs/out_job<id>.txt (relative to where you run sbatch)
#SBATCH -e ./logs/err_job%j.txt # stderr goes to logs/err_job<id>.txt

# Gate check G0-G3 for agurung/coconut-qwen3-4b-ff (plan: docs/planning/qwen3_4b_interleaved_plan.md).
# Submit from the repo root, after the model is in the HF cache:
#   cd /home/eran.b/neuroAI && sbatch coconut/experiments/qwen3_check.slurm

module purge # clean active modules list
source /home/eran.b/miniconda3/etc/profile.d/conda.sh # needed for "conda activate" in a batch job
conda activate neuroAI_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export USE_TF=0 # transformers: never import TensorFlow
export HF_HUB_OFFLINE=1 # model is already in the HF cache (setup step)
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR/coconut" || exit 1
[ -f coconut.py ] || { echo "FAILED: run sbatch from the repo root (/home/eran.b/neuroAI)"; exit 1; }

echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FAILED: torch sees no GPU"; exit 1; }

echo "[$(date +%T)] start qwen3_check"
if python experiments/qwen3_check.py > "$SLURM_SUBMIT_DIR/logs/qwen3_check_${SLURM_JOB_ID}.log" 2>&1; then
    echo "[$(date +%T)] done  qwen3_check"
else
    echo "[$(date +%T)] FAILED: qwen3_check, see logs/qwen3_check_${SLURM_JOB_ID}.log"
    exit 1
fi
