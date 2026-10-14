#!/bin/bash
#SBATCH --cpus-per-task=8 # number of cores (threads)
#SBATCH --gres=gpu:L40:1 # Request 1 gpu type L40 (Newton's name for the L40S 47GB)
#SBATCH --mem=64G # 7B model: ~15GB in bf16, loaded from fp32 shards
#SBATCH --time=02:00:00 # expected ~30-40 min
#SBATCH --mail-user=eran.b@campus.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="qwen_check"
#SBATCH -o ./logs/out_job%j.txt # stdout goes to logs/out_job<id>.txt (relative to where you run sbatch)
#SBATCH -e ./logs/err_job%j.txt # stderr goes to logs/err_job<id>.txt

# Pivot gate (plan: docs/planning/qwen7b_coconut_pivot.md):
#   1) qwen_check: R1 latent token ids + init, R2 latents change the output, 10 questions x S1-S5 preview
#   2) GPT-2 control: native prompts, latent slots without hidden-state feedback (plan D2), then re-score
# The two steps are independent: a failure in one does not stop the other.
# Submit from the repo root (Qwen must be in the HF cache first):
#   cd /home/eran.b/neuroAI && sbatch coconut/experiments/qwen_check.slurm

module purge # clean active modules list
source /home/eran.b/miniconda3/etc/profile.d/conda.sh # needed for "conda activate" in a batch job
conda activate neuroAI_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export USE_TF=0 # transformers: never import TensorFlow
export HF_HUB_OFFLINE=1 # models are already in the HF cache (setup step)
export TOKENIZERS_PARALLELISM=false

LOGS=$SLURM_SUBMIT_DIR/logs
JOB=$SLURM_JOB_ID
cd "$SLURM_SUBMIT_DIR/coconut" || exit 1
[ -f coconut.py ] || { echo "FAILED: run sbatch from the repo root (/home/eran.b/neuroAI)"; exit 1; }

echo "job $JOB on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FAILED: torch sees no GPU"; exit 1; }

failed=0
step() { # step <name> <script> [args...]: run one python step with its own log, remember failures
    local name=$1; shift
    echo "[$(date +%T)] start $name"
    if python "$@" > "$LOGS/${name}_${JOB}.log" 2>&1; then
        echo "[$(date +%T)] done  $name"
    else
        echo "[$(date +%T)] FAILED: $name, see logs/${name}_${JOB}.log"
        failed=1
    fi
}

step qwen_check experiments/qwen_check.py --out results/qwen7b/check/qwen_check.json
step gpt2_native_plain experiments/run_native.py --fill plain
step gpt2_score experiments/score.py --dir results/gpt2
echo "finished at $(date), failed=$failed"
[ "$failed" = 0 ] || exit 1
