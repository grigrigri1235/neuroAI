#!/bin/bash
#SBATCH --cpus-per-task=8 # number of cores (threads)
#SBATCH --gres=gpu:L40:1 # Request 1 gpu type L40 (Newton's name for the L40S 47GB; "gpu:L40S:1" gave "node configuration not available")
#SBATCH --mem=32G # RAM for up to 5 python processes at once (added, not in the Newton template)
#SBATCH --time=04:00:00 # expected ~1-1.5 hours (added, not in the Newton template)
#SBATCH --mail-user=eran.b@campus.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="coconut_gpt2"
#SBATCH -o ./logs/out_job%j.txt # stdout goes to logs/out_job<id>.txt (relative to where you run sbatch)
#SBATCH -e ./logs/err_job%j.txt # stderr goes to logs/err_job<id>.txt

# Phase 5: GPT-2 Coconut on GSM8K. Plan: docs/planning/gpt2_coconut_phase5_impl.md
# Submit from the repo root (logs/ must exist first):
#   cd /home/eran.b/neuroAI && mkdir -p logs && sbatch coconut/experiments/run_gpt2.slurm
# Order: smoke checks -> smoke runs (20 questions) -> full native (Part 1, alone)
#        -> full S1..S5 (Part 2, in parallel on the one GPU) -> score.
# The job stops at the first failure. Each python step has its own log: logs/gpt2_<step>_<jobid>.log

module purge # clean active modules list
source /home/eran.b/miniconda3/etc/profile.d/conda.sh # needed for "conda activate" in a batch job
conda activate neuroAI_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export USE_TF=0 # transformers: never import TensorFlow
export HF_HUB_OFFLINE=1 # GPT-2 + checkpoint are already in the HF cache (setup step)
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1 # up to 5 processes share the 8 cores

LOGS=$SLURM_SUBMIT_DIR/logs
JOB=$SLURM_JOB_ID
cd "$SLURM_SUBMIT_DIR/coconut" || exit 1
[ -f coconut.py ] || { echo "FAILED: run sbatch from the repo root (/home/eran.b/neuroAI)"; exit 1; }

echo "job $JOB on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch, transformers; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.is_available(), '| transformers', transformers.__version__)"
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FAILED: torch sees no GPU"; exit 1; }

step() { # step <name> <script> [args...]: run one python step, stop the job if it fails
    local name=$1; shift
    echo "[$(date +%T)] start $name"
    if ! python "$@" > "$LOGS/gpt2_${name}_${JOB}.log" 2>&1; then
        echo "[$(date +%T)] FAILED: $name, see logs/gpt2_${name}_${JOB}.log"
        exit 1
    fi
    echo "[$(date +%T)] done  $name"
}

# 1) smoke test: checks C0-C8, then the first 20 questions of every run
SMOKE=results/gpt2/smoke
step smoke_checks experiments/smoke_test.py --out $SMOKE/smoke_report.json
step smoke_native experiments/run_native.py --limit 20 --out $SMOKE/native.json
for S in S1 S2 S3 S4 S5; do
    step smoke_$S experiments/run_fewshot.py --setting $S --limit 20 --out $SMOKE/fewshot_$S.json
done
step smoke_score experiments/score.py --dir $SMOKE

# 2) Part 1, full test set, alone (so its time per question is not slowed by other runs)
step full_native experiments/run_native.py

# 3) Part 2, full test set: S1..S5 in parallel on the same GPU (GPT-2 is small)
echo "[$(date +%T)] start full S1..S5 (parallel)"
pids=(); names=()
for S in S1 S2 S3 S4 S5; do
    python experiments/run_fewshot.py --setting $S > "$LOGS/gpt2_full_${S}_${JOB}.log" 2>&1 &
    pids+=($!); names+=($S)
done
failed=0
for i in "${!pids[@]}"; do
    if wait "${pids[$i]}"; then
        echo "[$(date +%T)] done  full_${names[$i]}"
    else
        echo "[$(date +%T)] FAILED: full_${names[$i]}, see logs/gpt2_full_${names[$i]}_${JOB}.log"
        failed=1
    fi
done
[ "$failed" = 0 ] || exit 1

# 4) score the full run -> coconut/results/gpt2/results.json
step full_score experiments/score.py --dir results/gpt2
echo "all done at $(date)"
