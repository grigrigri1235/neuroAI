#!/bin/bash
#SBATCH --cpus-per-task=8 # number of cores (threads)
#SBATCH --gres=gpu:L40:1 # Request 1 gpu type L40 (Newton's name for the L40S 47GB)
#SBATCH --mem=48G # GSM8K train json + two GPT-2s
#SBATCH --time=10:00:00 # expected ~3-4 h
#SBATCH --mail-user=eran.b@campus.technion.ac.il
#SBATCH --mail-type=ALL # Valid values are NONE, BEGIN, END, FAIL, REQUEUE, ALL
#SBATCH --job-name="simcot_ft"
#SBATCH -o ./logs/out_job%j.txt # stdout goes to logs/out_job<id>.txt (relative to where you run sbatch)
#SBATCH -e ./logs/err_job%j.txt # stderr goes to logs/err_job<id>.txt

# SIM-CoT GPT-2 short fine-tune: own steps -> train FT-mix -> train FT-plain -> test (L, T2, T2-gold).
# Plan: docs/planning/simcot_finetune_plan.md. Safe to resubmit: finished parts are skipped.
#   cd /home/eran.b/neuroAI && sbatch coconut/experiments/simcot_ft.slurm

module purge # clean active modules list
source /home/eran.b/miniconda3/etc/profile.d/conda.sh # needed for "conda activate" in a batch job
conda activate neuroAI_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export USE_TF=0 # transformers: never import TensorFlow
export HF_HUB_OFFLINE=1 # checkpoint + GPT-2 are already in the HF cache
export TOKENIZERS_PARALLELISM=false

cd "$SLURM_SUBMIT_DIR/coconut" || exit 1
[ -f coconut.py ] || { echo "FAILED: run sbatch from the repo root (/home/eran.b/neuroAI)"; exit 1; }
for f in data/gsm_train.json data/gsm_valid.json data/gsm_test.json results/simcot/simcot_run.json; do
    [ -f "$f" ] || { echo "FAILED: $f missing"; exit 1; }
done

echo "job $SLURM_JOB_ID on $(hostname) at $(date)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "FAILED: torch sees no GPU"; exit 1; }

run_step () { # name, then the python command; log in logs/simcot_ft_<name>_<job>.log
    local name=$1; shift
    echo "[$(date +%T)] start $name"
    if python "$@" > "$SLURM_SUBMIT_DIR/logs/simcot_ft_${name}_${SLURM_JOB_ID}.log" 2>&1; then
        echo "[$(date +%T)] done  $name"
    else
        echo "[$(date +%T)] FAILED: $name, see logs/simcot_ft_${name}_${SLURM_JOB_ID}.log"
        exit 1
    fi
}

# step 0 continues a partial file by itself; it is fast if already complete
run_step steps experiments/simcot_ft_steps.py
[ -f checkpoints_ft/mix.pt ] && echo "skip train_mix (checkpoints_ft/mix.pt exists)" || run_step train_mix experiments/simcot_finetune.py --mode mix
[ -f checkpoints_ft/plain.pt ] && echo "skip train_plain (checkpoints_ft/plain.pt exists)" || run_step train_plain experiments/simcot_finetune.py --mode plain
run_step eval experiments/simcot_ft_eval.py
