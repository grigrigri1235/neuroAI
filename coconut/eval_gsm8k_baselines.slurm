#!/bin/bash
#SBATCH --job-name=eval_gsm8k_baselines
#SBATCH --output=slurm_logs/eval_gsm8k_baselines_%j.out
#SBATCH --error=slurm_logs/eval_gsm8k_baselines_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

echo "Starting GSM8K Baselines for Qwen2.5-1.5B Coconut"

source /home/eran.b/miniconda3/etc/profile.d/conda.sh
conda activate neuroAI_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXXABI_1_3_15_FOUND=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0

cd /home/eran.b/neuroAI/coconut

echo "Running GSM8K Baselines..."
python eval_qwen_1.5b_gsm8k_baselines.py

echo "Scoring All Baselines..."
export CUDA_VISIBLE_DEVICES=""
python score_baselines.py > score_baselines_results.txt

echo "GSM8K Baselines complete!"
