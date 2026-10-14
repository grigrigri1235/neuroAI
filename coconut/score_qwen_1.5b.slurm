#!/bin/bash
#SBATCH --job-name=score_qwen_1.5b
#SBATCH --output=slurm_logs/score_qwen_1.5b_%j.out
#SBATCH --error=slurm_logs/score_qwen_1.5b_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:1

echo "Starting scoring for Qwen2.5-1.5B Coconut"

source /home/eran.b/miniconda3/etc/profile.d/conda.sh
conda activate neuroAI_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXXABI_1_3_15_FOUND=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0

cd /home/eran.b/neuroAI/coconut

echo "Scoring Outputs..."
export CUDA_VISIBLE_DEVICES=""
python score_outputs.py > score_qwen_1.5b_results.txt

echo "Scoring complete!"
