#!/bin/bash
#SBATCH --job-name=eval_iterative_all
#SBATCH --output=slurm_logs/eval_iterative_all_%j.out
#SBATCH --error=slurm_logs/eval_iterative_all_%j.err
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G

echo "Starting Unified Iterative Latent & Control Evaluation for Qwen2.5-1.5B"

source /home/eran.b/miniconda3/etc/profile.d/conda.sh
conda activate neuroAI_project

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export CXXABI_1_3_15_FOUND=1
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export USE_TF=0

cd /home/eran.b/neuroAI/coconut

echo "Step 1/4: Running GSM8K Iterative Latents..."
python eval_qwen_1.5b_gsm8k_less_handholding.py

echo "Step 2/4: Running GSM8K No-Latents Control..."
python eval_qwen_1.5b_gsm8k_no_latents.py

echo "Step 3/4: Running EntailmentBank Iterative Latents..."
python eval_qwen_1.5b_entailment_fewshot.py

echo "Step 4/4: Running EntailmentBank No-Latents Control..."
python eval_qwen_1.5b_entailment_no_latents.py

echo "Scoring All Outputs..."
export CUDA_VISIBLE_DEVICES=""
python score_outputs.py > score_outputs_results.txt

echo "All evaluations and scoring complete!"
