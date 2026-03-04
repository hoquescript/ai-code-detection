#!/bin/bash
#SBATCH --job-name=ast_svm
#SBATCH --partition=gpubase_bygpu_b5
#SBATCH --time=23:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --array=0-1
#SBATCH --output=logs/%x-%A_%a.out

module load python scipy-stack
VENV=$HOME/venvs/how-far-are-we

source ~/venvs/how-far-are-we/bin/activate

export HF_HOME=$HOME/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME/transformers

$VENV/bin/python -c "import torch; print(torch.__version__)"

# Pick dataset based on array index
if [ "$SLURM_ARRAY_TASK_ID" -eq 0 ]; then
  export DATA_CSV=data/java.csv
else
  export DATA_CSV=data/python.csv
fi

$VENV/bin/python scripts/embeddings/main.py
