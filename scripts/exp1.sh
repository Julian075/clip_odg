#!/bin/bash
#SBATCH --job-name=exp1             # Name of your job
#SBATCH --output=logs/%x_%j.out            # Output file (%x for job name, %j for job ID)
#SBATCH --error=logs/%x_%j.err             # Error file
#SBATCH --partition=mm       # Select a partition to submit
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --cpus-per-task=8             # Request 8 CPU cores
#SBATCH --mem=32G                     # Request 32 GB of memory
#SBATCH --time=24:00:00               # Time limit for the job (hh:mm:ss)

# Print job details
echo "Starting job on node: $(hostname)"
echo "Job started at: $(date)"

# Define variables for your job
DATA_DIR="./data/officehome"
BATCH_SIZE=32
NUM_WORKERS=8
PRUNE_PERCENTAGE=10
MODEL_NAME="ViT-B/16"
OUTPUT_DIR="results"
EXPERIMENT_NAME="clip_odg_exp1"

# Activate the environment
source ~/anaconda3/etc/profile.d/conda.sh
conda activate clip_odg

# Execute the Python script with specific arguments
srun python main.py \
    --data_dir $DATA_DIR \
    --model_name $MODEL_NAME \
    --prune_percentage $PRUNE_PERCENTAGE \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --output_dir $OUTPUT_DIR \
    --experiment_name $EXPERIMENT_NAME

# Print job completion time
echo "Job finished at: $(date)"
