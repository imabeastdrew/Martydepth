#!/bin/bash
#SBATCH -J offline_model_evaluation
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A TRA24006

# --- Configuration ---
# PASTE THE W&B ARTIFACT PATH FOR THE OFFLINE TEACHER MODEL HERE
ARTIFACT_PATH="marty1ai/martydepth/offline_teacher_epoch_19_loss_0.23:v0"
OUTPUT_FILE="offline_evaluation_results.json"

# --- Environment Setup ---
export PROJECT_ROOT=$SLURM_SUBMIT_DIR
cd $PROJECT_ROOT
echo "Changed directory to project root: $PROJECT_ROOT"

# Load required TACC modules
module purge
module load gcc cuda python3
echo "--- Loaded Modules ---"
module list
echo "----------------------"

# --- Python Virtual Environment ---
VENV_PATH="$SCRATCH/python-envs/martydepth"
echo "Python virtual environment path: $VENV_PATH"

if [ ! -d "$VENV_PATH" ]; then
    echo "ERROR: Virtual environment not found at $VENV_PATH"
    exit 1
else
    echo "Activating existing virtual environment..."
    source $VENV_PATH/bin/activate
fi

# --- WandB Setup ---
API_KEY_PATH="$PROJECT_ROOT/wandb_api_key.txt"
if [ -f "$API_KEY_PATH" ]; then
    export WANDB_API_KEY=$(cat "$API_KEY_PATH")
    echo "W&B API Key exported."
else
    echo "ERROR: W&B API key file not found at $API_KEY_PATH"
    exit 1
fi

# --- Run Evaluation ---
echo "Starting OFFLINE model evaluation for artifact: $ARTIFACT_PATH"
python3 -m src.evaluation.run_offline_evaluation \
    --artifact_path "$ARTIFACT_PATH" \
    --output_file "$PROJECT_ROOT/$OUTPUT_FILE" \
    --data_dir "$PROJECT_ROOT/data/interim" \
    --batch_size 64 \
    --num_workers 8

echo "Evaluation script finished. Results saved to $PROJECT_ROOT/$OUTPUT_FILE" 