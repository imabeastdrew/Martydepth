#!/bin/bash
#SBATCH -J shap_visualization
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 00:30:00
#SBATCH -A TRA24006

# --- Environment Setup ---
# Set the project root to the directory where the sbatch command was submitted
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
# Activate the same environment used for training
VENV_PATH="$SCRATCH/python-envs/martydepth"
echo "Activating Python virtual environment: $VENV_PATH"
source $VENV_PATH/bin/activate

# --- WandB Setup ---
# Export W&B API Key so the script can download the artifact
API_KEY_PATH="$PROJECT_ROOT/wandb_api_key.txt"
if [ -f "$API_KEY_PATH" ]; then
    export WANDB_API_KEY=$(cat "$API_KEY_PATH")
    echo "W&B API Key exported."
else
    echo "ERROR: W&B API key file not found at $API_KEY_PATH"
    exit 1
fi

# --- Run Visualization ---
echo "Starting SHAP visualization script..."
python3 -m src.evaluation.visualize_model \
    "marty1ai/martydepth/contrastive_reward-mx8rtyja:v7" \
    "data/interim" \
    --split test \
    --task shap \
    --save_path "shap_visualization.png"

echo "Visualization script finished."
echo "Output saved to: $PROJECT_ROOT/shap_visualization.png" 