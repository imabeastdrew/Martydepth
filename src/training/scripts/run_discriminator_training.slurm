#!/bin/bash
#SBATCH -J discriminator_reward_training
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 12:00:00
#SBATCH -A TRA24006

# --- Environment Setup ---
# Set the project root to the directory where the sbatch command was submitted
export PROJECT_ROOT=$SLURM_SUBMIT_DIR
cd $PROJECT_ROOT
echo "Changed directory to project root: $PROJECT_ROOT"

# Load required TACC modules (or equivalent for your cluster)
module purge
module load gcc cuda python3
echo "--- Loaded Modules ---"
module list
echo "----------------------"

# --- Python Virtual Environment ---
# Using a common env path, e.g., in $SCRATCH
VENV_PATH="$SCRATCH/python-envs/martydepth"
echo "Python virtual environment path: $VENV_PATH"

if [ ! -d "$VENV_PATH" ]; then
    echo "Creating new virtual environment..."
    python3 -m venv $VENV_PATH
    source $VENV_PATH/bin/activate
    
    echo "Upgrading pip and setuptools..."
    python3 -m pip install --upgrade pip setuptools wheel
    
    echo "Installing PyTorch..."
    python3 -m pip install torch torchvision torchaudio
    
    echo "Installing dependencies from requirements.txt..."
    python3 -m pip install -r $PROJECT_ROOT/requirements.txt
    
    echo "Installing project in editable mode..."
    python3 -m pip install -e $PROJECT_ROOT
else
    echo "Activating existing virtual environment..."
    source $VENV_PATH/bin/activate
fi

# --- WandB Setup ---
# Export W&B API Key
API_KEY_PATH="$PROJECT_ROOT/wandb_api_key.txt"
if [ -f "$API_KEY_PATH" ]; then
    export WANDB_API_KEY=$(cat "$API_KEY_PATH")
    echo "W&B API Key exported."
else
    echo "ERROR: W&B API key file not found at $API_KEY_PATH"
    exit 1
fi

# --- Run Training ---
echo "Starting discriminative reward model training..."
python3 $PROJECT_ROOT/src/training/train_discriminator.py \
    --config $PROJECT_ROOT/src/training/configs/discriminator_reward_base.yaml

echo "Training script finished." 