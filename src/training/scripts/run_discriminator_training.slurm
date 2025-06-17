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
# This assumes the job is submitted from the project root.
PROJECT_ROOT=$(pwd)
cd $PROJECT_ROOT || exit

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
WANDB_KEY_FILE="$PROJECT_ROOT/wandb_api_key.txt"
if [ ! -f "$WANDB_KEY_FILE" ]; then
    echo "ERROR: W&B API key file not found at $WANDB_KEY_FILE"
    exit 1
fi
export WANDB_API_KEY=$(cat "$WANDB_KEY_FILE")
echo "W&B API key configured."

# --- Run Training ---
echo "Starting discriminative reward model training..."
python3 $PROJECT_ROOT/src/training/train_discriminator.py \
    --config $PROJECT_ROOT/src/training/configs/discriminator_reward_base.yaml

echo "Training script finished." 