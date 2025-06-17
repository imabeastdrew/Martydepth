#!/bin/bash
#SBATCH -J parallel_eval
#SBATCH -p gh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -t 01:00:00
#SBATCH -A TRA24006
#SBATCH -o logs/eval_%A_%a.out
#SBATCH -e logs/eval_%A_%a.err

# --- USAGE ---
# sbatch src/evaluation/scripts/evaluate_parallel.sbatch <MODEL_TYPE> <ARTIFACT_PATH> <NUM_SHARDS>
#
# <MODEL_TYPE>: 'online' or 'offline'
# <ARTIFACT_PATH>: Full W&B path to the model artifact
# <NUM_SHARDS>: The number of parallel jobs to run

# --- Script Arguments ---
MODEL_TYPE=$1
ARTIFACT_PATH=$2
NUM_SHARDS=$3

# --- Validate Arguments ---
if [ -z "$MODEL_TYPE" ] || [ -z "$ARTIFACT_PATH" ] || [ -z "$NUM_SHARDS" ]; then
    echo "ERROR: Missing arguments."
    echo "Usage: sbatch $0 <MODEL_TYPE> <ARTIFACT_PATH> <NUM_SHARDS>"
    exit 1
fi

if [ "$MODEL_TYPE" != "online" ] && [ "$MODEL_TYPE" != "offline" ]; then
    echo "ERROR: MODEL_TYPE must be 'online' or 'offline'"
    exit 1
fi

# Set the job array size
#SBATCH --array=0-$((NUM_SHARDS-1))

# --- Environment Setup ---
PROJECT_ROOT="/work/10539/drewtaylor635/vista/Martydepth"
VENV_PATH="$SCRATCH/python-envs/martydepth"
DATA_DIR="$PROJECT_ROOT/data/interim"
EVAL_SCRIPT_PATH="$PROJECT_ROOT/src/evaluation/evaluate_${MODEL_TYPE}.py"
LOG_DIR="$PROJECT_ROOT/logs"

# Create log directory if it doesn't exist
mkdir -p $LOG_DIR

echo "--- Job Info ---"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Job ID: $SLURM_ARRAY_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Model Type: $MODEL_TYPE"
echo "Artifact Path: $ARTIFACT_PATH"
echo "Number of Shards: $NUM_SHARDS"
echo "----------------"

# Load required TACC modules
module purge
module load gcc cuda python3
echo "--- Loaded Modules ---"
module list
echo "----------------------"

# Set Python Path
export PYTHONPATH=$PROJECT_ROOT

# Activate virtual environment
source $VENV_PATH/bin/activate

# Generate a unique ID for this entire evaluation run
EVAL_ID=$(python3 -c "import wandb; print(wandb.util.generate_id())")
echo "Evaluation Group ID: $EVAL_ID"

# --- Run Evaluation on the specified shard ---
echo "Starting evaluation for shard $SLURM_ARRAY_TASK_ID..."
python3 -m src.evaluation.evaluate_${MODEL_TYPE} \
    "$ARTIFACT_PATH" \
    "$DATA_DIR" \
    --num_shards $NUM_SHARDS \
    --shard_id $SLURM_ARRAY_TASK_ID \
    --eval_id $EVAL_ID

echo "Evaluation script finished for shard $SLURM_ARRAY_TASK_ID." 