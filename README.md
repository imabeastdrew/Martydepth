# MartyDepth

Online chord prediction from melody using transformers.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate  # On Windows
```

2. Install the package:
```bash
pip install -e .  # Install in editable mode
pip install -e ".[dev]"  # Install with development dependencies
```

3. Set up Weights & Biases:
- Create a `wandb_api_key.txt` file in the project root
- Add your W&B API key to the file
- The file is gitignored for security

## Project Structure

```
martydepth/
├── data/
│   ├── raw/          # Original data
│   ├── interim/      # Processed data
│   └── processed/    # Final datasets
├── src/
│   ├── data/         # Data processing
│   ├── models/       # Model definitions
│   └── training/     # Training scripts
├── tests/            # Test files
└── notebooks/        # Jupyter notebooks
```

## Training

To train the model on TACC's Grace Hopper:

1. Update the configuration in `src/training/config.py`
2. Submit the SLURM job:
```bash
sbatch src/training/scripts/train.sh
```

## Development

- Format code: `black src tests`
- Sort imports: `isort src tests`
- Run tests: `pytest tests/`
- Lint code: `flake8 src tests`

