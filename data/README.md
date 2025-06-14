# Data Directory

This directory contains the data used for the transformer models project.

## Directory Structure

- `raw/`: Contains the original, immutable data dump
  - This directory is git-ignored

- `interim/`: Contains intermediate data that has been transformed
  - Data that has been cleaned, filtered, or preprocessed
  - This directory is git-ignored

- `processed/`: Contains the final, canonical data sets for modeling
  - Data that is ready to be used by the models
  - This directory is git-ignored

## Data Setup Instructions

1. Place your `Hooktheory.json` file in the `data/raw/` directory
2. Run the data preprocessing scripts to generate the processed datasets
3. The processed data will be available in the `data/processed/` directory

## Data Processing

The data processing pipeline can be found in `src/data/`. The pipeline will:
1. Load the raw data from `data/raw/`
2. Perform necessary preprocessing
3. Save the processed data to `data/processed/` 