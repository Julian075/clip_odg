# CLIP Pruning for Domain Generalization

This project implements layer pruning for CLIP's vision encoder to evaluate its impact on domain generalization performance.

## Setup

1. Create and activate the conda environment:
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup script
./setup.sh

# Activate environment
conda activate clip_odg
```

2. Download the Office-Home dataset:
- Visit [Office-Home Dataset](https://www.hemanthdv.org/officeHomeDataset.html)
- Download the dataset
- Extract it to the `data` directory

## Project Structure

```
.
├── data/                   # Dataset directory
├── models/                 # Model implementations
│   └── pruned_clip.py     # Pruned CLIP implementation
├── experiments/           # Experiment scripts
│   └── run_office_home.py # Office-Home evaluation
├── utils/                 # Utility functions
│   ├── dataset.py        # Dataset loading
│   └── metrics.py        # Evaluation metrics
├── results/              # Experiment results
├── requirements.txt      # Project dependencies
└── setup.sh             # Environment setup script
```

## Running Experiments

To run experiments on the Office-Home dataset:

```bash
python experiments/run_office_home.py --data_dir data/OfficeHome --output_dir results
```

Arguments:
- `--data_dir`: Path to Office-Home dataset (required)
- `--output_dir`: Directory to save results (default: 'results')
- `--batch_size`: Batch size for evaluation (default: 32)
- `--num_workers`: Number of workers for data loading (default: 4)
- `--device`: Device to run experiments on (default: 'cuda' if available, else 'cpu')

## Results

Results are saved in the specified output directory with a timestamp:
```
results/
└── office_home_YYYYMMDD_HHMMSS/
    └── results.json
```

The results file contains:
- Accuracy and F1 score for each pruning configuration
- Number of layers pruned
- Total layers and remaining layers 