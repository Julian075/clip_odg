# CLIP Feature Analysis

This project analyzes the features extracted from different layers of a CLIP model, focusing on domain generalization aspects.

## Project Structure

```
clip_odg/
├── main.py                 # Main script to run all analyses
├── src/
│   ├── analysis/          # Analysis scripts
│   │   ├── analyze_clustering.py
│   │   ├── analyze_tsne_by_class.py
│   │   └── analyze_latent_features.py
│   ├── feature_extraction/ # Feature extraction scripts
│   │   ├── compute_clip.py
│   │   └── compute_acc_from_pt_features.py
│   └── visualization/     # Visualization utilities
├── utils/
│   ├── dataset.py         # Dataset loading utilities
│   └── metrics.py         # Evaluation metrics
├── models/
│   └── pruned_clip.py     # Pruned CLIP implementation
├── data/
│   ├── officehome/        # Office-Home dataset
│   └── features/          # Extracted features
├── visualizations/
│   └── officehome/        # Plots for Office-Home dataset (organized by dataset)
├── results/
│   └── officehome/        # Results for Office-Home dataset (organized by dataset)
└── logs/                # Log files
```

## Setup

1. Create a conda environment:
```bash
conda create -n clip_odg python=3.8
conda activate clip_odg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The main script (`main.py`) provides several options for analyzing CLIP features:

### Basic Usage

```bash
python main.py
```

This will run both single-class and all-classes analysis with default parameters.

### Command Line Arguments

- `--data_dir`: Path to dataset directory (default: 'data/officehome')
- `--features_file`: Path to features file (default: 'clip_block_features_20250515_092209.pt')
- `--domains`: List of domains to analyze (default: ['art', 'clipart', 'product', 'real_world'])
- `--analysis_type`: Type of analysis to perform (choices: 'single_class', 'all_classes', 'both', default: 'both')
- `--class_idx`: Class index to analyze for single-class analysis (default: 0)
- `--max_samples`: Maximum number of samples to use per class (default: 1000)
- `--blocks`: List of blocks to analyze (default: all 12 blocks)
- `--vis_dir`: Directory to save visualizations (default: 'visualizations')
- `--timestamp`: Add timestamp to output filenames

### Examples

1. Analyze a specific class:
```bash
python main.py --analysis_type single_class --class_idx 5
```

2. Analyze all classes with a maximum of 500 samples per class:
```bash
python main.py --analysis_type all_classes --max_samples 500
```

3. Analyze specific blocks:
```bash
python main.py --blocks 0 1 2 3 4 5
```

4. Add timestamp to output files:
```bash
python main.py --timestamp
```

## Output

The script generates t-SNE visualizations in the specified visualization directory:
- For single-class analysis: `tsne_multiblock_domain_{class_name}_{timestamp}.png`
- For all-classes analysis: 
  - `tsne_multiblock_byclass_{timestamp}.png` (colored by class)
  - `tsne_multiblock_bydomain_{timestamp}.png` (colored by domain)

## Datasets

Este proyecto utiliza los siguientes datasets para experimentos de generalización de dominio:

- **OfficeHome**
- **Terra Incognita**

Ambos datasets fueron descargados desde el repositorio [DomainBed](https://github.com/facebookresearch/DomainBed) y no se incluyen en este repositorio debido a su tamaño.

**Nota:**
- Los archivos y carpetas de datos (`data/officehome/`, `data/terra_incognita/`, `OfficeHome.zip`, etc.) están listados en el `.gitignore` y no se suben al repositorio.
- La estructura esperada es:
  ```
  data/
    officehome/
      <domain>/<clase>/<imagen>
    terra_incognita/
      <domain>/<clase>/<imagen>
  ```

Para obtener los datos, sigue las instrucciones de descarga de [DomainBed](https://github.com/facebookresearch/DomainBed#download-datasets) y colócalos en la carpeta `data/` siguiendo la estructura anterior.

## Notas adicionales
- Los archivos de características, visualizaciones y modelos generados también están excluidos del repositorio.
- Consulta el `.gitignore` para ver todos los patrones excluidos. 