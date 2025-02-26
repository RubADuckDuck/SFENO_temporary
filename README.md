# SFENO: Signal Flow Estimation based on Neural ODE

## Introduction
SFENO represents "**S**ignal **F**low **E**stimation based on **N**eural **O**DE", a flexible framework for modeling biological networks using ordinary differential equations (ODEs). SFENO provides a modular, hardware-agnostic approach to simulate and learn network dynamics.

## Project Structure
The project is organized as follows:
```
sfeno/
├── sfeno/                 # Core package code
├── scripts/               # Organized run scripts 
│   ├── training/          # Training scripts
│   ├── evaluation/        # Evaluation scripts
│   └── data/              # Data generation scripts
├── configs/               # Configuration files
├── manuscript/               # For Plotting
└── README.md
```

## Installation
```bash
pip install -e .
```

## Usage

### Data Preparation

#### Generate Synthetic Data
To generate synthetic network data for testing:
```bash
python scripts/data/generate_synthetic_data.py --network_size 100
```

#### Prepare Real Biological Data
First, place your experimental data files under the following path:
```
sfeno/datasets/(dataset_name)/conds.tsv
sfeno/datasets/(dataset_name)/exp.tsv
```

Then convert the data to SFENO format:
```bash
cd sfeno/datasets
python data_converter.py
```

This will create the following files:
```
sfeno/datasets/(dataset_name)/sfeno_data/conds.tsv   
sfeno/datasets/(dataset_name)/sfeno_data/expr.tsv   
sfeno/datasets/(dataset_name)/sfeno_data/node_Index.json
```

### Training

#### Train on Synthetic Data
You can train using either the Cellbox model or the Neural Network model:

**Cellbox Model:**
```bash
python scripts/training/train_synthetic.py --model cellbox --batch-size 16 --gpu 4 --epochs 4000 --ddp --network-size 100
```

**Neural Network Model:**
```bash
python scripts/training/train_synthetic.py --model nn --batch-size 16 --gpu 4 --epochs 4000 --ddp --network-size 100
```

#### Train on Real Biological Data
```bash
python scripts/training/train.py --batch-size 16 --gpu 4 --epochs 4000 --ddp --data-path sfeno/datasets/(dataset_name)/sfeno_data
```

#### Train on All Available Datasets
```bash
python scripts/training/train_all_datasets.py --batch-size 16 --gpu 4 --epochs 4000 --ddp
```

### Evaluation
Model predictions from test data will be saved under:
```
sfeno/datasets/(dataset_name)/results/(test_index)
```

To evaluate a pretrained model:
```bash
python scripts/evaluation/evaluate.py --checkpoint path/to/checkpoint.ckpt
```

## Model Types
SFENO supports multiple model types:
- **Cellbox**: Traditional ODE-based model with interaction matrices
- **NN**: Neural network-based ODE model

## Visualization
Refer to the manuscript for plotting instructions and result interpretation.