#!/bin/bash

# Create conda environment
conda create -n clip_odg python=3.10 -y

# Activate conda environment
conda activate clip_odg

# Install PyTorch and torchvision with CUDA support
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia -y

# Install other dependencies from requirements.txt
pip install -r requirements.txt

# Create necessary directories
mkdir -p results
mkdir -p data

echo "Environment setup complete!"
echo "To activate the environment, run: conda activate clip_odg" 