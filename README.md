# PyTorch CUDA Starter

Minimal setup for running PyTorch with CUDA on an NVIDIA GPU (RTX 3050 Laptop GPU).

## Requirements
- Python 3.12
- Virtual environment (venv)
- PyTorch 2.x with CUDA 12.6

## Setup
```bash
git clone https://github.com/roblem28/pytorch-cuda-starter.git
cd pytorch-cuda-starter
py -3.12 -m venv .venv312
.\.venv312\Scripts\activate
pip install -r requirements.txt
