# Toy Experiment on Federated Learning

# Requirements

- pytorch, torchvision
- tqdm
- tensorboard
- scipy

# Implementation detail

- FL update algorithm: FedAVG [![arXiv](https://img.shields.io/badge/arXiv-1602.05629-f9f107.svg)](https://arxiv.org/abs/1602.05629)
- Client selection strategy: Random, loss based sampling

# Running

```bash
python3 main.py
```

```
sbatch --nodelist=unistml8 fmnist.sh 
```

in detail refer `sample.sh`

## Argument


The `NaiveCNN` code from FedCor implementation