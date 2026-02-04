import torch
from dataclasses import dataclass

@dataclass
class Hyperparameters:
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    # data hyperparams
    num_classes: int = 10

    # diffusion hyperparams
    timesteps: int = 500

    # ddpm hyperparams
    beta1: float = 1e-4
    beta2: float = 0.02

    # model hyperparams
    n_feat: int = 64 # 64 hidden dimension feature
    n_cfeat: int = 10 # context vector is of size 10
    height: int = 28 # 28x28 image
    n_channels: int = 1
    save_dir: str = 'weights/'

    # training hyperparams
    n_epochs: int = 32
    batch_size: int = 100
    learning_rate: float = 1e-3