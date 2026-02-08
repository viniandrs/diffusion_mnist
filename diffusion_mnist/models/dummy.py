import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor

class DummyNN(nn.Module):
    def __init__(self):
        super().__init__()

    def sample_with_context(self, *args):
        samples = torch.randn(1, 1, 28, 28)
        intermediate = torch.randn(20, 1, 28, 28)
        return samples, intermediate
    
    def load_weights(self):
        pass