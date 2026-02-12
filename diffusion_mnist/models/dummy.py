import torch

class DummyNN():
    def __init__(self):
        super().__init__()

    def sample_with_context(self, *args, **kwargs):
        samples = torch.randn(1, 1, 28, 28)
        intermediate = torch.randn(20, 1, 28, 28)

        return samples, intermediate
    
    def load_weights(self):
        try:
            torch.load('weights/dummy.pt')
        except:
            raise FileNotFoundError('Weight file not found for model Dummy')