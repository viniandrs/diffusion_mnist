import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import diffusion_mnist.models as models
from diffusion_mnist.hyperparams import Hyperparameters

class DiffMNISTGenerator:
    def __init__(self, model: str, seed: int):

        self.model = None
        match model:
            case "DDPM":
                self.model = models.DDPMGenerator()
                self.model.load_state_dict(torch.load('weights/ddpm.pt'))
            case "DDIM":
                self.model = models.DDIMGenerator()
                self.model.load_state_dict(torch.load('weights/ddim.pt'))
            case "Dummy":
                self.model = models.DummyNN()
            case _:
                raise NotImplementedError("Model currently not supported")

    def generate(self, selected_digit: int):
        context = F.one_hot(torch.tensor([selected_digit]), 10).to(Hyperparameters.DEVICE).float()

        # x_T ~ N(0, 1), sample initial noise
        samples_0 = torch.randn(1, Hyperparameters.n_channels, Hyperparameters.height, Hyperparameters.height).to(Hyperparameters.DEVICE)  

        samples_unorm, intermediate_unorm = self.model(samples_0, context)

        # Convert the tensor to a PIL image
        pil_image = transforms.ToPILImage()(samples_unorm[0])
        pil_frames = [transforms.ToPILImage()(frame) for frame in intermediate_unorm]


        return pil_image, pil_frames