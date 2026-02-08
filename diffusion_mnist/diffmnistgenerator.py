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
            case "DDIM":
                self.model = models.DDIMGenerator()
                self.model.load_state_dict(torch.load('weights/ddim.pt'))
            case "Dummy":
                self.model = models.DummyNN()
            case _:
                raise NotImplementedError("Model currently not supported")
            
        self.model.load_weights()

    def generate(self, selected_digit: int):
        context = F.one_hot(torch.tensor([selected_digit]), 10).to(Hyperparameters.DEVICE).float()

        # x_T ~ N(0, 1), sample initial noise
        samples_0 = torch.randn(1, Hyperparameters.n_channels, Hyperparameters.height, Hyperparameters.height).to(Hyperparameters.DEVICE)  

        samples_unorm, intermediates_unorm = self.model.sample_with_context(samples_0, context)
        
        # normalizing tensors to [0, 1]
        samples_min = samples_unorm.view(samples_unorm.shape[0], -1).min(dim=1)[0]
        samples_max = samples_unorm.view(samples_unorm.shape[0], -1).max(dim=1)[0]
        samples = (samples_unorm - samples_min) / (samples_max - samples_min)

        intermediates_min = intermediates_unorm.view(intermediates_unorm.shape[0], -1).min(dim=1)[0][:,None,None,None]
        intermediates_max = intermediates_unorm.view(intermediates_unorm.shape[0], -1).min(dim=1)[0][:,None,None,None]
        intermediates = (intermediates_unorm - intermediates_min) / (intermediates_max - intermediates_min)

        # Convert the tensor to a PIL image
        pil_image = transforms.ToPILImage()(samples[0])
        pil_frames = [transforms.ToPILImage()(tensor) for tensor in intermediates]


        return pil_image, pil_frames