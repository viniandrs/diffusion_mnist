import torch
import torch.nn.functional as F

import diffusion_mnist.models as models
from diffusion_mnist.hyperparams import Hyperparameters as hp

class DiffMNISTGenerator:
    def __init__(self, model: str, seed: int):

        self.model = None
        match model:
            case "DDPM":
                self.model = models.DDPMGenerator()
            case "DDIM":
                self.model = models.DDIMGenerator()
            case "Dummy":
                self.model = models.DummyNN()
            case _:
                raise NotImplementedError("Model currently not supported")
            
        self.model.load_weights()

    def generate(self, digits: torch.Tensor):
        """
        Generate a batch of digits

        Params:
            digits : (batch, 1) = digits of the images to generate
        """

        # x_T ~ N(0, 1), sample initial noise
        samples_0 = torch.randn(digits.shape[0], hp.n_channels, hp.height, hp.height).to(hp.DEVICE)  

        samples_unorm = self.model.sample_with_context(samples_0, digits, grad=False)
        
        # normalizing tensors to [0, 1]
        samples_min = samples_unorm.view(samples_unorm.shape[0], -1).min(dim=1)[0][:,None,None,None]
        samples_max = samples_unorm.view(samples_unorm.shape[0], -1).max(dim=1)[0][:,None,None,None]
        samples = (samples_unorm - samples_min) / (samples_max - samples_min)

        return samples.numpy()

    # def animate(self, digit):
        # """
        # Generate an animation of the latents turning into the final image

        # Params:
        #     digit : (1, 1) = digit of the animation to generate
        # """

        # assert digit.shape[0] == 1, NotImplementedError("You can only animate one image at time")

        # context = F.one_hot(digit, 10).to(hp.DEVICE).float()

        # # x_T ~ N(0, 1), sample initial noise
        # samples_0 = torch.randn(digit.shape[0], hp.n_channels, hp.height, hp.height).to(hp.DEVICE)  

        # samples_unorm = self.model.sample_with_context(samples_0, context, grad=False)
        
        # # normalizing tensors to [0, 1]
        # samples_min = samples_unorm.view(samples_unorm.shape[0], -1).min(dim=1)[0]
        # samples_max = samples_unorm.view(samples_unorm.shape[0], -1).max(dim=1)[0]
        # samples = (samples_unorm - samples_min) / (samples_max - samples_min)

        # intermediates_min = intermediates_unorm.view(intermediates_unorm.shape[0], -1).min(dim=1)[0][:,None,None,None]
        # intermediates_max = intermediates_unorm.view(intermediates_unorm.shape[0], -1).max(dim=1)[0][:,None,None,None]
        # intermediates = (intermediates_unorm - intermediates_min) / (intermediates_max - intermediates_min)

        # # Convert the tensor to a numpy array
        # image = samples[0].numpy()
        # frames = [tensor.numpy()[0] for tensor in intermediates]


        # return frames