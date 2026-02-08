import torch
from pathlib import Path

from .context_unet import ContextUnet
from ..hyperparams import Hyperparameters as hp

class DDPMGenerator():
    def __init__(self):
        super().__init__()
        self.model = ContextUnet(in_channels=1)

        # construct DDPM noise schedule
        b_t = (hp.beta2 - hp.beta1) * torch.linspace(0, 1, hp.timesteps + 1, device=hp.DEVICE) + hp.beta1
        a_t = 1 - b_t
        ab_t = torch.cumsum(a_t.log(), dim=0).exp()    
        ab_t[0] = 1

        self.b_t = b_t
        self.a_t = a_t
        self.ab_t = ab_t

    def load_weights(self):
        self.model.load_state_dict(torch.load('weights/ddpm.pt'))

    def sample_with_context(self, x, c=None):

        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(x.shape[0], hp.n_channels, hp.height, hp.height).to(hp.DEVICE)  

        # array to keep track of generated steps for plotting
        intermediates = [] 
        save_rate = 20

        # for each timestep
        for i in range(hp.timesteps, 0, -1):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / hp.timesteps])[:, None, None, None].to(hp.DEVICE)

            # sample some random noise to inject back in. For i = 1, don't add back in noise
            z = torch.randn_like(samples) if i > 1 else 0

            eps = self.model(samples, t, c)    # predict noise e_(x_t,t, ctx)
            samples = self._denoise(samples, i, eps, z)
            if i % save_rate==0 or i==hp.timesteps or i<8:
                intermediates.append(samples.squeeze(0).detach().cpu())

        intermediates = torch.stack(intermediates)
        return samples, intermediates
    
    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
    def _denoise(self, x, t, pred_noise, z=None):
        if z is None:
            z = torch.randn_like(x)
        noise = self.b_t.sqrt()[t] * z
        mean = (x - pred_noise * ((1 - self.a_t[t]) / (1 - self.ab_t[t]).sqrt())) / self.a_t[t].sqrt() # TO-DO: check if 1-a_t[t] can be replaced with b_t[t]
        return mean + noise
        