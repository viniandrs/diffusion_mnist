import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from .context_unet import ContextUnet
from ..hyperparams import Hyperparameters as hp

class DDIMGenerator(nn.Module):
    def __init__(self, weights_path: Path = 'weights/ddim.pt'):
        super(self).__init__()
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
        raise NotImplementedError()

    def sample_with_context(self, x, c=None):

        # x_T ~ N(0, 1), sample initial noise
        samples = torch.randn(x.shape[0], hp.n_channels, hp.height, hp.height).to(hp.DEVICE)  

        # array to keep track of generated steps for plotting
        intermediate = [] 
        save_rate = 20
        n = 20

        # array to keep track of generated steps for plotting
        intermediate = [] 
        step_size = hp.timesteps // n
        for i in range(hp.timesteps, 0, -step_size):
            print(f'sampling timestep {i:3d}', end='\r')

            # reshape time tensor
            t = torch.tensor([i / hp.timesteps])[:, None, None, None].to(hp.DEVICE)

            eps = self.model(samples, t, c)    # predict noise e_(x_t,t)
            samples = self._denoise(samples, i, i - step_size, eps)
            intermediate.append(samples.detach().cpu())

        intermediate = torch.stack(intermediate)
        return samples, intermediate
    
    # helper function; removes the predicted noise (but adds some noise back in to avoid collapse)
    def _denoise(self, x, t, t_prev, pred_noise):
        ab = self.ab_t[t]
        ab_prev = self.ab_t[t_prev]
        
        x0_pred = ab_prev.sqrt() / ab.sqrt() * (x - (1 - ab).sqrt() * pred_noise)
        dir_xt = (1 - ab_prev).sqrt() * pred_noise

        return x0_pred + dir_xt
        