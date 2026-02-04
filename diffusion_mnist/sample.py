import torch

from diffusion_mnist.hyperparams import Hyperparameters, DEVICE

# sample with context using standard algorithm
@torch.no_grad()
def sample_ddpm_context(n_sample, context, save_rate=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, Hyperparameters.n_channels, Hyperparameters.height, Hyperparameters.height).to(DEVICE)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    for i in range(Hyperparameters.timesteps, 0, -1):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / Hyperparameters.timesteps])[:, None, None, None].to(DEVICE)

        # sample some random noise to inject back in. For i = 1, don't add back in noise
        z = torch.randn_like(samples) if i > 1 else 0

        eps = nn_model(samples, t, c=context)    # predict noise e_(x_t,t, ctx)
        samples = denoise_add_noise(samples, i, eps, z)
        if i % save_rate==0 or i==Hyperparameters.timesteps or i<8:
            intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate

@torch.no_grad()
def sample_ddim(n_sample, context, n=20):
    # x_T ~ N(0, 1), sample initial noise
    samples = torch.randn(n_sample, Hyperparameters.n_channels, Hyperparameters.height, Hyperparameters.height).to(DEVICE)  

    # array to keep track of generated steps for plotting
    intermediate = [] 
    step_size = Hyperparameters.timesteps // n
    for i in range(Hyperparameters.timesteps, 0, -step_size):
        print(f'sampling timestep {i:3d}', end='\r')

        # reshape time tensor
        t = torch.tensor([i / Hyperparameters.timesteps])[:, None, None, None].to(DEVICE)

        eps = nn_model(samples, t, c=context)    # predict noise e_(x_t,t)
        samples = denoise_ddim(samples, i, i - step_size, eps)
        intermediate.append(samples.detach().cpu().numpy())

    intermediate = np.stack(intermediate)
    return samples, intermediate