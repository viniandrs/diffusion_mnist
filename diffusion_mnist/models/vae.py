import torch
from torch import nn
from torch.nn import functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(in_channels, out_channels, kernel_size),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        return self.model(x)

class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # (Batch_Size, 3, Height, Width) -> (Batch_Size, 64, Height / 2, Width / 2)
            nn.Conv2d(3, 64, 3, 2, 1),
            nn.GELU(),
            # (Batch_Size, 64, Height / 2, Width / 2) -> (Batch_Size, 128, Height / 4, Width / 4)
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.GELU(),
            # (Batch_Size, 128, Height / 4, Width / 4) -> (Batch_Size, 256, Height / 8, Width / 8)
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.GELU(),
            # (Batch_Size, 256, Height / 8, Width / 8) -> (Batch_Size, 512, Height / 16, Width / 16)
            nn.Conv2d(256, 512, 3, 2, 1),
            nn.GELU(),
            # (Batch_Size, 512, Height / 16, Width / 16) -> (Batch_Size, 1024, Height / 32, Width / 32)
            nn.Conv2d(512, 1024, 3, 2, 1),
            nn.GELU(),
            # (Batch_Size, 1024, Height / 32, Width / 32) -> (Batch_Size, 512, Height / 32, Width / 32)
            nn.Conv2d(1024, 512, 3, 1, 1),
            nn.GELU(),
            # (Batch_Size, 512, Height / 32, Width / 32) -> (Batch_Size, 256, Height / 32, Width / 32)
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.GELU(),
            # (Batch_Size, 256, Height / 32, Width / 32) -> (Batch_Size, 128, Height / 32, Width / 32)
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.G
        )


    def forward(self, x, noise):
        # x: (Batch_Size, Channel, Height, Width)
        # noise: (Batch_Size, 4, Height / 8, Width / 8)

        for module in self:

            if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
                # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
                # Pad with zeros on the right and bottom.
                # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
                x = F.pad(x, (0, 1, 0, 1))
            
            x = module(x)
        # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
        mean, log_variance = torch.chunk(x, 2, dim=1)
        # Clamp the log variance between -30 and 20, so that the variance is between (circa) 1e-14 and 1e8. 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        log_variance = torch.clamp(log_variance, -30, 20)
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        variance = log_variance.exp()
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        stdev = variance.sqrt()
        
        # Transform N(0, 1) -> N(mean, stdev) 
        # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = mean + stdev * noise
        
        # Scale by a constant
        # Constant taken from: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/configs/stable-diffusion/v1-inference.yaml#L17C1-L17C1
        x *= 0.18215
        
        return x