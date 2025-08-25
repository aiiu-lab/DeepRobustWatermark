import torch
import torch.nn as nn
import torchvision.transforms as T

class Posterize(nn.Module):
    """
    Identity-mapping noise layer. Does not change the image
    """
    def __init__(self, bits=4):
        super(Posterize, self).__init__()
        self.posterize = T.RandomPosterize(bits=bits, p=1)

    def forward(self, noised_and_cover):
        noised = torch.clamp((noised_and_cover[0] + 1.0) / 2.0, 0, 1)
        noised = (noised * 255).to(dtype=torch.uint8)
        noised = self.posterize(noised).to(dtype=torch.float32)
        noised_and_cover[0] = noised / 255 * 2.0 - 1.0
        return noised_and_cover
        