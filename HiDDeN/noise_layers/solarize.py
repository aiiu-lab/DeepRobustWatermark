import torch.nn as nn
import torchvision.transforms as T

class Solarize(nn.Module):
    """
    Identity-mapping noise layer. Does not change the image
    """
    def __init__(self, threshold=192./255.):
        super(Solarize, self).__init__()
        self.solarize = T.RandomSolarize(threshold=threshold, p=1)

    def forward(self, noised_and_cover):
        noised_and_cover[0] = (noised_and_cover[0]+1.0)/ 2.0
        noised_and_cover[0] = self.solarize(noised_and_cover[0])
        noised_and_cover[0] = 2.0 * noised_and_cover[0] - 1.0
        return noised_and_cover
        