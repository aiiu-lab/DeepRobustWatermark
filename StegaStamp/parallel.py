import torch.nn as nn
import numpy as np
from vit_mask import ViT
import torch
import models

class Parallel(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(Parallel, self).__init__()

        self.vit  = ViT(
                image_size=128,
                patch_size=8,
                dim=256,
                depth=6,
                heads=12,
                mlp_dim=256,
                dropout=0.0,
                emb_dropout=0.0
                )
        self.cnn = models.Generator()

    def forward(self, enc_image):
    
        if np.random.random() < 0.7:
            return self.vit(enc_image)
        else:
            return self.cnn(enc_image)




