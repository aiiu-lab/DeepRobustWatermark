# +
import torch
import torch.nn as nn
import numpy as np
from model.vit_mask import ViT
from model.generator import Generator

class Cascade(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self,FLAGS, accelerator):
        super(Cascade, self).__init__()

        self.vit  = ViT(
                image_size=FLAGS.image_size,
                patch_size=8,
                dim=256,
                depth=6,
                heads=12,
                mlp_dim=256,
                dropout=0.0,
                emb_dropout=0.0,
                accelerator=accelerator
                )
        self.cnn = Generator()

    def forward(self, enc_image):
        if np.random.random() < 0.5:
            adv_images = self.cnn(enc_image)
            return self.vit(adv_images)
        else:
            adv_images = self.vit(enc_image)
            return self.cnn(adv_images)



# -


