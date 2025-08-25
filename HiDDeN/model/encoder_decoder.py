import torch.nn as nn
from model.encoder import Encoder
from model.decoder import Decoder

class EncoderDecoder(nn.Module):
    """
    Combines Encoder->Noiser->Decoder into single pipeline.
    The input is the cover image and the watermark message. The module inserts the watermark into the image
    (obtaining encoded_image), then applies Noise layers (obtaining noised_image), then passes the noised_image
    to the Decoder which tries to recover the watermark (called decoded_message). The module outputs
    a three-tuple: (encoded_image, noised_image, decoded_message)
    """
    def __init__(self, FLAGS):

        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(FLAGS)
        self.decoder = Decoder(FLAGS)

   
    def encode(self, images, messages):
        return self.encoder(images, messages)
    
    def decode(self, encoded_images):
        return self.decoder(encoded_images)

