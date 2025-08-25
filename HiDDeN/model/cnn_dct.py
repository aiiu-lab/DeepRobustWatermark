import torch
import torch.nn as nn
import torchjpeg.dct as DCT
import numpy as np


def rgb2yuv(image_rgb, image_yuv_out):
    """ Transform the image from rgb to yuv """
    image_yuv_out[:, 0, :, :] = 0.299 * image_rgb[:, 0, :, :].clone() + 0.587 * image_rgb[:, 1, :, :].clone() + 0.114 * image_rgb[:, 2, :, :].clone()
    image_yuv_out[:, 1, :, :] = -0.14713 * image_rgb[:, 0, :, :].clone() + -0.28886 * image_rgb[:, 1, :, :].clone() + 0.436 * image_rgb[:, 2, :, :].clone()
    image_yuv_out[:, 2, :, :] = 0.615 * image_rgb[:, 0, :, :].clone() + -0.51499 * image_rgb[:, 1, :, :].clone() + -0.10001 * image_rgb[:, 2, :, :].clone()


def yuv2rgb(image_yuv, image_rgb_out):
    """ Transform the image from yuv to rgb """
    image_rgb_out[:, 0, :, :] = image_yuv[:, 0, :, :].clone() + 1.13983 * image_yuv[:, 2, :, :].clone()
    image_rgb_out[:, 1, :, :] = image_yuv[:, 0, :, :].clone() + -0.39465 * image_yuv[:, 1, :, :].clone() + -0.58060 * image_yuv[:, 2, :, :].clone()
    image_rgb_out[:, 2, :, :] = image_yuv[:, 0, :, :].clone() + 2.03211 * image_yuv[:, 1, :, :].clone()


class DCT_CNN(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self, accelerator):
        super(DCT_CNN, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 3, 3, stride=1, padding=1),
        )
        self.jpeg_mask = None
        yuv_keep_weights = (25,9,9)
        self.yuv_keep_weighs = yuv_keep_weights
        # self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.device = accelerator.device
        self.create_mask((1000, 1000))

    def create_mask(self, requested_shape):
        if self.jpeg_mask is None or requested_shape > self.jpeg_mask.shape[1:]:
            self.jpeg_mask = torch.empty((3,) + requested_shape, device=self.device)
            for channel, weights_to_keep in enumerate(self.yuv_keep_weighs):
                mask = torch.from_numpy(self.get_jpeg_yuv_filter_mask(requested_shape, 8, weights_to_keep))
                self.jpeg_mask[channel] = mask

    def get_jpeg_yuv_filter_mask(self,image_shape: tuple, window_size: int, keep_count: int):
        mask = np.zeros((window_size, window_size), dtype=np.uint8)

        index_order = sorted(((x, y) for x in range(window_size) for y in range(window_size)),
                             key=lambda p: (p[0] + p[1], -p[1] if (p[0] + p[1]) % 2 else p[1]))

        for i, j in index_order[0:keep_count]:
            mask[i, j] = 1

        return np.tile(mask, (int(np.ceil(image_shape[0] / window_size)),
                              int(np.ceil(image_shape[1] / window_size))))[0: image_shape[0], 0: image_shape[1]]

    def get_mask(self, image_shape):
        if self.jpeg_mask.shape < image_shape:
            self.create_mask(image_shape)
        # return the correct slice of it
        return self.jpeg_mask[:, :image_shape[1], :image_shape[2]].clone()


    def forward(self, enc_image):

        N, C, H, W = enc_image.shape
        B = H // 8
        image_yuv = torch.empty_like(enc_image)
        rgb2yuv(enc_image, image_yuv)
        block_dct = DCT.batch_dct(image_yuv) # [batch_size,3, H, W]
        mask = self.get_mask(block_dct.shape[1:])
        block_dct = torch.mul(block_dct, mask) # [batch_size,3, H, W]
        x = self.conv_layers(block_dct)  # [batch_size, 192 ,(H/8),(W/8)]
        x = DCT.batch_idct(x)
        #print("idctx:",x.shape)
        adv_image = torch.empty_like(x)
        yuv2rgb(x, adv_image)
        return adv_image


if __name__ == '__main__':
#   device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
  enc_img = torch.randn(2,3,128,128).to(device)
  cnn_dct = DCT_CNN().to(device)
  adv_img = cnn_dct(enc_img)
  print("adv_img:",adv_img.shape)




