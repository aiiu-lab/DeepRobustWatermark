import math
import torch
import random
import torchvision.transforms as transforms
from torch import nn
import torch.nn.init as init
from torch.nn.utils.spectral_norm import spectral_norm
from torch.nn.functional import relu, sigmoid

class Blur(nn.Module):
    def __init__(self, ratio=random.randint(1, 3)):
        super(Blur, self).__init__()
        # import pdb;pdb.set_trace()
        self.blur = transforms.GaussianBlur(random.randrange(3, 16, 2),ratio)

    def forward(self, noised_and_cover):
        noised_and_cover[0] = self.blur(noised_and_cover[0])
        return noised_and_cover

class ConvBNRelu(nn.Module):
    """
    Building block used in HiDDeN network. Is a sequence of Convolution, Batch Normalization, and ReLU activation
    """

    def __init__(self, channels_in, channels_out, stride=1,padding=1):
        super(ConvBNRelu, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, 3, stride, padding=padding),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.layers(x)

class Discriminator(nn.Module):
    """
    Discriminator network. Receives an image and has to figure out whether it has a watermark inserted into it, or not.
    """
    def __init__(self):
        super(Discriminator, self).__init__()

        layers = [ConvBNRelu(3, 64)]
        for _ in range(2):
            layers.append(ConvBNRelu(64, 64))

        layers.append(nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        self.before_linear = nn.Sequential(*layers)
        self.linear = nn.Linear(64, 1)
        self.initialize()

    def initialize(self):
        for m in self.before_linear.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight, math.sqrt(2))
                init.zeros_(m.bias)
                spectral_norm(m)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                spectral_norm(m)

    def forward(self, image):
        X = self.before_linear(image)
        X.squeeze_(3).squeeze_(2)
        X = self.linear(X)
        return X

class Generator(nn.Module):
    """
    Inserts a watermark into an image.
    """
    def __init__(self):
        super(Generator, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(16, 3, 3, stride=1, padding=1),
        )

    def forward(self, enc_image):
        return self.conv_layers(enc_image)

class StegaStampEncoder(nn.Module):
    def __init__(
        self,
        resolution=32,
        IMAGE_CHANNELS=1,
        fingerprint_size=100,
        return_residual=False,
    ):
        super(StegaStampEncoder, self).__init__()
        self.fingerprint_size = fingerprint_size
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.return_residual = return_residual
        self.secret_dense = nn.Linear(self.fingerprint_size, 16 * 16 * IMAGE_CHANNELS)

        log_resolution = int(math.log(resolution, 2))
        assert resolution == 2 ** log_resolution, f"Image resolution must be a power of 2, got {resolution}."

        self.fingerprint_upsample = nn.Upsample(scale_factor=(2**(log_resolution-4), 2**(log_resolution-4)))
        self.conv1 = nn.Conv2d(2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv2 = nn.Conv2d(32, 32, 3, 2, 1)
        self.conv3 = nn.Conv2d(32, 64, 3, 2, 1)
        self.conv4 = nn.Conv2d(64, 128, 3, 2, 1)
        self.conv5 = nn.Conv2d(128, 256, 3, 2, 1)
        self.pad6 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up6 = nn.Conv2d(256, 128, 2, 1)
        self.upsample6 = nn.Upsample(scale_factor=(2, 2))
        self.conv6 = nn.Conv2d(128 + 128, 128, 3, 1, 1)
        self.pad7 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up7 = nn.Conv2d(128, 64, 2, 1)
        self.upsample7 = nn.Upsample(scale_factor=(2, 2))
        self.conv7 = nn.Conv2d(64 + 64, 64, 3, 1, 1)
        self.pad8 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up8 = nn.Conv2d(64, 32, 2, 1)
        self.upsample8 = nn.Upsample(scale_factor=(2, 2))
        self.conv8 = nn.Conv2d(32 + 32, 32, 3, 1, 1)
        self.pad9 = nn.ZeroPad2d((0, 1, 0, 1))
        self.up9 = nn.Conv2d(32, 32, 2, 1)
        self.upsample9 = nn.Upsample(scale_factor=(2, 2))
        self.conv9 = nn.Conv2d(32 + 32 + 2 * IMAGE_CHANNELS, 32, 3, 1, 1)
        self.conv10 = nn.Conv2d(32, 32, 3, 1, 1)
        self.residual = nn.Conv2d(32, IMAGE_CHANNELS, 1)

    def forward(self, fingerprint, image):
        # import pdb;pdb.set_trace()
        fingerprint = relu(self.secret_dense(fingerprint))
        fingerprint = fingerprint.view((-1, self.IMAGE_CHANNELS, 16, 16))
        fingerprint_enlarged = self.fingerprint_upsample(fingerprint)
        inputs = torch.cat([fingerprint_enlarged, image], dim=1)
        conv1 = relu(self.conv1(inputs))
        conv2 = relu(self.conv2(conv1))
        conv3 = relu(self.conv3(conv2))
        conv4 = relu(self.conv4(conv3))
        conv5 = relu(self.conv5(conv4))
        up6 = relu(self.up6(self.pad6(self.upsample6(conv5))))
        merge6 = torch.cat([conv4, up6], dim=1)
        conv6 = relu(self.conv6(merge6))
        up7 = relu(self.up7(self.pad7(self.upsample7(conv6))))
        merge7 = torch.cat([conv3, up7], dim=1)
        conv7 = relu(self.conv7(merge7))
        up8 = relu(self.up8(self.pad8(self.upsample8(conv7))))
        merge8 = torch.cat([conv2, up8], dim=1)
        conv8 = relu(self.conv8(merge8))
        up9 = relu(self.up9(self.pad9(self.upsample9(conv8))))
        merge9 = torch.cat([conv1, up9, inputs], dim=1)
        conv9 = relu(self.conv9(merge9))
        conv10 = relu(self.conv10(conv9))
        residual = self.residual(conv10)
        if not self.return_residual:
            residual = sigmoid(residual)
        return residual


class StegaStampDecoder(nn.Module):
    def __init__(self, resolution=32, IMAGE_CHANNELS=1, fingerprint_size=1):
        super(StegaStampDecoder, self).__init__()
        self.resolution = resolution
        self.IMAGE_CHANNELS = IMAGE_CHANNELS
        self.decoder = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNELS, 32, (3, 3), 2, 1),  # 16
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 8
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 2, 1),  # 4
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),  # 2
            nn.ReLU(),
            nn.Conv2d(128, 128, (3, 3), 2, 1),
            nn.ReLU(),
        )
        self.dense = nn.Sequential(
            nn.Linear(resolution * resolution * 128 // 32 // 32, 512),
            nn.ReLU(),
            nn.Linear(512, fingerprint_size),
        )

    def forward(self, image):
        # import pdb;pdb.set_trace()
        x = self.decoder(image)
        x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        # try:
        #     x = x.view(-1, self.resolution * self.resolution * 128 // 32 // 32)
        # except:
        #     x = x.reshape(-1, self.resolution * self.resolution * 128 // 32 // 32)
        return self.dense(x)

