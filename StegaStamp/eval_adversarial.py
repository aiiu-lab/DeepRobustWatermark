import argparse

import torchvision.transforms.functional as function
import torch.nn.functional as F
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torchvision.transforms as T
import numpy as np
import torch
import torch.utils.data as data
import os
import torchvision.transforms as transforms

from accelerate import Accelerator

parser = argparse.ArgumentParser()
parser.add_argument(
    "--data_dir", type=str, required=True, help="Directory with image dataset."
)
parser.add_argument(
    "--encoder_path", type=str, help="Path to trained StegaStamp encoder."
)
parser.add_argument(
    "--decoder_path", type=str, help="Path to trained StegaStamp decoder."
)
parser.add_argument(
    "--dec_enc_optim", type=str, help="Path to trained StegaStamp encoder."
)
parser.add_argument(
    "--use_celeba_preprocessing",
    action="store_true",
    help="Use CelebA specific preprocessing when loading the images.",
)
parser.add_argument(
    "--fingerprint_length",
    type=int,
    default=100,
    required=True,
    help="Number of bits in the fingerprint.",
)
parser.add_argument(
    "--image_resolution",
    type=int,
    default=128,
    required=True,
    help="Height and width of square images.",
)
parser.add_argument("--batch_size", type=int, default=64, help="Batch size.")
parser.add_argument("--lr", type=float, default=0.0001, help="Learning rate.")
parser.add_argument("--cuda", type=str, default=0)

parser.add_argument(
    "--l2_loss_await",
    help="Train without L2 loss for the first x iterations",
    type=int,
    default=1000,
)
parser.add_argument(
    "--l2_loss_weight",
    type=float,
    default=10,
    help="L2 loss weight for image fidelity.",
)
parser.add_argument(
    "--l2_loss_ramp",
    type=int,
    default=3000,
    help="Linearly increase L2 loss weight over x iterations.",
)
parser.add_argument(
    "--BCE_loss_weight",
    type=float,
    default=1,
    help="BCE loss weight for fingerprint reconstruction.",
)

args = parser.parse_args()

import glob
import os
from os.path import join
from time import time

from datetime import datetime

from tqdm import tqdm
import PIL

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image

from torch.optim import Adam

import models
from feature_extractors import (
    ResNet18Embedding,
    VAEEmbedding,
    ClipEmbedding,
    KLVAEEmbedding,
)

### WAVES adversarial ###
EPS_FACTOR = 1 / 255
ALPHA_FACTOR = 0.05
N_STEPS = 200
    
class WarmupPGDEmbedding:
    def __init__(
        self,
        model,
        device,
        eps=8 / 255,
        alpha=2 / 255,
        steps=10,
        loss_type="l2",
        random_start=True,
    ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.loss_type = loss_type
        self.random_start = random_start
        self.device = device

        # Initialize the loss function
        if self.loss_type == "l1":
            self.loss_fn = torch.nn.L1Loss()
        elif self.loss_type == "l2":
            self.loss_fn = torch.nn.MSELoss()
        else:
            raise ValueError("Unsupported loss type")

    def forward(self, images, init_delta=None):
        self.model.eval()
        images = images.clone().detach().to(self.device)

        # Get the original embeddings
        original_embeddings = self.model(images).detach()

        # initialize adv images
        if self.random_start:
            adv_images = images.clone().detach()
            # Starting at a uniformly random point
            adv_images = adv_images + torch.empty_like(adv_images).uniform_(
                -self.eps, self.eps
            )
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        elif init_delta is not None:
            clamped_delta = torch.clamp(init_delta, min=-self.eps, max=self.eps)
            adv_images = images.clone().detach() + clamped_delta
            adv_images = torch.clamp(adv_images, min=0, max=1).detach()
        else:
            assert False

        # PGD
        for _ in range(self.steps):
            self.model.zero_grad()
            adv_images.requires_grad = True
            adv_embeddings = self.model(adv_images)

            # Calculate loss
            cost = self.loss_fn(adv_embeddings, original_embeddings)

            # Update adversarial images
            grad = torch.autograd.grad(
                cost, adv_images, retain_graph=False, create_graph=False
            )[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
################################################################################

def generate_random_fingerprints(fingerprint_length, batch_size=4, size=(400, 400)):
    z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)
    return z


plot_points = (
    list(range(0, 1000, 100))
    + list(range(1000, 3000, 200))
    + list(range(3000, 100000, 1000))
)


class CustomImageFolder(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.filenames = glob.glob(os.path.join(data_dir, "*.png"))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpeg")))
        self.filenames.extend(glob.glob(os.path.join(data_dir, "*.jpg")))
        self.filenames = sorted(self.filenames)
        self.transform = transform

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        image = PIL.Image.open(filename)
        if self.transform:
            image = self.transform(image)
            if image.shape[0] != 3:
                image = image.expand(3, 128, 128)
            # print(image.shape)
        return image, 0

    def __len__(self):
        return len(self.filenames)


def load_data():
    global dataset, dataloader
    global IMAGE_CHANNELS, IMAGE_HEIGHT, IMAGE_WIDTH, SECRET_SIZE

    IMAGE_RESOLUTION = args.image_resolution
    IMAGE_CHANNELS = 3

    SECRET_SIZE = args.fingerprint_length

    if args.use_celeba_preprocessing:
        assert args.image_resolution == 128, f"CelebA preprocessing requires image resolution 128, got {args.image_resolution}."
        transform = transforms.Compose(
            [
                transforms.CenterCrop(148),
                transforms.Resize(128),
                transforms.ToTensor(),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(IMAGE_RESOLUTION),
                transforms.CenterCrop(IMAGE_RESOLUTION),
                transforms.ToTensor(),
            ]
        )

    s = time()
    print(f"Loading image folder {args.data_dir} ...")
    dataset = CustomImageFolder(args.data_dir, transform=transform)
    print(f"Finished. Loading took {time() - s:.2f}s")


def main():
    now = datetime.now()
    dt_string = now.strftime("%d%m%Y_%H:%M:%S")
    EXP_NAME = f"stegastamp_{args.fingerprint_length}_{dt_string}"

    accelerator = Accelerator()
    device = accelerator.device
    
    load_data()
    ### create models ###
    encoder = models.StegaStampEncoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.fingerprint_length,
        return_residual=False,
    )
    decoder = models.StegaStampDecoder(
        args.image_resolution,
        IMAGE_CHANNELS,
        args.fingerprint_length,
    )

    encoder = encoder.to(device)
    decoder = decoder.to(device)
    ### load models ###
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {"map_location": device}
    encoder.load_state_dict(torch.load(args.encoder_path, **kwargs))
    decoder.load_state_dict(torch.load(args.decoder_path, **kwargs))

    encoder.eval()
    decoder.eval()

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
    )

    data_ld = accelerator.prepare(dataloader)

    ### eval adversarial attack ###
    identity_acc = 0
    resnet18_acc = 0
    clip_acc = 0
    klvae8_acc = 0
    sdxlvae_acc = 0
    klvae16_acc = 0
    adv_acc_avg = 0

    ### create an instance of the attack ###
    attack_res = WarmupPGDEmbedding(
        model=ResNet18Embedding("last").to(device).eval(),
        eps=EPS_FACTOR * 8,
        alpha=ALPHA_FACTOR * EPS_FACTOR * 8,
        steps=N_STEPS,
        device=device,
    )
    attack_clip = WarmupPGDEmbedding(
        model=ClipEmbedding().to(device).eval(),
        eps=EPS_FACTOR * 8,
        alpha=ALPHA_FACTOR * EPS_FACTOR * 8,
        steps=N_STEPS,
        device=device,
    )
    attack_vae_mse = WarmupPGDEmbedding(
        model=VAEEmbedding("stabilityai/sd-vae-ft-mse").to(device).eval(),
        eps=EPS_FACTOR * 8,
        alpha=ALPHA_FACTOR * EPS_FACTOR * 8,
        steps=N_STEPS,
        device=device,
    )
    attack_vae = WarmupPGDEmbedding(
        model=VAEEmbedding("stabilityai/sdxl-vae").to(device).eval(),
        eps=EPS_FACTOR * 8,
        alpha=ALPHA_FACTOR * EPS_FACTOR * 8,
        steps=N_STEPS,
        device=device,
    )
    attack_kl = WarmupPGDEmbedding(
        model=KLVAEEmbedding("kl-f16").to(device).eval(),
        eps=EPS_FACTOR * 8,
        alpha=ALPHA_FACTOR * EPS_FACTOR * 8,
        steps=N_STEPS,
        device=device,
    )

    for images, _ in tqdm(data_ld):

        batch_size = min(args.batch_size, images.size(0))
        fingerprints = generate_random_fingerprints(
            args.fingerprint_length, batch_size, (args.image_resolution, args.image_resolution)
        )
        clean_images = images.to(device)
        fingerprints = fingerprints.to(device)
        fingerprinted_images = encoder(fingerprints, clean_images)
        decoder_output = decoder(fingerprinted_images)

        ### identity ###
        fingerprints_predicted = (decoder_output > 0).float()
        identity_acc += (1.0 - torch.mean(
            torch.abs(fingerprints - fingerprints_predicted)
        )).item()
        ### resnet18 ###
        resnet18_image = attack_res.forward(fingerprinted_images)
        adversarial_predicted = (decoder(resnet18_image) > 0).float()
        resnet18_acc += (1.0 - torch.mean(
            torch.abs(fingerprints - adversarial_predicted)
        )).item()
        ### clip ###
        clip_image = attack_clip.forward(fingerprinted_images)
        adversarial_predicted = (decoder(clip_image) > 0).float()
        clip_acc += (1.0 - torch.mean(
            torch.abs(fingerprints - adversarial_predicted)
        )).item()
        ### klvae8 ###
        klvae8_image = attack_vae_mse.forward(fingerprinted_images)
        adversarial_predicted = (decoder(klvae8_image) > 0).float()
        klvae8_acc += (1.0 - torch.mean(
            torch.abs(fingerprints - adversarial_predicted)
        )).item()
        ### sdxlvae ###
        sdxlvae_image = attack_vae.forward(fingerprinted_images)
        adversarial_predicted = (decoder(sdxlvae_image) > 0).float()
        sdxlvae_acc += (1.0 - torch.mean(
            torch.abs(fingerprints - adversarial_predicted)
        )).item()
        ### klvae16 ###
        klvae16_image = attack_kl.forward(fingerprinted_images)
        adversarial_predicted = (decoder(klvae16_image) > 0).float()
        klvae16_acc += (1.0 - torch.mean(
            torch.abs(fingerprints - adversarial_predicted)
        )).item()

    adv_acc_avg = resnet18_acc + clip_acc + klvae8_acc + sdxlvae_acc + klvae16_acc
    adv_acc_avg /= 5

    ### print results ###
    print("identity_acc", identity_acc / len(data_ld))
    print("resnet18", resnet18_acc / len(data_ld))
    print("clip", clip_acc / len(data_ld))
    print("klvae8", klvae8_acc / len(data_ld))
    print("sdxlvae", sdxlvae_acc / len(data_ld))
    print("klvae16", klvae16_acc / len(data_ld))
    print("adv_avg", adv_acc_avg / len(data_ld))


if __name__ == "__main__":
    main()
