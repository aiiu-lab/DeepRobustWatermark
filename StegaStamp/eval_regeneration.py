import argparse

import torchvision.transforms.functional as function
import torch.nn.functional as F
import random
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torchvision.transforms as T
import numpy as np
import torch
import io
import torch.utils.data as data
import os
import csv
from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    mbt2018_mean,
    mbt2018,
    cheng2020_anchor,
)

from accelerate import Accelerator

accelerator = Accelerator()
device = accelerator.device

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
# from tensorboardX import SummaryWriter

from torch.optim import Adam

import models
from diffusers import ReSDPipeline


### WAVES regeneration ###
class WMAttacker:
    def attack(self, imgs_path, out_path):
        raise NotImplementedError

def regen_diff(
    image, strength, model="CompVis/stable-diffusion-v1-4", device=device
):
    image = remove_watermark("regen_diffusion", image, strength, model, device)
    return image

def rinse_2xDiff(image, strength, model="CompVis/stable-diffusion-v1-4", device=device):
    first_attack = True
    for attack in ["regen_diffusion", "regen_diffusion"]:
        if first_attack:
            image = remove_watermark(attack, image, strength, model, device)
            first_attack = False
        else:
            image = remove_watermark(attack, image, strength, model, device)
    return image

def regen_vae(
    image, strength, model="bmshj2018-factorized", device=device
):
    image = remove_watermark("regen_vae", image, strength, model, device)
    return image

def rinse_4xDiff(image, strength, model="CompVis/stable-diffusion-v1-4", device=device):
    first_attack = True
    for attack in [
        "regen_diffusion",
        "regen_diffusion",
        "regen_diffusion",
        "regen_diffusion",
    ]:
        if first_attack:
            image = remove_watermark(attack, image, strength, model, device)
            first_attack = False
        else:
            image = remove_watermark(attack, image, strength, model, device)
    return image

def remove_watermark(attack_method, image, strength, model, device):
    # create attacker
    print(f"Creating attacker {attack_method}...")
    if attack_method == "regen_vae":
        attacker = VAEWMAttacker(model, strength=strength, metric="mse", device=device)
    elif attack_method == "regen_diffusion":
        pipe = ReSDPipeline.from_pretrained(
            model, torch_dtype=torch.float16, revision="fp16"
        )

        pipe.set_progress_bar_config(disable=True)
        pipe.to(device)
        attacker = DiffWMAttacker(pipe, noise_step=strength, captions={})
    else:
        raise Exception(f"Unknown attacking method: {attack_method}!")

    img = attacker.attack(image, device)

    return img

class VAEWMAttacker(WMAttacker):
    def __init__(self, model_name, strength=1, metric="mse", device=device):
        if model_name == "bmshj2018-factorized":
            self.model = (
                bmshj2018_factorized(quality=strength, pretrained=True)
                .eval()
                .to(device)
            )
        elif model_name == "bmshj2018-hyperprior":
            self.model = (
                bmshj2018_hyperprior(strength=strength, pretrained=True)
                .eval()
                .to(device)
            )
        elif model_name == "mbt2018-mean":
            self.model = (
                mbt2018_mean(strength=strength, pretrained=True).eval().to(device)
            )
        elif model_name == "mbt2018":
            self.model = mbt2018(strength=strength, pretrained=True).eval().to(device)
        elif model_name == "cheng2020-anchor":
            self.model = (
                cheng2020_anchor(strength=strength, pretrained=True).eval().to(device)
            )
        else:
            raise ValueError("model name not supported")
        self.device = device

    def attack(self, image, device):

        img = image.convert("RGB")
        img = img.resize((512, 512))
        img = transforms.ToTensor()(img).unsqueeze(0).to(self.device)
        out = self.model(img)
        out["x_hat"].clamp_(0, 1)
        out = transforms.ToPILImage()(out["x_hat"].squeeze().cpu())
        return out


class DiffWMAttacker(WMAttacker):
    def __init__(self, pipe, noise_step=60, captions={}):
        self.pipe = pipe
        self.device = pipe.device
        self.noise_step = noise_step
        self.captions = captions
        print(
            f"Diffuse attack initialized with noise step {self.noise_step} and use prompt {len(self.captions)}"
        )

    def attack(self, image, device, return_latents=False, return_dist=False):
        with torch.no_grad():
            generator = torch.Generator(device).manual_seed(1024)
            latents_buf = []
            prompts_buf = []
            outs_buf = []
            timestep = torch.tensor(
                [self.noise_step], dtype=torch.long, device=self.device
            )
            ret_latents = []

            def batched_attack(latents_buf, prompts_buf, outs_buf):
                latents = torch.cat(latents_buf, dim=0)
                images = self.pipe(
                    prompts_buf,
                    head_start_latents=latents,
                    head_start_step=50 - max(self.noise_step // 20, 1),
                    guidance_scale=7.5,
                    generator=generator,
                )
                images = images[0]
                for img, out in zip(images, outs_buf):
                    return img

            img = np.asarray(image) / 255
            img = (img - 0.5) * 2
            img = torch.tensor(img).permute(2, 0, 1).unsqueeze(0)
            latents = self.pipe.vae.encode(
                img.to(device=device, dtype=torch.float16)
            ).latent_dist
            latents = latents.sample(generator) * self.pipe.vae.config.scaling_factor
            noise = torch.randn(
                [1, 4, img.shape[-2] // 8, img.shape[-1] // 8], device=self.device
            )

            latents = self.pipe.scheduler.add_noise(latents, noise, timestep).type(
                torch.half
            )
            latents_buf.append(latents)
            outs_buf.append("")
            prompts_buf.append("")

            img = batched_attack(latents_buf, prompts_buf, outs_buf)
            return img

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

    ### eval regeneration ###
    identity_acc = 0
    attack_regen_diff_acc = 0
    attack_regen_vae_acc = 0
    attack_regen_diff2x_acc = 0
    attack_regen_diff4x_acc = 0
    regeneration_acc_avg = 0
    
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

        ### attack_regen_diff ###
        adv_images_list = []
        for i in range(len(fingerprinted_images)):
            attack_regen_diff = regen_diff(image=T.ToPILImage()(fingerprinted_images[i]), strength=5, device=device)
            adv_images_list.append(T.ToTensor()(attack_regen_diff).unsqueeze(0))
        adv_images = torch.cat(adv_images_list, dim=0).to(device)
        regeneration_predicted = (decoder(adv_images) > 0).float()
        attack_regen_diff_acc += (1.0 - torch.mean(
            torch.abs(fingerprints - regeneration_predicted)
        )).item()

        ### attack_regen_vae ###
        h, w = fingerprinted_images.shape[2:4]
        adv_images_list = []
        for i in range(len(fingerprinted_images)):
            attack_regen_vae = regen_vae(image=T.ToPILImage()(fingerprinted_images[i]), strength=1, device=device).resize((h, w)) ### strength=(1,8)
            adv_images_list.append(T.ToTensor()(attack_regen_vae).unsqueeze(0))
        adv_images = torch.cat(adv_images_list, dim=0).to(device)
        regeneration_predicted = (decoder(adv_images) > 0).float()
        attack_regen_vae_acc += (1.0 - torch.mean(
            torch.abs(fingerprints - regeneration_predicted)
        )).item()

        ### attack_regen_diff2x ###
        # adv_images_list = []
        # for i in range(len(fingerprinted_images)):
        #     attack_regen_diff2x = rinse_2xDiff(image=T.ToPILImage()(fingerprinted_images[i]), strength=5, device=device)
        #     adv_images_list.append(T.ToTensor()(attack_regen_diff2x).unsqueeze(0))
        # adv_images = torch.cat(adv_images_list, dim=0).to(device)
        # regeneration_predicted = (decoder(adv_images) > 0).float()
        # attack_regen_diff2x_acc += (1.0 - torch.mean(
        #     torch.abs(fingerprints - regeneration_predicted)
        # )).item()
        
        ### attack_regen_diff4x ###
        # adv_images_list = []
        # for i in range(len(fingerprinted_images)):
        #     attack_regen_diff4x = rinse_4xDiff(image=T.ToPILImage()(fingerprinted_images[i]), strength=5, device=device)
        #     adv_images_list.append(T.ToTensor()(attack_regen_diff4x).unsqueeze(0))
        # adv_images = torch.cat(adv_images_list, dim=0).to(device)
        # regeneration_predicted = (decoder(adv_images) > 0).float()
        # attack_regen_diff4x_acc += (1.0 - torch.mean(
        #     torch.abs(fingerprints - regeneration_predicted)
        # )).item()

    # regeneration_acc_avg = identity_acc + attack_regen_diff_acc + attack_regen_vae_acc + attack_regen_diff2x_acc + attack_regen_diff4x_acc
    regeneration_acc_avg = identity_acc + attack_regen_diff2x_acc + attack_regen_diff4x_acc
    # regeneration_acc_avg /= 3

    ### print results ###
    print("identity_acc", identity_acc / len(data_ld))
    print("attack_regen_diff_acc", attack_regen_diff_acc / len(data_ld))
    print("attack_regen_vae_acc", attack_regen_vae_acc / len(data_ld))
    # print("attack_regen_diff2x_acc", attack_regen_diff2x_acc / len(data_ld))
    # print("attack_regen_diff4x_acc", attack_regen_diff4x_acc / len(data_ld))
    # print("regeneration_acc_avg", regeneration_acc_avg / len(data_ld))


if __name__ == "__main__":
    main()
