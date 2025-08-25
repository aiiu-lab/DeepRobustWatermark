import argparse

import torchvision.transforms.functional as function
import torch.nn.functional as F
from parallel import Parallel
import random
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torchvision.transforms as T
import numpy as np
import torch
import io
import torch.utils.data as data
import os
import csv

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
    "--attack_path", type=str, help="Path to trained attack network."
)
parser.add_argument(
    "--use_celeba_preprocessing",
    action="store_true",
    help="Use CelebA specific preprocessing when loading the images.",
)
parser.add_argument(
    "--output_dir", type=str, required=True, help="Directory to save results to."
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
parser.add_argument(
    "--num_epochs", type=int, default=20, help="Number of training epochs."
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
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

CHECKPOINTS_PATH = os.path.join(args.output_dir, "checkpoints")
SAVED_IMAGES = os.path.join(args.output_dir, "./saved_images")

if not os.path.exists(CHECKPOINTS_PATH):
    os.makedirs(CHECKPOINTS_PATH)
if not os.path.exists(SAVED_IMAGES):
    os.makedirs(SAVED_IMAGES)

def set_random_seed(seed=0):
    torch.manual_seed(seed + 0)
    torch.cuda.manual_seed(seed + 1)
    torch.cuda.manual_seed_all(seed + 2)
    np.random.seed(seed + 3)
    torch.cuda.manual_seed_all(seed + 4)
    random.seed(seed + 5)

distortion_strength_paras = dict(
    rotation=(0, 45),
    resizedcrop=(1, 0.5),
    erasing=(0, 0.25),
    brightness=(1, 2),
    contrast=(1, 2),
    blurring=(0, 20),
    noise=(0, 0.1),
    compression=(90, 10),
)

def to_tensor(images, norm_type="naive"):
    assert isinstance(images, list) and all(
        [isinstance(image, Image.Image) for image in images]
    )
    images = torch.stack([transforms.ToTensor()(image) for image in images])
    if norm_type is not None:
        images = normalize_tensor(images, norm_type)
    return images

def to_pil(images, norm_type="naive"):
    assert isinstance(images, torch.Tensor)
    if norm_type is not None:
        images = unnormalize_tensor(images, norm_type).clamp(0, 1)
    return [transforms.ToPILImage()(image) for image in images.cpu()]

def apply_single_distortion(image, distortion_type, strength=None, distortion_seed=0):
    # Accept a single image
    assert isinstance(image, Image.Image)
    # Set the random seed for the distortion if given
    set_random_seed(distortion_seed)
    # Assert distortion type is valid
    assert distortion_type in distortion_strength_paras.keys()
    # Assert strength is in the correct range
    if strength is not None:
        assert (
            min(*distortion_strength_paras[distortion_type])
            <= strength
            <= max(*distortion_strength_paras[distortion_type])
        )

    # Apply the distortion
    if distortion_type == "rotation":
        angle = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["rotation"])
        )
        distorted_image = function.rotate(image, angle)

    elif distortion_type == "resizedcrop":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["resizedcrop"])
        )
        i, j, h, w = T.RandomResizedCrop.get_params(
            image, scale=(scale, scale), ratio=(1, 1)
        )
        distorted_image = function.resized_crop(image, i, j, h, w, image.size)

    elif distortion_type == "erasing":
        scale = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["erasing"])
        )
        image = to_tensor([image], norm_type=None)
        i, j, h, w, v = T.RandomErasing.get_params(
            image, scale=(scale, scale), ratio=(1, 1), value=[0]
        )
        distorted_image = function.erase(image, i, j, h, w, v)
        distorted_image = to_pil(distorted_image, norm_type=None)[0]

    elif distortion_type == "brightness":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["brightness"])
        )
        enhancer = ImageEnhance.Brightness(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "contrast":
        factor = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["contrast"])
        )
        enhancer = ImageEnhance.Contrast(image)
        distorted_image = enhancer.enhance(factor)

    elif distortion_type == "blurring":
        kernel_size = (
            int(strength)
            if strength is not None
            else random.uniform(*distortion_strength_paras["blurring"])
        )
        distorted_image = image.filter(ImageFilter.GaussianBlur(kernel_size))

    elif distortion_type == "noise":
        std = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["noise"])
        )
        image = to_tensor([image], norm_type=None)
        noise = torch.randn(image.size()) * std
        distorted_image = to_pil((image + noise).clamp(0, 1), norm_type=None)[0]

    elif distortion_type == "compression":
        quality = (
            strength
            if strength is not None
            else random.uniform(*distortion_strength_paras["compression"])
        )
        quality = int(quality)
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG", quality=quality)
        distorted_image = Image.open(buffered)

    else:
        assert False

    return distorted_image

def generate_random_fingerprints(fingerprint_length, batch_size=4, size=(400, 400)):
    z = torch.zeros((batch_size, fingerprint_length), dtype=torch.float).random_(0, 2)
    return z

def ns_loss(pred_fake, pred_real=None):
    bce = F.binary_cross_entropy_with_logits
    if pred_real is not None:
        loss_real = bce(pred_real, torch.ones_like(pred_real))
        loss_fake = bce(pred_fake, torch.zeros_like(pred_fake))
        return loss_fake, loss_real
    else:
        loss_fake = bce(pred_fake, torch.ones_like(pred_fake))
        return loss_fake

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
    device = torch.device("cuda")
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
    generator = Parallel().to(device)
    discriminator = models.Discriminator().to(device)

    ###  create optims ###
    decoder_encoder_optim = Adam(
        params=list(decoder.parameters()) + list(encoder.parameters()), lr=args.lr
    )
    gen_optim = Adam(generator.parameters(),lr=1e-4,weight_decay=1e-3)
    dis_optim = Adam(discriminator.parameters())

    ### load models ###
    kwargs = {"map_location": "cpu"} if args.cuda == -1 else {}
    encoder.load_state_dict(torch.load(args.encoder_path, **kwargs))
    decoder.load_state_dict(torch.load(args.decoder_path, **kwargs))
    if args.attack_path != None:
        attack_dis_checkpoint = torch.load(args.attack_path, **kwargs)
        generator.load_state_dict(attack_dis_checkpoint['attack-model'])
        discriminator.load_state_dict(attack_dis_checkpoint['discrim-model'])

    ### load optims ###
    dec_enc_optim = torch.load(args.dec_enc_optim, **kwargs)
    decoder_encoder_optim.load_state_dict(dec_enc_optim)
    if args.attack_path != None:
        gen_optim.load_state_dict(attack_dis_checkpoint['attack-optim'])
        dis_optim.load_state_dict(attack_dis_checkpoint['discrim-optim'])

    ### training ###
    global_step = 0
    steps_since_l2_loss_activated = -1

    for i_epoch in range(args.num_epochs):
        dataloader = DataLoader(
            dataset, batch_size=args.batch_size, shuffle=True, num_workers=16
        )
        for images, _ in tqdm(dataloader):

            ### criterion ###
            global_step += 1

            batch_size = min(args.batch_size, images.size(0))

            fingerprints = generate_random_fingerprints(
                args.fingerprint_length, batch_size, (args.image_resolution, args.image_resolution)
            )

            l2_loss_weight = min(
                max(
                    0,
                    args.l2_loss_weight
                    * (steps_since_l2_loss_activated - args.l2_loss_await)
                    / args.l2_loss_ramp,
                ),
                args.l2_loss_weight,
            )
            BCE_loss_weight = args.BCE_loss_weight
            l2_criterion = nn.MSELoss()
            bce_criterion = nn.BCEWithLogitsLoss()

            ### start training ###
            clean_images = images.to(device)
            fingerprints = fingerprints.to(device)
            fingerprinted_images = encoder(fingerprints, clean_images)

            ### train generator ###
            for _ in range(5):
                gen_optim.zero_grad()
                adv_images = generator(fingerprinted_images)
                adv_decoder_output = decoder(adv_images)
                l2_adv_fingerprint = l2_criterion(adv_images, fingerprinted_images)
                bce_adv_fingerprint = bce_criterion(adv_decoder_output.view(-1), fingerprints.view(-1))
                loss_g = 15.0 * l2_adv_fingerprint - 1.0 * bce_adv_fingerprint
                loss_g.backward(retain_graph=True)
                gen_optim.step()
                # save_image(adv_images ,'/scratch2/users/carl/ArtificialGANFingerprints/aaa.png' ,normalize=True)

            ### train discriminator ###
            dis_optim.zero_grad()
            pred_real = discriminator(clean_images)
            pred_fake = discriminator(fingerprinted_images.detach())
            loss_fake, loss_real = ns_loss(pred_fake, pred_real)
            loss_d = 0.5 * (loss_real + loss_fake)
            loss_d.backward(retain_graph=True)
            dis_optim.step()
            
            ### train encoder decoder ###
            encoder.zero_grad()
            decoder.zero_grad()
            adv_images = generator(fingerprinted_images)
            adv_decoder_output = decoder(adv_images)
            pred_fake = discriminator(fingerprinted_images)
            loss_fake =  ns_loss(pred_fake)
            l2_fingerprint_clean = l2_criterion(fingerprinted_images, clean_images)
            bce_adv_clean = bce_criterion(adv_decoder_output.view(-1), fingerprints.view(-1))
            decoder_output = decoder(fingerprinted_images)
            bce_fingerprint_clean = bce_criterion(decoder_output.view(-1), fingerprints.view(-1))
            loss_enc_dec = 0.15 * loss_fake + 10.0 * l2_fingerprint_clean + BCE_loss_weight * bce_fingerprint_clean + 0.7 * bce_adv_clean
            loss_enc_dec.backward()
            decoder_encoder_optim.step()
            
            ### results ###
            if global_step % 5000 == 0:
                ### bit accuracy ###
                distortion_acc_avg = 0
                ### identity ###
                fingerprints_predicted = (decoder_output > 0).float()
                identity_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - fingerprints_predicted)
                )
                distortion_acc_avg += identity_accuracy.item()
                ### rotation ###
                distortion_images_list = []
                for i in range(len(fingerprinted_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(fingerprinted_images[i]), "rotation")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(device)
                distortion_predicted = (decoder(distortion_images) > 0).float()
                rotation_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - distortion_predicted)
                )
                distortion_acc_avg += rotation_accuracy.item()
                ### resizedcrop ###
                distortion_images_list = []
                for i in range(len(fingerprinted_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(fingerprinted_images[i]), "resizedcrop")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(device)
                distortion_predicted = (decoder(distortion_images) > 0).float()
                resizedcrop_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - distortion_predicted)
                )
                distortion_acc_avg += resizedcrop_accuracy.item()
                ### erasing ###
                distortion_images_list = []
                for i in range(len(fingerprinted_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(fingerprinted_images[i]), "erasing")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(device)
                distortion_predicted = (decoder(distortion_images) > 0).float()
                erasing_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - distortion_predicted)
                )
                distortion_acc_avg += erasing_accuracy.item()
                ### brightness ###
                distortion_images_list = []
                for i in range(len(fingerprinted_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(fingerprinted_images[i]), "brightness")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(device)
                distortion_predicted = (decoder(distortion_images) > 0).float()
                brightness_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - distortion_predicted)
                )
                distortion_acc_avg += brightness_accuracy.item()
                ### contrast ###
                distortion_images_list = []
                for i in range(len(fingerprinted_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(fingerprinted_images[i]), "contrast")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(device)
                distortion_predicted = (decoder(distortion_images) > 0).float()
                contrast_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - distortion_predicted)
                )
                distortion_acc_avg += contrast_accuracy.item()
                ### blurring ###
                distortion_images_list = []
                for i in range(len(fingerprinted_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(fingerprinted_images[i]), "blurring")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(device)
                distortion_predicted = (decoder(distortion_images) > 0).float()
                blurring_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - distortion_predicted)
                )
                distortion_acc_avg += blurring_accuracy.item()
                ### noise ###
                distortion_images_list = []
                for i in range(len(fingerprinted_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(fingerprinted_images[i]), "noise")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(device)
                distortion_predicted = (decoder(distortion_images) > 0).float()
                noise_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - distortion_predicted)
                )
                distortion_acc_avg += noise_accuracy.item()
                ### compression ###
                distortion_images_list = []
                for i in range(len(fingerprinted_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(fingerprinted_images[i]), "compression")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(device)
                distortion_predicted = (decoder(distortion_images) > 0).float()
                compression_accuracy = 1.0 - torch.mean(
                    torch.abs(fingerprints - distortion_predicted)
                )
                distortion_acc_avg += compression_accuracy.item()
                distortion_acc_avg /= 9

                ### write csv ###
                csv_path = CHECKPOINTS_PATH + "/bit_accuracy.csv"
                csv_exist = os.path.exists(csv_path)
                with open(csv_path, mode='a' if csv_exist else 'w', newline='') as file:
                    csv_writer = csv.writer(file)
                    if not csv_exist:
                        csv_writer.writerow(["Steps", "Identity", "rotation", "resizedcrop", "erasing", "brightness", "contrast", "blurring", "noise", "compression", "avg"])
                    result_list = [identity_accuracy.item(), rotation_accuracy.item(), resizedcrop_accuracy.item(), 
                                    erasing_accuracy.item(), brightness_accuracy.item(), contrast_accuracy.item(), 
                                    blurring_accuracy.item(), noise_accuracy.item(), compression_accuracy.item(), distortion_acc_avg]
                    result = [round(x * 100, 3) for x in result_list]
                    result.insert(0, global_step)
                    csv_writer.writerow(result)   

                ### save image ###
                save_image(fingerprinted_images, SAVED_IMAGES + "/fingerprinted_images_{}.png".format(global_step), normalize=True)
                save_image(adv_images, SAVED_IMAGES + "/adv_images_{}.png".format(global_step), normalize=True)

                ### save checkpoints ###
                torch.save(
                    decoder_encoder_optim.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_optim_" + str(global_step) + ".pth"),
                )
                torch.save(
                    encoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_encoder_" + str(global_step) + ".pth"),
                )
                torch.save(
                    decoder.state_dict(),
                    join(CHECKPOINTS_PATH, EXP_NAME + "_decoder_" + str(global_step) + ".pth"),
                )

if __name__ == "__main__":
    main()
