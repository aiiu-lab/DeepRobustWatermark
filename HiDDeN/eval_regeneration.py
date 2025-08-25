import os
import csv
import gc
import numpy as np
import torch
from absl import flags, app
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
import utils
from losses import loss_map
from model.watermark import Watermark
from tqdm import tqdm
from accelerate import Accelerator

import torchvision.transforms.functional as function
import torch.nn.functional as F
import random
from PIL import Image, ImageFilter, ImageEnhance, ImageOps
import torchvision.transforms as T
import numpy as np
from tqdm import tqdm
import PIL
from torchvision import transforms
import io
from Respipeline import ReSDPipeline

from compressai.zoo import (
    bmshj2018_factorized,
    bmshj2018_hyperprior,
    mbt2018_mean,
    mbt2018,
    cheng2020_anchor,
)

FLAGS = flags.FLAGS

### Embedding Configuration ###
flags.DEFINE_integer('image_size', 128, "size of the images generated")
flags.DEFINE_integer('batch_size', 32, "size of batch")
flags.DEFINE_string('dataset', '/home/Documents/coco', 'Dataset used')
flags.DEFINE_integer('redundant_length', 120, "length of redundant message")
flags.DEFINE_integer('message_length', 30, "length of message")
flags.DEFINE_string('load_checkpoint', None, 'load from checkpoint')
flags.DEFINE_string('load_eval_waves', None, 'load from checkpoint')

### Training Configuration ###
flags.DEFINE_integer('nc', 3, "Channel Dimension of image")
flags.DEFINE_integer('seed', 0, "random seed")
flags.DEFINE_float('lr', 1e-5, "learning rate of attack module")
flags.DEFINE_float('weight_decay', 1e-3, "weight decay for attack optimizer")

### NECST ###
flags.DEFINE_integer('necst_iter', 20000, "number of iterations necst trained")
### Decoder ###
flags.DEFINE_integer('decoder_channels', 64, "number of channels of decoder")
flags.DEFINE_integer('decoder_blocks', 7, "number of blocks of decoder")
### Encoder ###
flags.DEFINE_integer('encoder_channels', 64, "number of channels of encoder")
flags.DEFINE_integer('encoder_blocks', 4, "number of blocks of encoder")
### Discriminator ###
flags.DEFINE_integer('discriminator_channels', 64, "number of channels of discriminator")
flags.DEFINE_integer('discriminator_blocks', 3, "number of blocks of discriminator")
flags.DEFINE_enum('loss', 'ns', loss_map.keys(), "loss function")
### Generator ###
flags.DEFINE_integer('n_gen', 5, "number of generator train per iterations")
flags.DEFINE_string('adv_type', 'dct_transformer', 'choice: [none,hidden,cnn,transformer,dct_cnn,dct_transformer,parallel,cascade]')


class MyDataset(Dataset):
    def __init__(self):
        self.root = FLAGS.dataset
        self.data_list = os.listdir(self.root + "/test/test_class")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(size=(FLAGS.image_size, FLAGS.image_size)),
                transforms.ToTensor()
            ]),
            'test': transforms.Compose([
                transforms.Resize(size=(FLAGS.image_size, FLAGS.image_size)),
                transforms.ToTensor()
            ])
        }
        validation_images = datasets.ImageFolder(self.root + "/test", data_transforms['test'])
        img = validation_images[idx][0]
        image_filename = validation_images.imgs[idx][0].split('/')[-1]
        return img, image_filename

def eval_adversarial(accelerator):
    device = accelerator.device
    dataset = MyDataset()
    Data =  DataLoader(dataset, batch_size=FLAGS.batch_size)

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

    model = Watermark(FLAGS, accelerator)
    model.setup_model()
    model.setup_optimizer(lr=FLAGS.lr,weight_decay=FLAGS.weight_decay)

    model, data_ld = accelerator.prepare(
        model, Data
    )

    ## Loading checkpoint ##
    if FLAGS.load_eval_waves:
        print("Loading checkpoints")
        checkpoint = torch.load(FLAGS.load_eval_waves, map_location=device)
        model.load_eval_waves(checkpoint)
        del checkpoint
    else:
        assert FLAGS.load_eval_waves == True

    ### eval regeneration ###
    identity_acc = 0
    attack_regen_diff_acc = 0
    attack_regen_vae_acc = 0
    attack_regen_diff2x_acc = 0
    attack_regen_diff4x_acc = 0
    regeneration_acc_avg = 0

    model.eval()
    for  (img_data, filename) in tqdm(data_ld):
        img_data = img_data.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (img_data.shape[0], FLAGS.message_length))).to(device)
        encoded_image, _ = model.encode(img_data, message)
        ### identity ###
        decoded_messages = model.decode(encoded_image)
        if model.net_Necst is not None:
            decoded_messages = model.net_Necst.decode(decoded_messages)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        identity_acc += ( 1 - bitwise_avg_err)
        ### attack_regen_diff ###
        adv_images_list = []
        for i in range(len(encoded_image)):
            attack_regen_diff = regen_diff(image=T.ToPILImage()(encoded_image[i]), strength=5, device=device)
            adv_images_list.append(T.ToTensor()(attack_regen_diff).unsqueeze(0))
        adv_images = torch.cat(adv_images_list, dim=0).to(device)
        regeneration_predicted = model.decode(adv_images)
        if model.net_Necst is not None:
            regeneration_predicted = model.net_Necst.decode(regeneration_predicted)
        regeneration_rounded = regeneration_predicted.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(regeneration_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        attack_regen_diff_acc += (1 - bitwise_avg_err)
        ### attack_regen_vae ###
        adv_images_list = []
        for i in range(len(encoded_image)):
            attack_regen_vae = regen_vae(image=T.ToPILImage()(encoded_image[i]), strength=5, device=device)
            adv_images_list.append(T.ToTensor()(attack_regen_vae).unsqueeze(0))
        adv_images = torch.cat(adv_images_list, dim=0).to(device)
        regeneration_predicted = model.decode(adv_images)
        if model.net_Necst is not None:
            regeneration_predicted = model.net_Necst.decode(regeneration_predicted)
        regeneration_rounded = regeneration_predicted.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(regeneration_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        attack_regen_vae_acc += (1 - bitwise_avg_err)
        ### attack_regen_diff2x ###
        adv_images_list = []
        for i in range(len(encoded_image)):
            attack_regen_diff2x = rinse_2xDiff(image=T.ToPILImage()(encoded_image[i]), strength=5, device=device)
            adv_images_list.append(T.ToTensor()(attack_regen_diff2x).unsqueeze(0))
        adv_images = torch.cat(adv_images_list, dim=0).to(device)
        regeneration_predicted = model.decode(adv_images)
        if model.net_Necst is not None:
            regeneration_predicted = model.net_Necst.decode(regeneration_predicted)
        regeneration_rounded = regeneration_predicted.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(regeneration_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        attack_regen_diff2x_acc += (1 - bitwise_avg_err)
        ### attack_regen_diff4x ###
        adv_images_list = []
        for i in range(len(encoded_image)):
            attack_regen_diff4x = rinse_4xDiff(image=T.ToPILImage()(encoded_image[i]), strength=5, device=device)
            adv_images_list.append(T.ToTensor()(attack_regen_diff4x).unsqueeze(0))
        adv_images = torch.cat(adv_images_list, dim=0).to(device)
        regeneration_predicted = model.decode(adv_images)
        if model.net_Necst is not None:
            regeneration_predicted = model.net_Necst.decode(regeneration_predicted)
        regeneration_rounded = regeneration_predicted.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(regeneration_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        attack_regen_diff4x_acc += (1 - bitwise_avg_err)

        print("identity_acc", identity_acc)
        print("attack_regen_diff_acc", attack_regen_diff_acc)
        print("attack_regen_vae_acc", attack_regen_vae_acc)
        print("attack_regen_diff2x_acc", attack_regen_diff2x_acc)
        print("attack_regen_diff4x_acc", attack_regen_diff4x_acc)
        breakpoint()

    regeneration_acc_avg = attack_regen_diff_acc + attack_regen_vae_acc + attack_regen_diff2x_acc + attack_regen_diff4x_acc
    # regeneration_acc_avg = attack_regen_diff_acc + attack_regen_vae_acc
    # regeneration_acc_avg /= 2

    ### print results ###
    print("identity_acc", identity_acc / len(data_ld))
    print("attack_regen_diff_acc", attack_regen_diff_acc / len(data_ld))
    print("attack_regen_vae_acc", attack_regen_vae_acc / len(data_ld))
    print("attack_regen_diff2x_acc", attack_regen_diff2x_acc / len(data_ld))
    print("attack_regen_diff4x_acc", attack_regen_diff4x_acc / len(data_ld))
    # print("regeneration_acc_avg", regeneration_acc_avg / len(data_ld))

def main(argv):
    utils.set_seed(FLAGS.seed)
    accelerator = Accelerator()
    eval_adversarial(accelerator=accelerator)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass  