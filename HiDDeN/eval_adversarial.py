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

from feature_extractors import (
    ResNet18Embedding,
    VAEEmbedding,
    ClipEmbedding,
    KLVAEEmbedding,
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

    model.eval()
    for  (img_data, filename) in tqdm(data_ld):
        img_data = img_data.to(device)
        message = torch.Tensor(np.random.choice([0, 1], (img_data.shape[0], FLAGS.message_length))).to(device)
        encoded_image, _ = model.encode(img_data, message)
        # torchvision.utils.save_image(encoded_image, '/scratch1/users/jason890425/aaa.png')
        ### identity ###
        decoded_messages = model.decode(encoded_image)
        if model.net_Necst is not None:
            decoded_messages = model.net_Necst.decode(decoded_messages)
        decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(decoded_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        identity_acc += (1 - bitwise_avg_err)
        ### resnet18 ###
        resnet18_image = attack_res.forward(encoded_image)
        adversarial_predicted = model.decode(resnet18_image)
        if model.net_Necst is not None:
            adversarial_predicted = model.net_Necst.decode(adversarial_predicted)
        adversarial_rounded = adversarial_predicted.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(adversarial_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        resnet18_acc += (1 - bitwise_avg_err)
        ### clip ###
        clip_image = attack_clip.forward(encoded_image)
        adversarial_predicted = model.decode(clip_image)
        if model.net_Necst is not None:
            adversarial_predicted = model.net_Necst.decode(adversarial_predicted)
        adversarial_rounded = adversarial_predicted.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(adversarial_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        clip_acc += (1 - bitwise_avg_err)
        ### klvae8 ###
        klvae8_image = attack_vae_mse.forward(encoded_image)
        adversarial_predicted = model.decode(klvae8_image)
        if model.net_Necst is not None:
            adversarial_predicted = model.net_Necst.decode(adversarial_predicted)
        adversarial_rounded = adversarial_predicted.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(adversarial_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        klvae8_acc += (1 - bitwise_avg_err)
        ### sdxlvae ###
        sdxlvae_image = attack_vae.forward(encoded_image)
        adversarial_predicted = model.decode(sdxlvae_image)
        if model.net_Necst is not None:
            adversarial_predicted = model.net_Necst.decode(adversarial_predicted)
        adversarial_rounded = adversarial_predicted.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(adversarial_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        sdxlvae_acc += (1 - bitwise_avg_err)
        ### klvae16 ###
        klvae16_image = attack_kl.forward(encoded_image)
        adversarial_predicted = model.decode(klvae16_image)
        if model.net_Necst is not None:
            adversarial_predicted = model.net_Necst.decode(adversarial_predicted)
        adversarial_rounded = adversarial_predicted.detach().cpu().numpy().round().clip(0, 1)
        bitwise_avg_err = (np.sum(np.abs(adversarial_rounded - message.detach().cpu().numpy())) / (message.shape[0] * message.shape[1])).item()
        klvae16_acc += (1 - bitwise_avg_err)

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

def main(argv):
    utils.set_seed(FLAGS.seed)
    accelerator = Accelerator()

    eval_adversarial(accelerator=accelerator)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass  