import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
from tqdm import tqdm
import torchvision
from model.discriminator import Discriminator
from model.encoder_decoder import EncoderDecoder
from model.necst import NECST
from noise_layers.noiser import Noiser
from noise_layers.identity import Identity
from noise_layers.jpeg_compression import JpegCompression
from noise_layers.dropout import Dropout
from noise_layers.cropout import Cropout
from noise_layers.posterize import Posterize
from noise_layers.solarize import Solarize
from noise_layers.quantization import Quantization 
from noise_layers.crop import Crop
from noise_layers.resize import Resize
from noise_layers.hue import Hue
from noise_layers.gaussian_noise import Gaussian_Noise
from noise_layers.sat import Sat
from noise_layers.blur import Blur
from noise_layers.jpeg import JPEG

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

class Watermark():
    def __init__(self, FLAGS, accelerator):
        self.FLAGS = FLAGS
        self.device = accelerator.device
        self.net_Necst = None
        self.net_Attack = None
        self.noiser = Noiser(self.device, accelerator)
        self.accelerator = accelerator
    
    def eval(self):
        self.net_Encdec.eval()
        self.net_Dis.eval()
        if self.net_Attack is not None:
            self.net_Attack.eval()

    def train(self):
        self.net_Encdec.train()
        self.net_Dis.train()
        if self.net_Attack is not None:
            self.net_Attack.train()

    def load_eval_waves(self, checkpoint):
        print("pretrain_checkpoint",checkpoint.keys())
        self.net_Encdec.load_state_dict(checkpoint['enc-dec-model'])
        self.optim_EncDec.load_state_dict(checkpoint['enc-dec-optim'])
        self.net_Dis.load_state_dict(checkpoint['discrim-model'])
        self.optim_Dis.load_state_dict(checkpoint['discrim-optim'])
        if self.net_Necst is not None:
           self.net_Necst.load_state_dict(checkpoint['necst'])
           self.net_Necst.eval()

    def load_checkpoint(self, checkpoint):
        print("checkpoint",checkpoint.keys())
        self.net_Encdec.load_state_dict(checkpoint['enc-dec-model'])
        self.optim_EncDec.load_state_dict(checkpoint['enc-dec-optim'])
        self.net_Dis.load_state_dict(checkpoint['discrim-model'])
        self.optim_Dis.load_state_dict(checkpoint['discrim-optim'])

        if self.net_Attack is not None:
           self.net_Attack.load_state_dict(checkpoint['attack-model'])
           self.optim_Attack.load_state_dict(checkpoint['attack-optim'])
        if self.net_Necst is not None:
           self.net_Necst.load_state_dict(checkpoint['necst'])
           self.net_Necst.eval()
    
    def save_checkpoint(self, iter, pretrain_flag=False):

        checkpoint = {
            'enc-dec-model': self.net_Encdec.state_dict(),
            'enc-dec-optim': self.optim_EncDec.state_dict(),
            'discrim-model': self.net_Dis.state_dict(),
            'discrim-optim': self.optim_Dis.state_dict(),
            'iter': iter,
            'pretrain_flag': pretrain_flag
        }
        if self.net_Attack is not None:
            checkpoint['attack-model'] = self.net_Attack.state_dict()
            checkpoint['attack-optim'] = self.optim_Attack.state_dict()

        if self.net_Necst is not None:
            checkpoint['necst'] = self.net_Necst.state_dict()
        
        return checkpoint
    
    def pretrain_necst(self):
        if self.net_Necst is not None:
            self.net_Necst.pretrain()
            self.net_Necst.eval()

    def encode(self, images, messages):
        if self.net_Necst:
            messages = self.net_Necst.encode(messages)

        encoded_images = self.net_Encdec.encode(images,messages)
        return encoded_images, messages
    
    def decode(self, encoded_images):
        return self.net_Encdec.decode(encoded_images)


    def perturb_image(self, encoded_images, images=None):

        if self.net_Attack is not None:
            adv_images = self.net_Attack(encoded_images)

        elif self.FLAGS.adv_type == 'hidden':
            noised_and_cover = self.noiser([encoded_images, images])
            adv_images = noised_and_cover[0]

        return adv_images

    def setup_optimizer(self,lr=1e-4,weight_decay=1e-3):
        self.optim_EncDec = torch.optim.Adam(self.net_Encdec.parameters())
        self.optim_Dis = torch.optim.Adam(self.net_Dis.parameters())
        if self.net_Attack is not None:
            self.optim_Attack = torch.optim.Adam(self.net_Attack.parameters(),lr=lr,weight_decay=weight_decay)

    
    def setup_model(self):
        self.net_Encdec = EncoderDecoder(self.FLAGS).to(self.device)
        self.net_Dis = Discriminator(self.FLAGS).to(self.device)
        if bool(self.FLAGS.load_eval_waves) != True:
            self.setup_attack_network()

        if self.FLAGS.adv_type != 'none' and self.FLAGS.adv_type != 'hidden':
            self.net_Necst = NECST(self.FLAGS, self.device, self.accelerator).to(self.device)


    def setup_attack_network(self):
        from model.vit_mask import ViT
        from model.generator import Generator
        from model.cnn_dct import DCT_CNN
        from model.transformer import ViT_RGB
        from model.parallel import Parallel
        from model.cascade import Cascade

        if self.FLAGS.adv_type == 'cnn':
            self.net_Attack = Generator().to(self.device)  
        elif self.FLAGS.adv_type == 'dct_cnn':
            self.net_Attack = DCT_CNN(self.accelerator).to(self.device)  
        elif self.FLAGS.adv_type == 'transformer':
            self.net_Attack = ViT_RGB( image_size=self.FLAGS.image_size,
                                       patch_size=8,
                                       dim=256,
                                       depth=6,
                                       heads=12,
                                       mlp_dim=256,
                                       dropout=0.0,
                                       emb_dropout=0.0
                                    ).to(self.device)  
        elif self.FLAGS.adv_type == 'dct_transformer':
            self.net_Attack = ViT( image_size=self.FLAGS.image_size,
                                   patch_size=8,
                                   dim=256,
                                   depth=6,
                                   heads=12,
                                   mlp_dim=256,
                                   dropout=0.0,
                                   emb_dropout=0.0,
                                   accelerator=self.accelerator
                                ).to(self.device)
        elif self.FLAGS.adv_type == 'parallel':
            self.net_Attack = Parallel(self.FLAGS, accelerator=self.accelerator).to(self.device)  
        elif self.FLAGS.adv_type == 'cascade':
            self.net_Attack = Cascade(self.FLAGS, accelerator=self.accelerator).to(self.device)  
        
    def validation(self,validation_loader):
        from torchjpeg.metrics import ssim as SSIM
        from torchjpeg.metrics import psnrb as PSNRB
        from utils import PSNR_RGB, PSNR_YUV

        validation_dict = {
                'identity_accuracy': 0.00,
                'rotation_accuracy': 0.00,
                'resizedcrop_accuracy': 0.00,
                'erasing_accuracy': 0.00,
                'brightness_accuracy': 0.00,
                'contrast_accuracy': 0.00,
                'blurring_accuracy': 0.00,
                'noise_accuracy': 0.00,
                'compression_accuracy': 0.00,
                'PSNR_RGB': 0.00,
                'PSNR_Y':0.00,
                'PSNR_U':0.00,
                'PSNR_V':0.00,
                'PSNR_B':0.00,
                'SSIM':0.00,
                'distortion_acc_avg':0.00,
                }
        
        count = 0
        with torch.no_grad():
            for images, _ in tqdm(validation_loader):
                count += 1
                images = images.to(self.device)
                messages = torch.Tensor(np.random.choice([0, 1], (images.shape[0], self.FLAGS.message_length))).to(self.device)
                encoded_images, _ = self.encode(images, messages)
                # encoded_images = torch.clip((encoded_images+1)/2,min=0.0,max=1.0)
                # images = torch.clip((images+1)/2,min=0.0,max=1.0)
                validation_dict['PSNR_RGB']  += PSNR_RGB(encoded_images.clone(),images.clone())
                psnr_Y, psnr_U, psnr_V = PSNR_YUV(encoded_images.clone(), images.clone())
                validation_dict['PSNR_Y'] += psnr_Y
                validation_dict['PSNR_U'] += psnr_U
                validation_dict['PSNR_V'] += psnr_V
                validation_dict['PSNR_B'] += PSNRB(encoded_images.clone(),images.clone()).mean().item()
                validation_dict['SSIM'] += SSIM(encoded_images.clone(),images.clone()).mean().item()

                # encoded_images, _ = self.encode(images, messages)

                ### identity ###
                decoded_messages = self.decode(encoded_images)
                if self.net_Necst is not None:
                    decoded_messages = self.net_Necst.decode(decoded_messages)
                decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (messages.shape[0] * messages.shape[1])).item()
                validation_dict['identity_accuracy'] += bitwise_avg_err
                ### rotation ###
                distortion_images_list = []
                for i in range(len(encoded_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(encoded_images[i]), "rotation")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(self.device)
                distortion_predicted = self.decode(distortion_images)
                if self.net_Necst is not None:
                    distortion_predicted = self.net_Necst.decode(distortion_predicted)
                distortion_rounded = distortion_predicted.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(distortion_rounded - messages.detach().cpu().numpy())) / (messages.shape[0] * messages.shape[1])).item()
                validation_dict['rotation_accuracy'] += bitwise_avg_err
                ### resizedcrop ###
                distortion_images_list = []
                for i in range(len(encoded_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(encoded_images[i]), "resizedcrop")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(self.device)
                distortion_predicted = self.decode(distortion_images)
                if self.net_Necst is not None:
                    distortion_predicted = self.net_Necst.decode(distortion_predicted)
                distortion_rounded = distortion_predicted.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(distortion_rounded - messages.detach().cpu().numpy())) / (messages.shape[0] * messages.shape[1])).item()
                validation_dict['resizedcrop_accuracy'] += bitwise_avg_err
                ### erasing ###
                distortion_images_list = []
                for i in range(len(encoded_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(encoded_images[i]), "erasing")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(self.device)
                distortion_predicted = self.decode(distortion_images)
                if self.net_Necst is not None:
                    distortion_predicted = self.net_Necst.decode(distortion_predicted)
                distortion_rounded = distortion_predicted.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(distortion_rounded - messages.detach().cpu().numpy())) / (messages.shape[0] * messages.shape[1])).item()
                validation_dict['erasing_accuracy'] += bitwise_avg_err
                ### brightness ###
                distortion_images_list = []
                for i in range(len(encoded_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(encoded_images[i]), "brightness")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(self.device)
                distortion_predicted = self.decode(distortion_images)
                if self.net_Necst is not None:
                    distortion_predicted = self.net_Necst.decode(distortion_predicted)
                distortion_rounded = distortion_predicted.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(distortion_rounded - messages.detach().cpu().numpy())) / (messages.shape[0] * messages.shape[1])).item()
                validation_dict['brightness_accuracy'] += bitwise_avg_err
                ### contrast ###
                distortion_images_list = []
                for i in range(len(encoded_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(encoded_images[i]), "contrast")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(self.device)
                distortion_predicted = self.decode(distortion_images)
                if self.net_Necst is not None:
                    distortion_predicted = self.net_Necst.decode(distortion_predicted)
                distortion_rounded = distortion_predicted.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(distortion_rounded - messages.detach().cpu().numpy())) / (messages.shape[0] * messages.shape[1])).item()
                validation_dict['contrast_accuracy'] += bitwise_avg_err
                ### blurring ###
                distortion_images_list = []
                for i in range(len(encoded_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(encoded_images[i]), "blurring")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(self.device)
                distortion_predicted = self.decode(distortion_images)
                if self.net_Necst is not None:
                    distortion_predicted = self.net_Necst.decode(distortion_predicted)
                distortion_rounded = distortion_predicted.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(distortion_rounded - messages.detach().cpu().numpy())) / (messages.shape[0] * messages.shape[1])).item()
                validation_dict['blurring_accuracy'] += bitwise_avg_err
                ### noise ###
                distortion_images_list = []
                for i in range(len(encoded_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(encoded_images[i]), "noise")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(self.device)
                distortion_predicted = self.decode(distortion_images)
                if self.net_Necst is not None:
                    distortion_predicted = self.net_Necst.decode(distortion_predicted)
                distortion_rounded = distortion_predicted.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(distortion_rounded - messages.detach().cpu().numpy())) / (messages.shape[0] * messages.shape[1])).item()
                validation_dict['noise_accuracy'] += bitwise_avg_err
                ### compression ###
                distortion_images_list = []
                for i in range(len(encoded_images)):
                    adv_img = apply_single_distortion(T.ToPILImage()(encoded_images[i]), "compression")
                    distortion_images_list.append(T.ToTensor()(adv_img).unsqueeze(0))
                distortion_images = torch.cat(distortion_images_list, dim=0).to(self.device)
                distortion_predicted = self.decode(distortion_images)
                if self.net_Necst is not None:
                    distortion_predicted = self.net_Necst.decode(distortion_predicted)
                distortion_rounded = distortion_predicted.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(distortion_rounded - messages.detach().cpu().numpy())) / (messages.shape[0] * messages.shape[1])).item()
                validation_dict['compression_accuracy'] += bitwise_avg_err

        validation_dict['identity_accuracy'] = (1.0-validation_dict['identity_accuracy']/count) * 100
        validation_dict['rotation_accuracy'] = (1.0-validation_dict['rotation_accuracy']/count) * 100
        validation_dict['resizedcrop_accuracy'] = (1.0-validation_dict['resizedcrop_accuracy']/count) * 100
        validation_dict['erasing_accuracy'] = (1.0-validation_dict['erasing_accuracy']/count) * 100
        validation_dict['brightness_accuracy'] = (1.0-validation_dict['brightness_accuracy']/count) * 100
        validation_dict['contrast_accuracy'] = (1.0-validation_dict['contrast_accuracy']/count) * 100
        validation_dict['blurring_accuracy'] = (1.0-validation_dict['blurring_accuracy']/count) * 100
        validation_dict['noise_accuracy'] = (1.0-validation_dict['noise_accuracy']/count) * 100
        validation_dict['compression_accuracy'] = (1.0-validation_dict['compression_accuracy']/count) * 100
        validation_dict['PSNR_RGB'] /=  count
        validation_dict['PSNR_Y'] /=  count
        validation_dict['PSNR_U'] /=  count
        validation_dict['PSNR_V'] /=  count
        validation_dict['PSNR_B'] /=  count
        validation_dict['SSIM'] /=  count

        distortion_acc_avg = validation_dict['identity_accuracy'] + validation_dict['rotation_accuracy'] \
        + validation_dict['resizedcrop_accuracy'] + validation_dict['erasing_accuracy'] + validation_dict['brightness_accuracy'] \
        + validation_dict['contrast_accuracy'] + validation_dict['blurring_accuracy'] + validation_dict['noise_accuracy'] \
        + validation_dict['compression_accuracy']
        distortion_acc_avg /= 9
        validation_dict['distortion_acc_avg'] = distortion_acc_avg

        encoded_images, _ = self.encode(images.to(self.device), messages)
        if self.FLAGS.adv_type == 'hidden':
            adv_images = self.perturb_image(encoded_images,images)
        else:
            adv_images = self.perturb_image(encoded_images,None)
        return validation_dict, images, encoded_images, adv_images
    
    def decoding_fingerprint(self, dataloader, batch_size):
        count = 0
        decoding_accuracy = 0.0
        batch_size = batch_size
        with torch.no_grad():
            for (generated_image, filename, message) in tqdm(dataloader):
                count += 1
                generated_image = generated_image.to(self.device)
               
                gt_message = message
                decoded_messages = self.decode(generated_image)
                if self.net_Necst is not None:
                    decoded_messages = self.net_Necst.decode(decoded_messages)
                decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(decoded_rounded - gt_message.detach().cpu().numpy())) / (gt_message.shape[0] * gt_message.shape[1])).item()
                decoding_accuracy += bitwise_avg_err
                
            decoded_accuracy = (1.0-decoding_accuracy/count) * 100
            return decoded_accuracy

    
                      


    
        