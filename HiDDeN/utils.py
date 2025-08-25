import os
import random
from contextlib import contextmanager
import torch
import numpy as np
import torchvision
import torch.nn.functional as F

def save_images(original_images, watermarked_images, epoch, folder, resize_to=None, imgtype="enc"):
    images = original_images[:original_images.shape[0], :, :, :].cpu()
    watermarked_images = watermarked_images[:watermarked_images.shape[0], :, :, :].cpu()

    # scale values to range [0, 1] from original range of [-1, 1]
    images = (images + 1) / 2
    watermarked_images = (watermarked_images + 1) / 2

    if resize_to is not None:
        images = F.interpolate(images, size=resize_to)
        watermarked_images = F.interpolate(watermarked_images, size=resize_to)

    stacked_images = torch.cat([images, watermarked_images], dim=0)
    filename = os.path.join(folder, 'epoch-{}-{}.png'.format(epoch, imgtype))
    torchvision.utils.save_image(stacked_images, filename)

def infiniteloop(dataloader, message_length ,device):
    while True:
        for x, _ in iter(dataloader):
            message = torch.Tensor(np.random.choice([0, 1], (x.shape[0], message_length))).to(device)
            x = x.to(device)
            yield x, message

@contextmanager
def module_no_grad(m: torch.nn.Module):
    requires_grad_dict = dict()
    for name, param in m.named_parameters():
        requires_grad_dict[name] = param.requires_grad
        param.requires_grad_(False)
    yield m
    for name, param in m.named_parameters():
        param.requires_grad_(requires_grad_dict[name])

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def ema(source, target, decay):
    source_dict = source.state_dict()
    target_dict = target.state_dict()
    for key in source_dict.keys():
        target_dict[key].data.copy_(
            target_dict[key].data * decay +
            source_dict[key].data * (1 - decay))

def write_configurations(model,config_dict, file_path):
    f = open(file_path+'/config.txt', 'w')
    for attr, value in config_dict.items():
        f.write('{}:{}\n'.format(attr,value))
    
    f.write('{}:\n{}\n\n'.format("Encoder_Decoder",model.net_Encdec))
    f.write('{}:\n{}\n\n'.format("Discriminator",model.net_Dis))
    f.write('{}:\n{}\n\n'.format("noiser",model.noiser))
    if model.net_Necst is not None:
        f.write('{}:\n{}\n\n'.format("Necst",model.net_Necst))
    if model.net_Attack is not None:
        f.write('{}:\n{}\n\n'.format("Attack Netowrk",model.net_Attack))

    f.close()

def rgb2yuv(image_rgb, image_yuv_out):
    image_yuv_out[:, 0, :, :] = 0.299 * image_rgb[:, 0, :, :].clone() + 0.587 * image_rgb[:, 1, :, :].clone() + 0.114 * image_rgb[:, 2, :, :].clone()
    image_yuv_out[:, 1, :, :] = -0.14713 * image_rgb[:, 0, :, :].clone() + -0.28886 * image_rgb[:, 1, :, :].clone() + 0.436 * image_rgb[:, 2, :, :].clone()
    image_yuv_out[:, 2, :, :] = 0.615 * image_rgb[:, 0, :, :].clone() + -0.51499 * image_rgb[:, 1, :, :].clone() + -0.10001 * image_rgb[:, 2, :, :].clone()


def PSNR_YUV(image, target):
    image_yuv = torch.empty_like(image)
    target_yuv = torch.empty_like(target)
    rgb2yuv(image, image_yuv)
    rgb2yuv(target, target_yuv)
    psnr_Y = PSNR_RGB(image_yuv[:,0],target_yuv[:,0])
    psnr_U = PSNR_RGB(image_yuv[:,1],target_yuv[:,1])
    psnr_V = PSNR_RGB(image_yuv[:,2],target_yuv[:,2])
    return psnr_Y, psnr_U, psnr_V


def PSNR_RGB(img1, img2):
    img1 = img1 * 255
    img2 = img2 * 255

    mse = torch.mean((img1 - img2) ** 2)
    return (20 * torch.log10(255.0 / torch.sqrt(mse))).mean().item()
