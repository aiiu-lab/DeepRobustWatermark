import os
import csv
import gc
import numpy as np
import torch
from absl import flags, app
import torchvision
from torchvision import datasets, transforms
import utils
from losses import loss_map
from model.watermark import Watermark
from tqdm import tqdm
from accelerate import Accelerator


FLAGS = flags.FLAGS

### Eval Configuration ###
flags.DEFINE_integer('batch_size', 32, "size of batch")
flags.DEFINE_string('dataset', '/home/Documents/coco', 'Dataset used')
flags.DEFINE_string('load_checkpoint', None, 'load from checkpoint')
flags.DEFINE_string('load_eval_waves', None, 'load from checkpoint')
flags.DEFINE_integer('redundant_length', 120, "Please set to 30 while you are evaluating HiDDeN")
flags.DEFINE_integer('message_length', 30, "length of message")
flags.DEFINE_integer('image_size', 128, "size of the images generated")

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

def eval(accelerator):
    device = accelerator.device
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(size=(FLAGS.image_size, FLAGS.image_size )),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ]),
        'test': transforms.Compose([
            transforms.Resize(size=(FLAGS.image_size, FLAGS.image_size)),
            transforms.ToTensor()
        ])
    }

    validation_images = datasets.ImageFolder(FLAGS.dataset + "/test", data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    
    accelerator.print("validation file_count:{}".format(len(validation_loader.dataset)))

    model = Watermark(FLAGS, accelerator)
    model.setup_model()
    model.setup_optimizer(lr=FLAGS.lr,weight_decay=FLAGS.weight_decay)

    model, data_ld = accelerator.prepare(
        model, validation_loader
    )

    ## Loading checkpoint ##
    if FLAGS.load_checkpoint:
        accelerator.print("Loading checkpoints")
        checkpoint = torch.load(FLAGS.load_checkpoint, map_location=device)
        model.load_checkpoint(checkpoint)
        del checkpoint
    else:
        assert FLAGS.load_checkpoint == True

    ## Evaluation ##
    if accelerator.is_main_process:
        model.eval()
        validation_dict, _, _, _ = model.validation(data_ld)
        print("\n########## Validation ##########\n")
        print("identity_accuracy          :", round(validation_dict['identity_accuracy'], 3))
        print("rotation_accuracy          :", round(validation_dict['rotation_accuracy'], 3))
        print("resizedcrop_accuracy       :", round(validation_dict['resizedcrop_accuracy'], 3))
        print("erasing_accuracy           :", round(validation_dict['erasing_accuracy'], 3))
        print("brightness_accuracy        :", round(validation_dict['brightness_accuracy'], 3))
        print("contrast_accuracy          :", round(validation_dict['contrast_accuracy'], 3))
        print("blurring_accuracy          :", round(validation_dict['blurring_accuracy'], 3))
        print("noise_accuracy             :", round(validation_dict['noise_accuracy'], 3))
        print("compression_accuracy       :", round(validation_dict['compression_accuracy'], 3))
        print("PSNR_RGB                   :", round(validation_dict['PSNR_RGB'], 3))
        print("PSNR_Y                     :", round(validation_dict['PSNR_Y'], 3))
        print("PSNR_U                     :", round(validation_dict['PSNR_U'], 3))
        print("PSNR_V                     :", round(validation_dict['PSNR_V'], 3))
        print("PSNR_B                     :", round(validation_dict['PSNR_B'], 3))
        print("SSIM                       :", round(validation_dict['SSIM'], 3))
        print("distortion_acc_avg         :", round(validation_dict['distortion_acc_avg'], 3))
        print("\n###################################\n")

        gc.collect()
        torch.cuda.empty_cache()

def main(argv):
    utils.set_seed(FLAGS.seed)
    accelerator = Accelerator()
    eval(accelerator=accelerator)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass   


        

        




            
        


    







