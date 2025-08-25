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


FLAGS = flags.FLAGS

### Embedding Configuration ###
flags.DEFINE_integer('image_size', 128, "size of the images generated")
flags.DEFINE_integer('batch_size', 32, "size of batch")
flags.DEFINE_string('dataset', '/home/Documents/coco', 'Dataset used')
flags.DEFINE_string('out_dir', "output/", "Folder output sample_image")
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

def embed_fingerprint(accelerator):
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
    if FLAGS.load_checkpoint:
        print("Loading checkpoints")
        checkpoint = torch.load(FLAGS.load_checkpoint, map_location=device)
        model.load_checkpoint(checkpoint)
        del checkpoint
    else:
        assert FLAGS.load_checkpoint == True

    model.eval()
    with torch.no_grad():
        for  (img_data, filename) in tqdm(data_ld):
            if (len(Data.dataset.data_list)%torch.cuda.device_count()) == 0:
                img_data = img_data.to(device)
                message = torch.Tensor(np.random.choice([0, 1], (img_data.shape[0], FLAGS.message_length))).to(device)
                encoded_image, _ = model.encode(img_data, message)
                filename = filename[0].split('.')[0] + ".png"
                torchvision.utils.save_image(encoded_image, os.path.join(FLAGS.out_dir + '/imgs', f"{filename}"), padding=0)
                with open(FLAGS.out_dir + '/filename_messages.csv', 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    message_str = "".join(map(str, message[0].long().tolist()))
                    writer.writerow([filename, message_str])
            else:
                print("The number of GPUs must me divisible by the amount of data")
                break

def main(argv):
    utils.set_seed(FLAGS.seed)
    accelerator = Accelerator()
    if accelerator.is_main_process:
        if not os.path.exists(FLAGS.out_dir):
            os.mkdir(FLAGS.out_dir)
        if not os.path.exists(FLAGS.out_dir + '/imgs'):
            os.mkdir(FLAGS.out_dir + '/imgs')

    embed_fingerprint(accelerator=accelerator)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass  