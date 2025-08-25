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

###########################
FLAGS = flags.FLAGS

### Training Configuration ###
flags.DEFINE_string('name', 'exp', 'experiments name')
flags.DEFINE_string('load_eval_waves', None, 'load from pretrain_checkpoint')
flags.DEFINE_string('load_checkpoint', None, 'load from checkpoint')
flags.DEFINE_string('dataset', '/home/Documents/coco', 'Dataset used')
flags.DEFINE_string('out_dir', "output/", "Folder output sample_image")
flags.DEFINE_integer('image_size', 128, "size of the images generated")
flags.DEFINE_integer('batch_size', 32, "size of batch")
flags.DEFINE_integer('nc', 3, "Channel Dimension of image")
flags.DEFINE_integer('seed', 0, "random seed")
flags.DEFINE_bool('cuda', True, 'Flag using GPU')
flags.DEFINE_integer('pretrain_iter', 250000, 'iterations of HiDDeN identity pretrain')
flags.DEFINE_integer('eval_every', 5000, "validation bit error with every this iteration ")
flags.DEFINE_integer('print_every', 1000, "print training information with every this iteration")
flags.DEFINE_integer('redundant_length', 120, "length of redundant message")
flags.DEFINE_integer('message_length', 30, "length of message")
flags.DEFINE_integer('iter', 1000000, "number of iterations model trained")
flags.DEFINE_float('lr', 1e-5, "learning rate of attack module")
flags.DEFINE_float('weight_decay', 1e-3, "weight decay for attack optimizer")

### Loss Weight ###
flags.DEFINE_float('pre_coef_discrim_loss_fake', 0.15, "weight of loss fake for pretraining")
flags.DEFINE_float('pre_coef_enc_dec_img_loss', 1.0, "weight of watermarking image loss for pretraining")
flags.DEFINE_float('pre_coef_enc_dec_msg_loss', 5.0, "weight of watermarking message loss for pretraining")

flags.DEFINE_float('gen_coef_img_loss', 15.0, "weight of attack network image loss")
flags.DEFINE_float('gen_coef_msg_loss', 1.0, "weight of attack network message loss")

flags.DEFINE_float('coef_discrim_loss_fake', 0.01, "weight of loss fake ")
flags.DEFINE_float('coef_enc_dec_img_loss', 1.5, "weight of watermarking image loss")
flags.DEFINE_float('coef_enc_dec_msg_encloss', 0.3, "weight of watermarking message loss for watermarked images")
flags.DEFINE_float('coef_enc_dec_msg_advloss', 0.2, "weight of watermarking message loss for adversarial images")

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

def train(accelerator):
    # accelerator = Accelerator()
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
    print("Loading images...")
    train_images = datasets.ImageFolder(FLAGS.dataset + "/train", data_transforms['train'])
    train_loader = torch.utils.data.DataLoader(train_images, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)

    validation_images = datasets.ImageFolder(FLAGS.dataset + "/test", data_transforms['test'])
    validation_loader = torch.utils.data.DataLoader(validation_images, batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
    
    accelerator.print("training file_count:{} validation file_count:{}".format(len(train_loader.dataset),len(validation_loader.dataset)))
    accelerator.print("Print information every {} step".format(FLAGS.print_every))

    model = Watermark(FLAGS, accelerator)
    model.setup_model()
    model.setup_optimizer(lr=FLAGS.lr,weight_decay=FLAGS.weight_decay)

    model, train_loader, validation_loader = accelerator.prepare(
        model, train_loader, validation_loader
    )

    looper_train = utils.infiniteloop(train_loader, FLAGS.message_length, device)

    ## Loading checkpoint ##
    if FLAGS.load_eval_waves:
        checkpoint = torch.load(FLAGS.load_eval_waves, map_location=device)
        model.load_eval_waves(checkpoint)
        start = int(checkpoint['iter']) + 1
        accelerator.print("Loaded pretrain_checkpoint from iter:{} in dir{}".format(start, FLAGS.out_dir))
        best_avg_acc = 0
        model.pretrain_necst()
        del checkpoint
    elif FLAGS.load_checkpoint:
        checkpoint = torch.load(FLAGS.load_checkpoint, map_location=device)
        model.load_checkpoint(checkpoint)
        start = int(checkpoint['iter']) + 1
        accelerator.print("Loaded checkpoint from iter:{} in dir{}".format(start, FLAGS.out_dir))
        best_avg_acc = 0
        model.pretrain_necst()
        del checkpoint
    else:
        start = 1
        utils.write_configurations(model,FLAGS.flag_values_dict(),FLAGS.out_dir)
        accelerator.print("output dir :", FLAGS.out_dir)
        best_avg_acc = 0
        model.pretrain_necst()

    ## Define loss functions ##
    dis_loss = loss_map[FLAGS.loss]
    mse_loss = torch.nn.MSELoss()

    ## Pretraining with HiDDeN identity ##
    if FLAGS.pretrain_iter > 0:
        for i in tqdm(range(1, FLAGS.pretrain_iter+1), desc="Pretrain"):
            ########### training discriminator ###############
            model.optim_Dis.zero_grad()
            images, messages = next(looper_train)
            pred_real = model.net_Dis(images)
            encoded_images, messages = model.encode(images, messages)
            decoded_messages = model.decode(encoded_images)
            pred_fake = model.net_Dis(encoded_images.detach())
            loss_fake, loss_real = dis_loss(pred_fake, pred_real)
            loss_D = 0.5 *  (loss_real + loss_fake) 
            accelerator.backward(loss_D)
            model.optim_Dis.step()

            ########### training Encoder Decoder ###############
            model.optim_EncDec.zero_grad()
            pred_fake = model.net_Dis(encoded_images)
            loss_fake = dis_loss(pred_fake)
            enc_dec_image_loss = mse_loss(encoded_images, images)
            enc_dec_message_encloss = mse_loss(decoded_messages, messages)
            loss_ED = FLAGS.pre_coef_discrim_loss_fake * loss_fake + FLAGS.pre_coef_enc_dec_img_loss * enc_dec_image_loss + FLAGS.pre_coef_enc_dec_msg_loss * enc_dec_message_encloss
            accelerator.backward(loss_ED)
            model.optim_EncDec.step()

            if (i == 1 or i % FLAGS.print_every == 0) and accelerator.is_main_process:
                model.eval()
                with torch.no_grad():
                    decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                    bitwise_avg_err = (np.sum(np.abs(decoded_rounded - messages.detach().cpu().numpy())) / (
                                messages.shape[0] * messages.shape[1])).item()
                    psnr_rgb  = utils.PSNR_RGB(encoded_images.clone(),images.clone())

                    print("\n########## Iteration:{} ##########\n".format(i))
                    print("psnr_rgb               :", psnr_rgb)
                    print("loss_Discim            :", loss_D.item())
                    print("loss_Encdec            :", loss_ED.item())
                    print("encdec_message_encloss :", enc_dec_message_encloss.item())
                    print("encdec_image_loss      :", enc_dec_image_loss.item())
                    print("bitwise_avg_err        :", bitwise_avg_err)
                    print("\n###################################\n")
                model.train()

        if accelerator.is_main_process:
            checkpoint = model.save_checkpoint(iter=FLAGS.pretrain_iter,pretrain_flag=True)
            checkpoint['best_avg_acc'] = best_avg_acc
            torch.save(checkpoint, FLAGS.out_dir + "checkpoint/" + "pretrain_iter{}.pyt".format(FLAGS.pretrain_iter))
            accelerator.print('iter {} Saving pretrain checkpoint done.'.format(FLAGS.pretrain_iter))

    ##### Start Training ####
    pbar = tqdm(range(start, FLAGS.iter+1), desc="Train")
    for i in pbar:
        pbar.set_postfix(iter=i)
        ########### training generator ###############
        images, messages = next(looper_train)
        ori_messages = messages.clone().detach()
        
        if FLAGS.adv_type != 'none' and FLAGS.adv_type != 'hidden':
            encoded_images, redundant_messages = model.encode(images, messages)
            encoded_images = encoded_images.detach()
            redundant_messages = redundant_messages.detach()

            for _ in range(FLAGS.n_gen):
                model.optim_Attack.zero_grad()
                adv_images = model.perturb_image(encoded_images,None) 
                adv_dec_messages = model.decode(adv_images)
                gen_image_loss = mse_loss(adv_images, encoded_images)
                gen_message_loss = mse_loss(adv_dec_messages, redundant_messages)
                loss_G = FLAGS.gen_coef_img_loss * gen_image_loss - FLAGS.gen_coef_msg_loss * gen_message_loss
                accelerator.backward(loss_G)
                model.optim_Attack.step()

        ########### training discriminator ###############
        model.optim_Dis.zero_grad()
        pred_real = model.net_Dis(images)
        encoded_images, messages = model.encode(images, messages)
        pred_fake = model.net_Dis(encoded_images.detach())
        loss_fake, loss_real = dis_loss(pred_fake, pred_real)
        loss_D = 0.5 * (loss_real + loss_fake) 
        accelerator.backward(loss_D)
        model.optim_Dis.step()

        ########### training Encoder Decoder ###############
        model.optim_EncDec.zero_grad()
        adv_images = model.perturb_image(encoded_images, images) 
        adv_decoded_messages = model.decode(adv_images)
        pred_fake = model.net_Dis(encoded_images)
        loss_fake =  dis_loss(pred_fake)
        enc_dec_image_loss = mse_loss(encoded_images, images)
        enc_dec_message_advloss = mse_loss(adv_decoded_messages, messages)

        if FLAGS.adv_type == "hidden":
            loss_ED = FLAGS.coef_discrim_loss_fake * loss_fake + FLAGS.coef_enc_dec_img_loss * enc_dec_image_loss +\
                  FLAGS.coef_enc_dec_msg_advloss * enc_dec_message_advloss
        else:
            decoded_messages = model.decode(encoded_images)
            enc_dec_message_encloss = mse_loss(decoded_messages, messages)
            loss_ED = FLAGS.coef_discrim_loss_fake * loss_fake + FLAGS.coef_enc_dec_img_loss * enc_dec_image_loss +\
                  FLAGS.coef_enc_dec_msg_encloss * enc_dec_message_encloss + FLAGS.coef_enc_dec_msg_advloss * enc_dec_message_advloss
            
        accelerator.backward(loss_ED)
        model.optim_EncDec.step()

        if (i == 1 or i % FLAGS.print_every == 0) and accelerator.is_main_process:
            model.eval()
            with torch.no_grad():
                if FLAGS.adv_type != 'none' and FLAGS.adv_type != 'hidden':
                    decoded_messages = model.net_Necst.decode(decoded_messages)
                decoded_rounded = decoded_messages.detach().cpu().numpy().round().clip(0, 1)
                bitwise_avg_err = (np.sum(np.abs(decoded_rounded - ori_messages.detach().cpu().numpy())) / ( ori_messages.shape[0] * ori_messages.shape[1])).item()
                psnr_rgb  = utils.PSNR_RGB(encoded_images.clone(),images.clone())
                 
                validation_dict = {
                    'loss_Discrim            ': loss_D.item(),
                    'loss_Encdec             ': loss_ED.item(),
                    'enc_dec_message_encloss ': enc_dec_message_encloss.item(),
                    'enc_dec_message_advloss ': enc_dec_message_advloss.item(),
                    'encdec_image_loss       ': enc_dec_image_loss.item(),
                    'PSNR_RGB                ': psnr_rgb,
                    'bitwise_avg_err         ': bitwise_avg_err}
                
                if FLAGS.adv_type != 'none' and FLAGS.adv_type != 'hidden':
                    validation_dict['gen_image_loss          '] = gen_image_loss.item()
                    validation_dict['gen_message_loss        '] = gen_message_loss.item()

                print("\n########## Iteration:{} Training ##########\n".format(i))
                for key, value in validation_dict.items():
                    print(key + ":", value)
                print("\n###################################\n")

                with open(FLAGS.out_dir + "/train.csv", 'a', newline='') as csvfile:
                    writer = csv.writer(csvfile)
                    if (i == start) or (i == 1):
                        row_to_write = ['Iter'] + [loss_name.strip() for loss_name in validation_dict.keys()]
                        writer.writerow(row_to_write)
                    row_to_write = [str(i).zfill(6)] + ['{:.4f}'.format(loss_avg) for loss_avg in validation_dict.values()]
                    writer.writerow(row_to_write)

            torch.cuda.empty_cache()
            model.train()

        if (i == 1 or i % FLAGS.eval_every == 0) and accelerator.is_main_process :
            model.eval()
            validation_dict, val_images, encoded_images, adv_images = model.validation(validation_loader)

            utils.save_images(val_images.cpu()[:8, :, :, :],
                            encoded_images[:8, :, :, :].cpu(),
                            str(i).zfill(6),
                            os.path.join(FLAGS.out_dir, 'images'), resize_to=(128, 128), imgtype="enc")

            utils.save_images(val_images.cpu()[:8, :, :, :],
                            adv_images[:8, :, :, :].cpu(),
                            str(i).zfill(6),
                            os.path.join(FLAGS.out_dir, 'images'), resize_to=(128, 128), imgtype="adv")
    
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

            with open(FLAGS.out_dir + "/validation.csv", 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                if (i == start) or (i == 1):
                    row_to_write = ['Iter'] + [loss_name.strip() for loss_name in validation_dict.keys()]
                    writer.writerow(row_to_write)
                row_to_write = [str(i).zfill(6)] + ['{:.4f}'.format(loss_avg) for loss_avg in validation_dict.values()]
                writer.writerow(row_to_write)

            gc.collect()
            torch.cuda.empty_cache()
            model.train()

            checkpoint = model.save_checkpoint(i,pretrain_flag=False)
            checkpoint['best_avg_acc'] = best_avg_acc

            if validation_dict['Average_Acc'] > best_avg_acc:
                best_avg_acc = validation_dict['Average_Acc']
                torch.save(checkpoint, FLAGS.out_dir + "checkpoint/" + "best.pyt")
                accelerator.print("best_avg_acc:{} checkpoint is saved".format(best_avg_acc))

            torch.save(checkpoint, FLAGS.out_dir + "checkpoint/" + "latest.pyt")
            accelerator.print('Iteration {} Saving checkpoint done.'.format(i))
            del checkpoint
            del validation_dict
            gc.collect()

def main(argv):
    utils.set_seed(FLAGS.seed)
    accelerator = Accelerator()
    if accelerator.is_main_process:
        if (FLAGS.load_eval_waves or FLAGS.load_checkpoint) is None or True:
            if not os.path.exists(FLAGS.out_dir):
                os.mkdir(FLAGS.out_dir)
            run = 0
            while os.path.exists(FLAGS.out_dir + FLAGS.name + str(run) + "/"):
                run += 1
            FLAGS.out_dir = FLAGS.out_dir + FLAGS.name + str(run) + "/"
            os.mkdir(FLAGS.out_dir)
            os.mkdir(FLAGS.out_dir + "images")
            os.mkdir(FLAGS.out_dir + "checkpoint")

    train(accelerator=accelerator)


if __name__ == '__main__':
        
    try:
        app.run(main)
    except SystemExit:
        pass


