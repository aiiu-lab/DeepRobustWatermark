accelerate launch --main_process_port 29800 ../eval_adversarial.py \
--data_dir ./Datasets/coco/test/test_class \
--image_resolution 128 \
--fingerprint_length 30 \
--batch_size 64 \
--encoder_path ./checkpoints/Stegastamp_Parallel/stegastamp_30_31082024_12:38:49_encoder_60000.pth \
--decoder_path ./checkpoints/Stegastamp_Parallel/stegastamp_30_31082024_12:38:49_decoder_60000.pth \
--dec_enc_optim ./checkpoints/Stegastamp_Parallel/stegastamp_30_31082024_12:38:49_optim_60000.pth