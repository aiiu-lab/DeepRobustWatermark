### Train Stegastamp ###
python3 ../train.py \
--data_dir ./Datasets/coco \
--image_resolution 128 \
--output_dir OUTPUT_PATH \
--fingerprint_length 30 \
--batch_size 64 \
--cuda 0 \
--num_epochs 1000

### Train Stegastamp_Parallel ###
python3 ../train_Parallel.py \
--data_dir ./Datasets/coco \
--image_resolution 128 \
--output_dir PATH_TO_DATA \
--fingerprint_length 30 \
--batch_size 64 \
--cuda 0 \
--num_epochs 1000000 \
--encoder_path PATH_TO_STEGASTAMP_ENCODER \
--decoder_path PATH_TO_STEGASTAMP_DECODER \
--dec_enc_optim PATH_TO_ENC_DEC_OPTIM 
# --attack_path resume_from_trained_attack_network