python3 ../detect_fingerprints.py \
--decoder_path ./checkpoints/Stegastamp_Parallel/stegastamp_30_31082024_12:38:49_decoder_60000.pth \
--data_dir ./Datasets/coco \
--image_resolution 128 \
--output_dir OUTPUT_PATH \
--batch_size 1 \
--cuda 0