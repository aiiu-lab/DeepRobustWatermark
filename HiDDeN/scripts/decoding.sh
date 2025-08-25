### Decoding fingerprints ###
accelerate launch --main_process_port 29700 ../decode_fingerprints.py \
--image_size 128 \
--batch_size 1 \
--csv_file PATH_TO_CSV_FILE \
--evl_dataset WATERMARKED_IMG_PATH \
--out_dir  OUTPUT_PATH \
--adv_type parallel \
--redundant_length 120 \
--message_length 30 \
--load_checkpoint ./checkpoints/Parallel_best.pyt