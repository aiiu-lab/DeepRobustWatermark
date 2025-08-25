### Embedding fingerprints ###
accelerate launch --main_process_port 29700 ../embed_fingerprints.py \
--image_size 128 \
--batch_size 1 \
--dataset ./Datasets/coco \
--out_dir OUTPUT_PATH \
--adv_type parallel \
--redundant_length 120 \
--message_length 30 \
--load_checkpoint ./checkpoints/Parallel_best.pyt