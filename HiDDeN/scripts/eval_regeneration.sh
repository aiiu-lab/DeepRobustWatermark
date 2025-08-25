### Eval regeneration attacks ###
accelerate launch --main_process_port 29700 ../eval_regeneration.py \
--image_size 128 \
--batch_size 16 \
--dataset ./Datasets/coco \
--adv_type parallel \
--redundant_length 120 \
--message_length 30 \
--load_eval_waves ./checkpoints/Parallel_best.pyt