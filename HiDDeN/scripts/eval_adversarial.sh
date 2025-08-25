### Eval adversarial attacks ###
accelerate launch --main_process_port 29700 ../eval_adversarial.py \
--image_size 128 \
--batch_size 32 \
--dataset ./Datasets/coco \
--adv_type hidden \
--redundant_length 30 \
--message_length 30 \
--load_eval_waves ./checkpoints/Hidden_best.pyt