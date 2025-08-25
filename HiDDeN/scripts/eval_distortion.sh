### Eval distortions ###
python3 ../eval_distortions.py \
--batch_size 10 \
--dataset ./Datasets/celeba \
--adv_type parallel \
--redundant_length 120 \
--message_length 30 \
--load_checkpoint ./checkpoints/Parallel_best.pyt