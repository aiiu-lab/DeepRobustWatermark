# accelerate launch --main_process_port 29600 ../train.py \
# --batch_size 32 \
# --dataset ../coco \
# --pretrain_iter 150000 \
# --adv_type parallel \
# --name Parallel_BIT30 \
# --redundant_length 120 \
# --message_length 30 


CUDA_VISIBLE_DEVICES=9 python3 ../train.py \
--batch_size 32 \
--dataset ./Datasets/coco \
--pretrain_iter 0 \
--adv_type parallel \
--name Parallel_BIT30 \
--redundant_length 120 \
--message_length 30 \
# --load_checkpoint /scratch1/users/jason890425/DeepRobustWatermark/HiDDeN/checkpoints/Parallel_best.pyt

### adv_type ###
# choice : [none,hidden,cnn,transformer,dct_cnn,dct_transformer,parallel,cascade]

