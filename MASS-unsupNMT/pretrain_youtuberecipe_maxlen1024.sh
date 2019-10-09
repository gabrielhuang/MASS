#!/bin/bash

export RANK=0
export WORLD_SIZE=1
export NGPU=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=6666
#export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=1

#python train.py                                      \
python -m torch.distributed.launch --nnodes=1 --nproc_per_node=$NGPU train.py \
	--exp_name unsupMT_youtuberecipe \
	--data_path ./data/processed/asr-cap/                  \
	--lgs 'asr-cap'                                        \
	--mass_steps 'asr,cap'                                 \
	--encoder_only false                                 \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--tokens_per_batch 3000                              \
	--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.0001 \
	--epoch_size 200000                                  \
	--max_epoch 100                                      \
	--eval_bleu true                                     \
	--word_mass 0.5                                      \
	--min_len 5                                        \
	--max_len 1024                                        \
	--exp_id experiment5maxlen1024   \
