#!/bin/bash

MODEL=${1}  # pretrained/mass_enfr_1024.pth or dumped/unsupMT_enfr/mass

python train.py \
	--exp_name unsupMT_enfr                              \
	--data_path ./data/processed/en-fr/                  \
	--lgs 'en-fr'                                        \
	--mass_steps 'en,fr'                                 \
	--encoder_only false                                 \
	--emb_dim 1024                                       \
	--n_layers 6                                         \
	--n_heads 8                                          \
	--dropout 0.1                                        \
	--attention_dropout 0.1                              \
	--gelu_activation true                               \
	--tokens_per_batch 3000                              \
	--optimizer adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00028 \
	--epoch_size 200000                                  \
	--max_epoch 100                                      \
	--eval_bleu true                                     \
	--word_mass 0.5                                      \
	--min_len 5                                        \
	--exp_id mass_eval   \
	--reload_model $MODEL,$MODEL  \
	--eval_only true \
	--eval_bleu true

