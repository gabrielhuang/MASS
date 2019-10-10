#!/bin/bash

export RANK=0
export WORLD_SIZE=1
export NGPU=1
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=6666
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MAX_LEN=100

#
# Read arguments
#
POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"
case $key in
  --src)
    SRC="$2"; shift 2;;  # asr1, asr2, asr3
  --tgt)
    TGT="$2"; shift 2;;  # cap
  --max_len)
    MAX_LEN="$2"; shift 2;;  # 100 default
  --gpu)
    GPU="$2"; shift 2;;  # 100 default
  *)
  POSITIONAL+=("$1")
  shift
  ;;
esac
done
set -- "${POSITIONAL[@]}"

echo "Passing those other arguments to train.py \"$@\""

if [ -z "$GPU" ]
then
	echo "Running on default GPU"
	SCRIPT="python train.py"
else
	echo "Running on GPU number $GPU"
	SCRIPT="python -m torch.distributed.launch --nnodes=1 --nproc_per_node=$NGPU train.py"
	export CUDA_VISIBLE_DEVICES=$GPU
fi

python train.py                                      \
	--exp_name unsupMT_youtuberecipe \
	--data_path ./data/processed/$SRC-$TGT/                  \
	--lgs "$SRC-$TGT"                                        \
	--mass_steps "$SRC,$TGT"                                 \
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
	--save_periodic 10                                      \
	--eval_bleu true                                     \
	--word_mass 0.5                                      \
	--min_len 5                                        \
	--exp_id "$SRC_$TGT_maxlen$MAXLEN"   \
	$@

# pass other arguments
