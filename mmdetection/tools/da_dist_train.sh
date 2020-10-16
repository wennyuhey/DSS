#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
#RESUMEPATH=$3
PORT=${PORT:-17500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/da_train.py $CONFIG --work-dir /lustre/S/wangyu/da_reppoint/dist/ --resume-from /lustre/S/wangyu/PretrainedModels/epoch_13.pth --launcher pytorch ${@:3}
