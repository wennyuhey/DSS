#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-17500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/da_train.py $CONFIG --resume-from /lustre/S/wangyu/cityscapes_model/epoch_9.pth --work-dir /lustre/S/wangyu/dis_train/ --launcher pytorch ${@:3}
