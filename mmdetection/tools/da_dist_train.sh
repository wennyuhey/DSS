#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-17600}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/da_train.py $CONFIG --work-dir /lustre/S/wangyu/checkpoint/cliipart/da//r101/aux/lr001 --launcher pytorch ${@:3}

