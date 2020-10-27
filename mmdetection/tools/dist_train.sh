#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-17500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/train.py $CONFIG --work-dir /lustre/S/wangyu/city_model/car_only --launcher pytorch ${@:3}
