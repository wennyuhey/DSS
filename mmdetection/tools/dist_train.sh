#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
PORT=${PORT:-17500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
    $(dirname "$0")/da_train.py $CONFIG --resume-from /workspace/cityscapes/faster_rcnn_r50_fpn_1x_cityscapes_20200502-829424c0.pth --work-dir /workspace/reppoint-wy/aux_faster --launcher pytorch ${@:3}
