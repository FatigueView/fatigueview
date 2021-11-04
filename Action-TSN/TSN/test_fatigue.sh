#!/usr/bin bash

python tools/test.py configs/recognition/tsn/tsn_test_fatigue.py \
        work_dirs/tsn_r50_320p_1x1x3_100e_fatigue_rgb/latest.pth \
        --gpus 2 --eval mean_class_accuracy