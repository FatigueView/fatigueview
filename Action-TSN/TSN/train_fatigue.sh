#!/usr/bin bash

python tools/train.py configs/recognition/tsn/tsn_r50_320p_1x1x3_100e_fatigue_rgb.py \
        --work-dir work_dirs/tsn_r50_320p_1x1x3_100e_fatigue_rgb \
        --validate --seed 0 --deterministic \
        --gpu-ids 2