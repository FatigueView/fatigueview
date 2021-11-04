#!/usr/bin bash

python3.6 tools/train.py configs/recognition/tsn/tsn_r50_dense_1x1x8_100e_kinetics400_rgb_4class.py \
        --work-dir work_dir/tsn_r50_dense_1x1x8_100e_kinetics400_rgb_4class_more_data_sequence_sample \
        --validate --seed 0 --deterministic \
        --gpu-ids 2
