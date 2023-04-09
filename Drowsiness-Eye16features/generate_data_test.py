# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/11/5 11:37 AM
=================================================='''

import sys

sys.path.append('./')
sys.path.insert(0, '/data/zhenyu.yang/modules')

import cv2
import json
import numpy as np
import random
import copy
from multiprocessing import Process
from multiprocessing import Pool

import os


def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if
            file.endswith(suffix)]


def get_ear(ldmk):
    eps = 1e-5
    get_distance = lambda x, y: ((x[0] - y[0]) ** 2 + (
            x[1] - y[1]) ** 2 + eps) ** 0.5

    w = get_distance(ldmk[0], ldmk[4])
    h = get_distance(ldmk[2], ldmk[6])

    ear = h / w
    ear = min(ear, 0.7)

    return ear


def get_ear_height(ldmk):
    heights = [ldmk[2][1] - ldmk[6][1], ldmk[1][1] - ldmk[7][1],
               ldmk[3][1] - ldmk[5][1]]
    return np.mean(np.abs(heights))


def get_fea_label(img_info):
    heights = []
    if 'left_eye_ldmk' in img_info and img_info[
        'left_eye_ldmk'] is not None and len(img_info['left_eye_ldmk']) > 4:
        heights.append(get_ear_height(img_info['left_eye_ldmk']))
    else:
        heights.append(-1)
    if 'right_eye_ldmk' in img_info and img_info[
        'right_eye_ldmk'] is not None and len(img_info['right_eye_ldmk']) > 4:
        heights.append(get_ear_height(img_info['right_eye_ldmk']))
    else:
        heights.append(-1)

    if heights[0] == -1 and heights[1] == -1:
        return -1

    if heights[0] == -1:
        return heights[1]

    if heights[1] == -1:
        return heights[0]

    return np.mean(heights)


def cal_alpha(d, pre_d, pre_b, median_d):
    alpha = [0.4, 15, 0.5, 2, 0.7]
    ans = alpha[0]
    ans *= np.exp(-alpha[1] * (d - pre_d) ** 2)
    ans *= np.exp(-alpha[2] * max((d - pre_b), 0))
    ans *= np.exp(-alpha[3] * max((pre_b - d), 0))
    ans *= (0, 1)[d >= median_d * alpha[4]]

    return ans


def finetune_eye_heights(heights):
    median_d = np.median([v for v in heights if v != -1])

    pre_b = median_d
    pre_d = median_d

    finetuned_heights = []
    for height in heights:
        if height == -1:
            finetuned_heights.append(height)
            continue

        alpha = cal_alpha(height, pre_d, pre_b, median_d)

        pre_b = (1 - alpha) * pre_b + height * alpha

        finetuned_heights.append(height / (pre_b + 0.00001))

    return finetuned_heights


def get_blink_info(blink_list, left_index, right_index):
    now_blink = []
    for blink in blink_list:
        if blink[1] > left_index and blink[0] < right_index:
            now_blink.append(blink[1] - blink[0] + 1)

    if len(now_blink) == 0:
        now_blink = [-1]

    average_duration = np.mean(now_blink)
    micro_sleep = sum(v >= 125 for v in now_blink)

    return average_duration, micro_sleep


def list2bin(data_list, min_data, max_data, bin_num=10):
    sep = (max_data - min_data) // bin_num
    bin_list = [0 for i in range(bin_num + 1)]
    if sep ==0:
        return bin_list
    for data in data_list:
        if data == -1:
            bin_list[-1] += 1
        bin = (data - min_data) / sep
        bin_list[int(bin)] += 1

    bin_list = [v / len(data_list) for v in bin_list]

    return bin_list


def list2num(slice_list):
    num_list = []
    for slice in slice_list:
        num_list.extend(list(range(slice[0], slice[1] + 1)))
    return num_list

def get_video_slice_dict(max_len,frame_len = 450000 ,setp_frame = 450000):
    video_slice_dict = {}
    for i in range(max_len):
        video_slice = i//(frame_len + setp_frame)
        video_frame = i%(frame_len + setp_frame)
        if video_frame < frame_len:
            video_slice_dict[i] = video_slice

    return video_slice_dict


def get_batch_data(video_dir_list, suffix, batch_size, start_id, dst_dir,
                   stage='train', time_len=30, random_ratio=0.5):
    camera_list = ['ir_down', 'ir_front', 'ir_left', 'ir_left_up', 'ir_up',
                   'rgb_down', 'rgb_front', 'rgb_left', 'rgb_left_up',
                   'rgb_up']

    half_frame_len = time_len * 25 // 2



    for video_dir in video_dir_list:
        data_id = -1

        all_heights = {}

        height_list_len = 1000000000
        for camera in camera_list:

            json_path = os.path.join(video_dir, camera + suffix)
            with open(json_path, 'r') as f:
                big_json = f.readlines()

            height_list = []

            for json_info in big_json:
                try:
                    json_info = json.loads(json_info.strip())
                except:
                    continue

                height_list.append(get_fea_label(json_info))

            height_list = finetune_eye_heights(height_list)

            height_list_len = min(len(height_list), height_list_len)

            if len([v for v in height_list if v != -1]) == 0:
                max_height = -1
                min_height = -1
            else:
                max_height = max(v for v in height_list if v != -1)
                min_height = min(v for v in height_list if v != -1)

            all_heights[camera] = {'height_list': height_list,
                                   'max_height': max_height,
                                   'min_height': min_height}

        blink_path = os.path.join(video_dir, 'blink.json')
        if not os.path.exists(blink_path):
            continue

        with open(blink_path, 'r') as f:
            blink_info = json.load(f)
            blink_all = blink_info['blink_all']
            blink_close = blink_info['blink_close']
            blink_open = blink_info['blink_open']
            close_list = blink_info['close']

        awake_path = os.path.join(video_dir, 'awake.json')
        poor_acting_path = os.path.join(video_dir, 'poor_acting.json')
        fatigue_path = os.path.join(video_dir, 'fatigue.json')

        awake_list = []
        poor_action_list = []
        fatigue_list = []

        if stage == 'train':
            if os.path.exists(awake_path):
                with open(awake_path, 'r') as f:
                    temp_awake_list = json.load(f)
                awake_list = list2num(temp_awake_list)

            if os.path.exists(poor_acting_path):
                with open(poor_acting_path, 'r') as f:
                    temp_poor_action_list = json.load(f)
                poor_action_list = list2num(temp_poor_action_list)

            fatigue_list = [i for i in range(height_list_len) if
                            i not in awake_list and i not in poor_action_list]

        if stage == 'test':
            if os.path.exists(fatigue_path):
                with open(fatigue_path, 'r') as f:
                    temp_fatigue_list = json.load(f)
                awake_list = list2num(temp_fatigue_list[0])
                fatigue_list = list2num(temp_fatigue_list[1]) + list2num(
                    temp_fatigue_list[2]) + list2num(temp_fatigue_list[3])+ list2num(temp_fatigue_list[4])
            else:
                fatigue_list = []
                awake_list = list(range(height_list_len))


        array_len = max(max(awake_list+[-1]),max(fatigue_list+[-1]),height_list_len) + 1

        np_awake = np.array([0 for _ in range(array_len)])
        np_awake[awake_list] = 1

        np_fatigue = np.array([0 for _ in range(array_len)])
        np_fatigue[fatigue_list] = 1

        video_slice_dict = get_video_slice_dict(array_len)

        for frame_id in range(height_list_len):

            if frame_id not in video_slice_dict:
                continue

            if frame_id % 5 != 0:
                continue

            if frame_id < half_frame_len or frame_id >= height_list_len - (
                    half_frame_len + 1):
                continue

            i = frame_id
            average_blink_duration, _ = get_blink_info(blink_all,
                                                       i - half_frame_len,
                                                       i + half_frame_len)
            average_closed_duration, micro_sleep = get_blink_info(close_list,
                                                                  i - half_frame_len,
                                                                  i + half_frame_len)
            average_close_duration, _ = get_blink_info(blink_close,
                                                       i - half_frame_len,
                                                       i + half_frame_len)
            average_open_duration, _ = get_blink_info(blink_open,
                                                      i - half_frame_len,
                                                      i + half_frame_len)

            awake_time = 0
            fatigue_time = 0

            awake_time = np.sum(
                np_awake[i - half_frame_len:i + half_frame_len])
            fatigue_time = np.sum(
                np_fatigue[i - half_frame_len:i + half_frame_len])


            # for i in range(frame_id - half_frame_len, frame_id + half_frame_len):
            #     if i in awake_list:
            #         awake_time += 1
            #     if i in fatigue_list:
            #         fatigue_time += 1

            # if awake_time < half_frame_len * 1.2 and fatigue_time < half_frame_len * 1.2:
            #     continue

            temp_label = 2
            if frame_id in awake_list:
                temp_label = 0
            if frame_id in fatigue_list:
                temp_label = 1

            label = -1
            if awake_time >= half_frame_len * 1.2:
                label = 0
            elif fatigue_time >= half_frame_len * 1.2:
                label = 1



            for camera, height_info in all_heights.items():
                height_list = height_info['height_list']
                max_height = height_info['max_height']
                min_height = height_info['min_height']

                temp_heights = height_list[
                               frame_id - half_frame_len:frame_id + half_frame_len]

                bin_list = list2bin(temp_heights, min_height, max_height)

                data_id += 1

                temp_feas = bin_list + [average_blink_duration,
                                        average_closed_duration, micro_sleep,
                                        average_close_duration,
                                        average_open_duration,
                                        label, temp_label]

                temp_data = list(map(str, temp_feas))
                temp_data = ' '.join(temp_data) + '\n'

                temp_data_name = '_'.join(video_dir.split(os.sep)[-4:])
                temp_data_name = '{}__{}__{}.txt'.format(temp_data_name, camera,video_slice_dict[frame_id])

                with open(os.path.join(dst_dir, temp_data_name), 'a') as f:
                    f.write(temp_data)


def split(input, num=60):
    random.shuffle(input)

    ans = []
    sep = len(input) // num
    for i in range(num - 1):
        ans.append(input[i * sep:(i + 1) * sep])

    ans.append(input[(num - 1) * sep:])

    return ans


if __name__ == '__main__':

    version = 'v0.1'
    suffix = '_{}.json'.format(version)
    time_len = 30
    random_ratio = 0.05

    all_num = 60000
    running_num = 32

    dst_dir = './data_new/test'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    src_dir = '/data/weiyu.li/DMSData/FatigueView/test_video'
    # src_dir = '/mnt/data-2/data/zhenyu.yang/DMSData_mount/FatigueView/test_video'


    need_precit_persons = None

    if need_precit_persons is None:
        video_list = getFiles(src_dir, '.json')
    else:
        video_list = []
        for person in need_precit_persons:
            video_list += getFiles(os.path.join(src_dir, person),
                                   'json')

    video_list = ['/' + os.path.join(*v.split(os.sep)[:-1]) for v in
                  video_list]
    video_list = list(set(video_list))


    running_num = min(running_num, len(video_list))
    batch_size = all_num // running_num

    split_videos = split(video_list, running_num)

    process_list = []
    for i in range(running_num):
        temp_p = Process(target=get_batch_data, args=(
            split_videos[i], suffix, batch_size, batch_size * i, dst_dir,
            'test',))
        process_list.append(temp_p)

    for temp_p in process_list:
        temp_p.start()

    for temp_p in process_list:
        temp_p.join()

    print('END')

