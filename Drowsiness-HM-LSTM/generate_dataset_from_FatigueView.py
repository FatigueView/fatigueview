# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/11/5 11:37 AM
=================================================='''

import sys
sys.path.append('./')
sys.path.insert(0,'/data/zhenyu.yang/modules')


import cv2
import json
import numpy as np
import random
import copy
from multiprocessing import Process
import os
import time


def getFiles(path, suffix,prefix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix) and file.startswith(prefix)]



def get_ear(ldmk):
    eps = 1e-5
    get_distance = lambda x,y:((x[0]-y[0])**2 + (x[1]-y[1])**2 + eps)**0.5

    w = get_distance(ldmk[0],ldmk[4])
    h = get_distance(ldmk[2],ldmk[6])

    ear = h/w
    ear = min(ear,0.7)

    return ear




def get_fea_label(img_info):


    heights = []
    if 'left_eye_ldmk' in img_info and img_info['left_eye_ldmk'] is not None and  len(img_info['left_eye_ldmk']) > 4:
        heights.append(get_ear(img_info['left_eye_ldmk']))
    else:
        heights.append(-1)
    if 'right_eye_ldmk' in img_info and img_info['right_eye_ldmk'] is not None and len(img_info['right_eye_ldmk']) > 4:
        heights.append(get_ear(img_info['right_eye_ldmk']))
    else:
        heights.append(-1)


    if heights[0] == -1 and heights[1] == -1:
        return -1

    if heights[0] == -1:
        return heights[1]

    if heights[1] == -1:
        return heights[0]

    return np.mean(heights)



def get_blink_info(blink_list, left_index, right_index,ear_list):
    eps = 1e-5
    blink_fea_list = []
    for blink in blink_list:
        if blink[1] <= right_index and blink[0] >= left_index:

            temp_ear = ear_list[blink[0]-left_index:blink[1]-left_index]
            duration = blink[1] - blink[0]  + 1
            amp = (temp_ear[0] + temp_ear[-1]- 2*min(temp_ear))/2
            velocity = (temp_ear[-1] - min(temp_ear))/(len(temp_ear) - 1 - np.argmin(temp_ear) + eps)
            frequence = len(blink_fea_list)/(blink[1]-left_index + eps)

            blink_fea_list.append([duration,amp,velocity,frequence])

    if len(blink_fea_list) == 0:
        blink_fea_list = [[0,0,0,0]]


    return blink_fea_list




def list2num(slice_list):
    num_list = []
    for slice in slice_list:
        num_list.extend(list(range(slice[0], slice[1] + 1)))
    return num_list


def get_batch_data(video_list,suffix,batch_size,start_id,dst_dir,stage= 'train',time_len = 30,random_ratio = 0.5):
    random.shuffle(video_list)
    frame_len = time_len*25
    half_frame_len = time_len*25//2

    data_id = -1
    while True:
        print('start')
        tic = time.time()
        if len(video_list) == 0:
            break


        video_path = video_list.pop()

        json_path = video_path.replace('.mp4', suffix)

        blink_path = video_path.replace(os.path.basename(video_path),'blink.json')

        if not os.path.exists(blink_path) or not os.path.exists(json_path) :
            continue

        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            big_json = f.readlines()


        ear_list = []


        for json_info in big_json:
            try:
                json_info = json.loads(json_info.strip())
            except:
                continue

            ear_list.append(get_fea_label(json_info))


        if len([v for v in ear_list if v != -1]) == 0:
            continue




        with open(blink_path, 'r') as f:
            blink_info = json.load(f)
            blink_all = blink_info['blink_all']
            blink_close = blink_info['blink_close']
            blink_open = blink_info['blink_open']
            close_list  = blink_info['close']

        blink_all.sort(key=lambda x:x[0])

        awake_path = video_path.replace(os.path.basename(video_path), 'awake.json')
        poor_acting_path = video_path.replace(os.path.basename(video_path), 'poor_acting.json')
        fatigue_path = video_path.replace(os.path.basename(video_path),'fatigue.json')

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

            fatigue_list = [i for i in range(len(ear_list)) if i not in awake_list and i not in poor_action_list]


        if stage == 'test'  or 'test' in stage:
            if os.path.exists(fatigue_path):
                with open(fatigue_path, 'r') as f:
                    temp_fatigue_list = json.load(f)
                awake_list = list2num(temp_fatigue_list[0])
                fatigue_list = list2num(temp_fatigue_list[1]) + list2num(temp_fatigue_list[2]) + list2num(temp_fatigue_list[3])
            else:
                fatigue_list = []
                awake_list = list(range(len(ear_list)))

        awake_list = set(awake_list)
        fatigue_list = set(fatigue_list)


        for i in range(len(ear_list)):

            if random.random() > random_ratio:
                continue

            if i < frame_len:
                continue


            awake_time = 0
            fatigue_time = 0
            for i in range(i-frame_len,i):
                if i in awake_list:
                    awake_time += 1
                if i in fatigue_list:
                    fatigue_time += 1


            label = -1
            if awake_time >= half_frame_len*1.2 :
                label = 0
            elif fatigue_time >= half_frame_len*1.2:
                label = 1

            temp_heights = ear_list[i - frame_len:i]
            if sum(v != -1 for v in temp_heights) < half_frame_len*1.2:
                continue


            blink_list_info = get_blink_info(blink_all,i - frame_len,i,temp_heights)

            temp_data = {'blink_fea':blink_list_info,'label':label}


            temp_data_name = '_'.join(video_path.split(os.sep)[-4:]).replace('.mp4', '')
            temp_data_name = '{}__{}.json'.format(temp_data_name, i)



            with open(os.path.join(dst_dir, temp_data_name), 'w') as f:
                json.dump(temp_data,f)

            if data_id >= batch_size:
                return True



        if len(video_list) == 0:
            break

    return False




def split(input,num=60):
    random.shuffle(input)

    ans = []
    sep = len(input) //num
    sep = max(sep,1)
    for i in range(num-1):
        if i*sep >  len(input):
            return ans
        ans.append(input[i*sep:(i+1)*sep])

    ans.append(input[(num-1)*sep:])

    return ans



if __name__ == '__main__':

    version = 'v0.1'
    suffix = '_{}.json'.format(version)
    time_len = 30
    random_ratio = 0.05

    all_num = 60000
    running_num = 32


    src_dir_dict = {'train':'/data/weiyu.li/DMSData/FatigueView/raw_video',
                    'test':'/data/weiyu.li/DMSData/FatigueView/test_video'
    }

    # src_dir_dict = {'test':'/data/weiyu.li/DMSData/FatigueView/test_video'
    # }

    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']


    data_type = 'train'
    camera_id = 0

    for data_type in src_dir_dict.keys():

        for camera_id in range(len(camera_list)):

            src_dir = src_dir_dict[data_type]

            camera_type = camera_list[camera_id]

            dst_dir = './data/{}/{}'.format(data_type,camera_type)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            video_list = getFiles(src_dir, '.mp4', camera_type)

            if data_type == 'test':
                video_list = [v for v in video_list if 'fengchunshen' not in v and 'panbijia' not in v and 'basic' not in v]
                random_ratio = 0.01
                # all_num = 20000

            if data_type == 'train':
                video_list = [v for v in video_list if 'zhaoxinmei' not in v and 'basic' not in v]

            running_num = min(running_num,len(video_list))
            batch_size = all_num//running_num
            split_videos = split(video_list, running_num)


            process_list = []
            for i in range(running_num):
                temp_p = Process(target=get_batch_data,args=(split_videos[i],suffix,batch_size,batch_size*i,dst_dir,data_type,))
                process_list.append(temp_p)

            for temp_p in process_list:
                temp_p.start()

            for temp_p in process_list:
                temp_p.join()


            print('END')
