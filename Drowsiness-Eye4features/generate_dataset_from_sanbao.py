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


def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix)]



def get_ear(ldmk):
    eps = 1e-5
    get_distance = lambda x,y:((x[0]-y[0])**2 + (x[1]-y[1])**2 + eps)**0.5

    w = get_distance(ldmk[0],ldmk[4])
    h = get_distance(ldmk[2],ldmk[6])

    ear = h/w
    ear = min(ear,0.7)

    return ear

def get_ear_height(ldmk):
    return abs(ldmk[2][1]-ldmk[6][1])



def get_fea_label(img_info):


    heights = []
    if 'left_eye_ldmk' in img_info and img_info['left_eye_ldmk'] is not None and  len(img_info['left_eye_ldmk']) > 4:
        heights.append(get_ear_height(img_info['left_eye_ldmk']))
    else:
        heights.append(-1)
    if 'right_eye_ldmk' in img_info and img_info['right_eye_ldmk'] is not None and len(img_info['right_eye_ldmk']) > 4:
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


def get_blink_info(blink_list,left_index,right_index):
    now_blink = []
    for blink in blink_list:
        if blink[1] > left_index and blink[0] < right_index:
            now_blink.append(blink[1] - blink[0] + 1)

    blink_ratio = len(now_blink)/(right_index + left_index + 0.01)

    if len(now_blink) == 0:
        now_blink = [-1]

    blind_duration = np.mean(now_blink)/25
    return [blink_ratio,blind_duration]



def get_perclose(height_list):
    max_height = max(height_list)
    preclose_list = [1 - v/max_height for v in  height_list]
    preclose_50 = sum(v > 0.5 for v in preclose_list)
    preclose_70 = sum(v > 0.7 for v in preclose_list)
    preclose_90 = sum(v > 0.9 for v in preclose_list)
    return [preclose_50,preclose_70,preclose_90]



def get_eye_movement(height_list):
    height_change = [abs(height_list[i+1] - height_list[i]) for i in range(len(height_list)-1)]
    return sum(v>1 for v in height_change) / len(height_list)



def list2num(slice_list):
    num_list = []
    for slice in slice_list:
        num_list.extend(list(range(slice[0], slice[1] + 1)))
    return num_list


def get_batch_data(video_list,suffix,batch_size,start_id,dst_dir,stage= 'train',random_ratio = 0.05):
    random.shuffle(video_list)

    data_id = -1

    while True:
        if len(video_list) == 0:
            break
        video_path = video_list.pop()

        video_suffix = '.mp4'
        if video_path.endswith('.mp4'):
            video_suffix = '.mp4'
        elif video_path.endswith('.avi'):
            video_suffix = '.avi'


        json_path = video_path.replace(video_suffix, suffix)

        if not os.path.exists(json_path):
            continue


        with open(json_path, 'r') as f:
            big_json = f.readlines()


        height_list = []


        for json_info in big_json:
            try:
                json_info = json.loads(json_info.strip())
            except:
                continue

            height_list.append(get_fea_label(json_info))


        blink_path = video_path.replace(os.path.basename(video_path), 'blink.json')
        if not os.path.exists(blink_path):
            continue

        with open(blink_path, 'r') as f:
            blink_info = json.load(f)
            try:
                blink_info = blink_info[
                    os.path.basename(video_path).replace(video_suffix, '')]
            except:
                continue


            blink_list = blink_info['blink_all']

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

            fatigue_list = [i for i in range(len(height_list)) if i not in awake_list and i not in poor_action_list]


        if 'test' in stage  :
            if os.path.exists(fatigue_path):
                with open(fatigue_path, 'r') as f:
                    temp_fatigue_list = json.load(f)
                    try:
                        temp_fatigue_list = temp_fatigue_list[
                            os.path.basename(video_path).replace(video_suffix, '')]
                    except:
                        continue

                awake_list = list2num(temp_fatigue_list[0])
                fatigue_list = list2num(temp_fatigue_list[1]) + list2num(temp_fatigue_list[2]) + list2num(temp_fatigue_list[3])
            else:
                fatigue_list = []
                awake_list = list(range(len(height_list)))


        fatigue_list = set(fatigue_list)
        awake_list = set(awake_list)


        for frame_id in range(len(height_list)):

            if random.random() > random_ratio:
                continue

            if frame_id < 50 or frame_id >= len(height_list) - 51:
                continue


            awake_time = 0
            fatigue_time = 0
            for i in range(frame_id-50,frame_id+50):
                if i in awake_list:
                    awake_time += 1
                if i in fatigue_list:
                    fatigue_time += 1

            if awake_time < 70 and fatigue_time < 70:
                continue

            if awake_time >= 70:
                label = 0
            else:
                label = 1

            temp_heights = height_list[frame_id - 50:frame_id + 50]
            if sum(v != -1 for v in temp_heights) < 80:
                continue


            preclose = get_perclose(temp_heights)
            movement = get_eye_movement(temp_heights)

            temp_feas = get_blink_info(blink_list,frame_id-50,frame_id+50)
            temp_feas = temp_feas + preclose + [movement]


            data_id += 1

            temp_feas = temp_feas + [label]

            temp_data = list(map(str, temp_feas))
            temp_data = ' '.join(temp_data)

            temp_data_name = '_'.join(video_path.split(os.sep)[-4:]).replace(video_suffix,'')
            temp_data_name = '{}__{}.txt'.format(temp_data_name, frame_id)

            with open(os.path.join(dst_dir,temp_data_name),'w') as f:
                f.write(temp_data)


            if data_id >= batch_size:
                return True


        if len(video_list) == 0:
            break

    return False


def split(input,num=60):
    random.shuffle(input)

    ans = []
    sep = len(input) //num
    for i in range(num-1):
        ans.append(input[i*sep:(i+1)*sep])

    ans.append(input[(num-1)*sep:])

    return ans



if __name__ == '__main__':

    version = 'v0.1'
    suffix = '_{}.json'.format(version)


    src_dir_dict = {'train':'/data/weiyu.li/DMSData/FatigueView/raw_video',
                    'test':'/data/weiyu.li/DMSData/FatigueView/test_video'}

    src_dir_dict = {'test':'/data/weiyu.li/DMSData/FatigueView/test_video'}

    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']



    src_dir_dict = {'sanbao_test':'/data/weiyu.li/DMSData/FatigueView/sanbao_test_video',

    }

    camera_list = ['ir_down']




    data_type = 'train'
    camera_id = 0
    random_ratio = 0.5

    for data_type in src_dir_dict.keys():
        for camera_id in range(len(camera_list)):

            src_dir = src_dir_dict[data_type]

            camera_type = camera_list[camera_id]

            dst_dir = './data/{}/{}'.format(data_type,camera_type)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            video_list = getFiles(src_dir, '.mp4')
            video_list += getFiles(src_dir, '.avi')


            # if data_type == 'test':
            #     video_list = [v for v in video_list if 'fengchunshen' not in v and 'panbijia' not in v and 'basic' not in v]
            #
            #
            # if data_type == 'train':
            #     video_list = [v for v in video_list if 'zhaoxinmei' not in v and 'basic' not in v]


            all_num = 60000
            running_num = 32
            running_num = min(running_num,len(video_list))
            batch_size = all_num//running_num
            split_videos = split(video_list, running_num)

            get_batch_data(split_videos[0],suffix,batch_size,batch_size*0,dst_dir,data_type,random_ratio)

            process_list = []
            for i in range(running_num):
                temp_p = Process(target=get_batch_data,args=(split_videos[i],suffix,batch_size,batch_size*i,dst_dir,data_type,random_ratio,))
                process_list.append(temp_p)

            for temp_p in process_list:
                temp_p.start()

            for temp_p in process_list:
                temp_p.join()


            print('END')
