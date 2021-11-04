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
    heights = [ldmk[2][1]-ldmk[6][1],ldmk[1][1]-ldmk[7][1],ldmk[3][1]-ldmk[5][1]]
    return np.mean(np.abs(heights))



def get_fea_label(img_info):
    eye_center = [-1,-1]
    if 'ldmk' in img_info and img_info['ldmk'] is not None and  len(img_info['ldmk']) > 4:
        ldmk = np.array(img_info['ldmk'])
        eye_ldmk = ldmk[36:47]
        x,y = np.mean(eye_ldmk,axis= 0)
        eye_center = [x,y]
    return eye_center



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


def is_stretch(stretch_list,left_index,right_index):
    # 1 : stretch 0: normal -1 : ignore

    max_union = -1
    frame_len = right_index - left_index
    for stretch in stretch_list:

        stretch_len = abs(stretch[1] - stretch[0])
        temp_left = max(left_index,stretch[0])
        temp_right = min(right_index,stretch[1])


        if [temp_left,temp_right] in [stretch,[left_index,right_index]]:
            return 1

        union =  (temp_right - temp_left) /( min(stretch_len,frame_len) + 0.1)
        max_union = max(max_union,union)


    if max_union < 0.1:
        return 0

    return -1


def min_is_nodding(x_list,threshold):
    if sum(v!=-1 for v in x_list) == 0 :
        return 0
    if x_list[len(x_list)//2] == -1:
        return 0
    _x = x_list[len(x_list)//2]
    x_list = [v for v in x_list if v != -1]

    if max(x_list) - min(x_list) > threshold and _x in [max(x_list) ,min(x_list)]:
        return 1


def is_nodding(x_list,half_frame_len = 8,threshold = 4):
    ans = []
    for i in range(half_frame_len, len(x_list) - half_frame_len):
        ans.append(min_is_nodding(x_list[i-half_frame_len:i+half_frame_len],threshold))
    return sum(ans)

def get_batch_data(video_list,suffix,dst_dir,time_len = 10):
    random.shuffle(video_list)

    half_frame_len = time_len*25//2
    half_frame_len = 40


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

        stretch_path = video_path.replace(os.path.basename(video_path), 'nodding.json')
        if not os.path.exists(stretch_path):
            continue

        with open(stretch_path, 'r') as f:
            stretch_list = json.load(f)

        try:
            stretch_list = stretch_list[os.path.basename(video_path).replace(video_suffix,'')]
        except:
            continue

        with open(json_path, 'r') as f:
            big_json = f.readlines()


        skeleton_list = []


        for json_info in big_json:
            try:
                json_info = json.loads(json_info.strip())
            except:
                continue

            skeleton_list.append(get_fea_label(json_info))







        for stretch in stretch_list:
            stretch = list(map(int,stretch))
            temp_eye_list = skeleton_list[stretch[0]:stretch[1]]
            temp_eye_list.append(1)
            frame_id = sum(stretch)//2
            npy_name = '_'.join(video_path.split(os.sep)[-4:]).replace(video_suffix,'')
            npy_name = '{}__{}__{}.json'.format(1, npy_name, frame_id)
            with open(os.path.join(dst_dir, npy_name), 'w') as f:
                json.dump(temp_eye_list, f)

        temp_count = 0
        for i in range(10000*len(stretch_list)):
            frame_id = int(random.random()*len(skeleton_list))
            if i < half_frame_len or i >= len(skeleton_list) - (half_frame_len+1):
                continue

            temp_stretch = is_stretch(stretch_list,frame_id-half_frame_len,frame_id+half_frame_len)

            if temp_stretch != 0:
                continue

            temp_count += 1
            temp_eye_list = skeleton_list[frame_id-half_frame_len:frame_id+half_frame_len]
            temp_eye_list.append(0)
            npy_name = '_'.join(video_path.split(os.sep)[-4:]).replace(video_suffix,'')
            npy_name = '{}__{}__{}.json'.format(0, npy_name, frame_id)
            with open(os.path.join(dst_dir, npy_name), 'w') as f:
                json.dump(temp_eye_list, f)

            if temp_count > len(stretch_list):
                break



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
                    'test':'/data/weiyu.li/DMSData/FatigueView/test_video'

    }


    src_dir_dict = {'test':'/data/weiyu.li/DMSData/FatigueView/test_video'

    }


    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']

    # camera_list = ['rgb_left','rgb_left_up','rgb_up']


    src_dir_dict = {'sanbao_test':'/data/weiyu.li/DMSData/FatigueView/sanbao_test_video',

    }

    camera_list = ['ir_down']



    data_type = 'train'
    camera_id = 0

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
            #     video_list = [v for v in video_list if 'fengchunshen' not in v and 'panbijia' not in v]
            #
            #
            # if data_type == 'train':
            #     video_list = [v for v in video_list if 'zhaoxinmei' not in v]


            all_num = 60000
            running_num = 32
            running_num = min(running_num,len(video_list))
            batch_size = all_num//running_num
            split_videos = split(video_list, running_num)


            process_list = []
            for i in range(running_num):
                temp_p = Process(target=get_batch_data,args=(split_videos[i],suffix,dst_dir,))
                process_list.append(temp_p)

            for temp_p in process_list:
                temp_p.start()

            for temp_p in process_list:
                temp_p.join()


            print('END')
