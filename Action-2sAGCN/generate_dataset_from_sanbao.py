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



def get_fea_label(img_info):
    skeleton = np.zeros((17,2)) - 1
    if 'skeleton' in img_info and img_info['skeleton'] is not None and  len(img_info['skeleton']) > 4:
        skeleton = np.array(img_info['skeleton'])
    return skeleton



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


    if max_union < 0.2:
        return 0

    return -1



def get_batch_data(video_list,suffix,dst_dir,time_len = 10):
    random.shuffle(video_list)

    half_frame_len = time_len*25//2


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


        skeleton_list = []


        for json_info in big_json:
            try:
                json_info = json.loads(json_info.strip())
            except:
                continue

            skeleton_list.append(get_fea_label(json_info))

        stretch_path = video_path.replace(os.path.basename(video_path), 'stretch.json')
        if not os.path.exists(stretch_path):
            continue
        print(stretch_path)

        with open(stretch_path, 'r') as f:
            stretch_list = json.load(f)
        try:
            stretch_list = stretch_list[os.path.basename(video_path).replace(video_suffix,'')]
        except:
            continue


        stretch_index_list = []
        normal_index_list = []

        for i in range(0,len(skeleton_list),40):

            if i < half_frame_len or i >= len(skeleton_list) - (half_frame_len+1):
                continue

            temp_stretch = is_stretch(stretch_list,i-half_frame_len,i+half_frame_len)
            if temp_stretch == 1:
                stretch_index_list.append(i)
                temp_skeleton_list = skeleton_list[i-half_frame_len:i+half_frame_len]
                temp_skeleton_list = np.stack(temp_skeleton_list,axis=0)

                npy_name = '_'.join(video_path.split(os.sep)[-4:]).replace(video_suffix,'')
                npy_name = '{}__{}__{}.npy'.format(1,npy_name,i)
                np.save(os.path.join(dst_dir,npy_name),temp_skeleton_list)


            if temp_stretch == 0:
                normal_index_list.append(i)

        if len(stretch_index_list) == 0:
            continue

        random.shuffle(normal_index_list)
        normal_index_list = normal_index_list[:min(len(normal_index_list),len(stretch_index_list))]

        for i in normal_index_list:
            temp_skeleton_list = skeleton_list[i - half_frame_len:i + half_frame_len]
            temp_skeleton_list = np.stack(temp_skeleton_list, axis=0)

            npy_name = '_'.join(video_path.split(os.sep)[-4:]).replace(video_suffix,'')
            npy_name = '{}__{}__{}.npy'.format(0, npy_name, i)
            np.save(os.path.join(dst_dir, npy_name), temp_skeleton_list)



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

    camera_list = ['rgb_left','rgb_left_up','rgb_up']


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
            running_num = 1
            batch_size = all_num//running_num
            split_videos = split(video_list, running_num)

            get_batch_data(split_videos[0],suffix,dst_dir)

            # process_list = []
            # for i in range(running_num):
            #     temp_p = Process(target=get_batch_data,args=(split_videos[i],suffix,dst_dir,))
            #     process_list.append(temp_p)
            #
            # for temp_p in process_list:
            #     temp_p.start()
            #
            # for temp_p in process_list:
            #     temp_p.join()
            #
            #
            # print('END')
