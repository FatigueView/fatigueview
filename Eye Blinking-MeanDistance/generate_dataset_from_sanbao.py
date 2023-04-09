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

def get_fea_label(img_info):

    label_dict = {'close':0,
                  'open':1,
                  'occluded':None,
                  'narrow':None,
                  'lookdown':None,
                  'unknown':None}

    ldmks = []
    if 'left_eye_ldmk' in img_info and img_info['left_eye_ldmk'] is not None and  len(img_info['left_eye_ldmk']) > 4:
        ldmks.append([get_ear(img_info['left_eye_ldmk']),label_dict[img_info['left_status']]])
    else:
        ldmks.append([-1,None])
    if 'right_eye_ldmk' in img_info and img_info['right_eye_ldmk'] is not None and len(img_info['right_eye_ldmk']) > 4:
        ldmks.append([get_ear(img_info['right_eye_ldmk']),label_dict[img_info['right_status']]])
    else:
        ldmks.append([-1,None])

    return ldmks

def list2num(slice_list):
    num_list = []
    for slice in slice_list:
        num_list.extend(list(range(slice[0], slice[1] + 1)))
    return num_list


def is_blink(blink_list,left_index,right_index):
    # 1 : stretch 0: normal -1 : ignore

    max_union = -1
    frame_len = right_index - left_index
    for stretch in blink_list:

        stretch_len = abs(stretch[1] - stretch[0])
        temp_left = max(left_index,stretch[0])
        temp_right = min(right_index,stretch[1])


        if [temp_left,temp_right] in [stretch,[left_index,right_index]]:
            return 1

        union =  (temp_right - temp_left) /( min(stretch_len,frame_len) + 0.1)
        max_union = max(max_union,union)

    if max_union > 0.7:
        return 1

    if max_union < 0.1:
        return 0

    return -1




def get_batch_data(video_list,suffix,batch_size,start_id,dst_dir,random_ratio = 0.01):
    random.shuffle(video_list)
    data_id = -1
    half_frame_len = 5

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


        blink_path = video_path.replace(video_suffix, '_blink.json')

        if not os.path.exists(blink_path):
            continue

        with open(blink_path, 'r') as f:
            jsons = json.load(f)

        close_list = jsons['blink_close']
        open_list = jsons['blink_open']


        with open(json_path, 'r') as f:
            big_json = f.readlines()

        left_ears = []
        right_ears = []
        left_label =  []
        right_label = []

        jsons_dict = {}
        for json_info in big_json:
            try:
                json_info = json.loads(json_info.strip())
            except:
                continue

            json_info =  get_fea_label(json_info)

            left_ears.append(json_info[0][0])
            right_ears.append(json_info[1][0])
            left_label.append(json_info[0][1])
            right_label.append(json_info[1][1])


        for i in range(len(left_ears)):

            if i < half_frame_len or i >= len(left_ears) - half_frame_len - 1:
                continue


            if random.random() > random_ratio:
                continue

            ears = left_ears
            labels = left_label
            if random.random() > 0.5:
                ears = right_ears
                labels = right_label


            if labels[i] is None:
                continue


            is_blink_close = is_blink(close_list, i - half_frame_len,i + half_frame_len)
            is_blink_open = is_blink(open_list, i - half_frame_len,i + half_frame_len)

            label = 0
            if is_blink_close ==  1 and is_blink_open == 1:
                label = 1
            elif 1 in [is_blink_close,is_blink_open]  :
                continue


            temp_ear = ears[i-half_frame_len:i+half_frame_len]
            data_id += 1

            temp_fea = temp_ear + [label]

            temp_data = list(map(str, temp_fea))
            temp_data = ' '.join(temp_data)

            temp_data_name = '_'.join(video_path.split(os.sep)[-4:]).replace(video_suffix, '')
            temp_data_name = '{}__{}.txt'.format(temp_data_name, i)

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
                    'test':'/data/weiyu.li/DMSData/FatigueView/test_video'

    }

    src_dir_dict = {'test':'/data/weiyu.li/DMSData/FatigueView/test_video'
    }

    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']


    src_dir_dict = {'sanbao_test':'/data/weiyu.li/DMSData/FatigueView/sanbao_test_video'

    }
    camera_list = ['ir_down']



    data_type = 'train'
    camera_id = 0

    for data_type in src_dir_dict.keys():
        for camera_id in range(len(camera_list)):

            src_dir = src_dir_dict[data_type]

            camera_type = camera_list[camera_id]

            dst_dir = './data_new/{}/{}'.format(data_type,camera_type)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            video_list = getFiles(src_dir, '.mp4')
            video_list += getFiles(src_dir, '.avi')

            # if data_type == 'train':
            #     video_list = [v for v in video_list if 'zhaoxinmei' not in v]


            all_num = 60000
            running_num = 32
            running_num = min(running_num,len(video_list))
            batch_size = all_num//running_num
            split_videos = split(video_list, running_num)



            process_list = []
            for i in range(running_num):
                temp_p = Process(target=get_batch_data,args=(split_videos[i],suffix,batch_size,batch_size*i,dst_dir,))
                process_list.append(temp_p)

            for temp_p in process_list:
                temp_p.start()

            for temp_p in process_list:
                temp_p.join()


            print('END')
