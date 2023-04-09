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

def get_ear_height(ldmk):
    heights = [ldmk[2][1]-ldmk[6][1],ldmk[1][1]-ldmk[7][1],ldmk[3][1]-ldmk[5][1]]
    return np.mean(np.abs(heights))



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


def cal_alpha(d,pre_d,pre_b,median_d):
    alpha = [0.4,15,0.5,2,0.7]
    ans = alpha[0]
    ans *= np.exp(-alpha[1]*(d-pre_d)**2)
    ans *= np.exp(-alpha[2] *max((d - pre_b),0))
    ans *= np.exp(-alpha[3] *max((pre_b - d),0))
    ans *= (0,1)[d >= median_d*alpha[4]]

    return ans



def finetune_eye_heights(heights,median_d= None):
    if median_d is None:
        median_d = np.median([v for v in heights if v != -1])

    pre_b = median_d
    pre_d = median_d

    finetuned_heights = []
    for height in heights:
        if height == -1:
            finetuned_heights.append(height)
            pre_b = median_d
            pre_d = median_d
            continue

        alpha = cal_alpha(height,pre_d,pre_b,median_d)

        pre_b = (1-alpha)*pre_b + height * alpha

        finetuned_heights.append(height/(pre_b + 0.00001))

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


def list2bin(data_list,min_data,max_data , bin_num = 10):
    sep = (max_data-min_data)/bin_num
    bin_list = [0 for i in range(bin_num+1)]
    for data in data_list:
        if data == -1:
            bin_list[-1] += 1
        bin = (data - min_data)/sep
        bin_list[int(bin)] += 1


    bin_list = [v/len(data_list) for v in bin_list]

    return bin_list


def list2num(slice_list):
    num_list = []
    for slice in slice_list:
        num_list.extend(list(range(slice[0], slice[1] + 1)))
    return num_list


def get_batch_data(video_list,suffix,batch_size,start_id,dst_dir,stage= 'train',time_len = 30,random_ratio = 0.5):
    badcase_list = np.load('badcase.npy')

    badcase_list = [os.path.basename(v) for v in badcase_list]




    random.shuffle(video_list)

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



        frames = []
        cap = cv2.VideoCapture(video_path)
        frame_id = -1
        while True:
            frame_id += 1
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame)
            if len(frames) > 2*half_frame_len:
                _ = frames.pop(0)

            temp_data_name = '_'.join(video_path.split(os.sep)[-4:]).replace('.mp4', '')
            temp_data_name = '{}__{}.txt'.format(temp_data_name, frame_id-half_frame_len)

            if temp_data_name not in badcase_list:
                continue


            fea_txt = os.path.join('./data/test/ir_front',temp_data_name)
            with open(fea_txt,'r') as f:
                fea_info = f.read().strip().split()
            label = fea_info[-2]

            if label == '0' and random.random() < 0.9:
                continue

            for temp_frame_id,frame in enumerate(frames):
                if temp_frame_id % 8 != 0:
                    continue
                jpg_name = temp_data_name.replace('.txt','_{}.jpg'.format(temp_frame_id))
                jpg_name = label + '_' + jpg_name
                cv2.imwrite(os.path.join(dst_dir,jpg_name),frame)


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

    src_dir_dict = {'test':'/data/weiyu.li/DMSData/FatigueView/test_video'
    }

    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']
    camera_list = ['ir_front']


    data_type = 'train'
    camera_id = 0

    for data_type in src_dir_dict.keys():

        for camera_id in range(len(camera_list)):

            src_dir = src_dir_dict[data_type]

            camera_type = camera_list[camera_id]

            dst_dir = './badcase/{}/{}'.format(data_type,camera_type)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            video_list = getFiles(src_dir, '.mp4', camera_type)
            video_list = [v for v in video_list if 'fanzhikang' in v]

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
