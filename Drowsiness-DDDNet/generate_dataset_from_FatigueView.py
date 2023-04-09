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
try:
    from .tools import eye_crop,get_box,eye_crop_with_ldmk
except:
    from tools import eye_crop,get_box,eye_crop_with_ldmk




def cal_Flow(img1,img2):
    scale = 0.5
    levels = 3
    winsize = 15
    iterations = 3
    poly_n = 5
    poly_sigma = 1.2
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(img1, img2,
                                        flow=None, pyr_scale=scale,
                                        levels=levels, iterations=iterations,
                                        winsize=winsize, poly_n=poly_n,
                                        poly_sigma=poly_sigma, flags=0)
    return flow




def getFiles(path, suffix,prefix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix) and file.startswith(prefix)]







def list2num(slice_list):
    num_list = []
    for slice in slice_list:
        num_list.extend(list(range(slice[0], slice[1] + 1)))
    return num_list



def get_fea_label(img_info):
    if isinstance(img_info,str):
        img_info = json.loads(img_info)

    eye_ldmk = None
    if 'left_eye_ldmk' in img_info and img_info['left_eye_ldmk'] is not None and  len(img_info['left_eye_ldmk']) > 4:
        eye_ldmk = np.array(img_info['left_eye_ldmk'])


    mouth_ldmk = None
    if 'ldmk' in img_info and img_info['ldmk'] is not None and  len(img_info['ldmk']) > 4:
        ldmk = np.array(img_info['ldmk'])
        mouth_ldmk = ldmk[48:67]

    return eye_ldmk,mouth_ldmk



def get_batch_data(video_list,suffix,batch_size,start_id,dst_dir,stage= 'train',time_len = 1,random_ratio = 0.5):
    random.shuffle(video_list)

    half_frame_len = time_len*25//2

    data_id = -1
    while True:
        print('start')
        tic = time.time()
        if len(video_list) == 0:
            break


        video_path = video_list.pop()

        video_suffix = '.mp4'
        if video_path.endswith('.mp4'):
            video_suffix = '.mp4'
        elif video_path.endswith('.avi'):
            video_suffix = '.avi'


        json_path = video_path.replace(video_suffix, suffix)

        yawning_path = video_path.replace(os.path.basename(video_path),'yawning.json')

        if not os.path.exists(yawning_path) or not os.path.exists(json_path) :
            continue



        if not os.path.exists(json_path):
            continue

        with open(json_path, 'r') as f:
            big_json = f.readlines()


        with open(yawning_path, 'r') as f:
            yawn_list = json.load(f)

        yawn_list = set(list2num(yawn_list))


        frame_len = len(big_json)




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

            fatigue_list = [i for i in range(frame_len) if i not in awake_list and i not in poor_action_list]


        if stage == 'test'  or 'test' in stage:
            if os.path.exists(fatigue_path):
                with open(fatigue_path, 'r') as f:
                    temp_fatigue_list = json.load(f)
                awake_list = list2num(temp_fatigue_list[0])
                fatigue_list = list2num(temp_fatigue_list[1]) + list2num(temp_fatigue_list[2]) + list2num(temp_fatigue_list[3])
            else:
                fatigue_list = []
                awake_list = list(range(frame_len))



        array_len = max(max(awake_list+[-1]),max(fatigue_list+[-1]),frame_len) + 1

        np_awake = np.array([0 for _ in range(array_len)])
        np_awake[awake_list] = 1

        np_fatigue = np.array([0 for _ in range(array_len)])
        np_fatigue[fatigue_list] = 1


        # print('time:{:.3f}'.format(time.time() - tic))


        cap = cv2.VideoCapture(video_path)


        for i in range(frame_len):

            ret,frame = cap.read()
            if not ret:
                break


            if random.random() > random_ratio:
                continue

            if i < half_frame_len or i >= frame_len - (half_frame_len+1):
                continue


            awake_time = np.sum(
                np_awake[i - half_frame_len:i + half_frame_len])
            fatigue_time = np.sum(
                np_fatigue[i - half_frame_len:i + half_frame_len])

            if awake_time < half_frame_len*1.2 and fatigue_time < half_frame_len*1.2:
                continue


            label = -1
            if awake_time >= half_frame_len*1.2 :
                label = 0
            elif fatigue_time >= half_frame_len*1.2:
                label = 1

            if i in yawn_list:
                label = 2

            eye_ldmk,mouth_ldmk = get_fea_label(big_json[i])

            if eye_ldmk is None and mouth_ldmk is None:
                continue

            if label == 2 and mouth_ldmk is None:
                continue


            eye = eye_crop_with_ldmk(frame,eye_ldmk)
            mouth = eye_crop_with_ldmk(frame, mouth_ldmk)


            base_name = '_'.join(video_path.split(os.sep)[-4:]).replace(video_suffix, '')


            frame_name = '{}__{}__{}_eye.jpg'.format(label,base_name, i)
            cv2.imwrite(os.path.join(dst_dir, frame_name),eye)


            frame_name = '{}__{}__{}_mouth.jpg'.format(label,base_name, i)
            cv2.imwrite(os.path.join(dst_dir, frame_name),mouth)

            data_id += 1


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

    src_dir_dict = {'test':'/data/weiyu.li/DMSData/FatigueView/test_video'
    }


    # src_dir_dict = {'test':'/data/weiyu.li/DMSData/FatigueView/test_video'
    # }

    camera_list = ['rgb_front']

    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']
    # camera_list = ['ir_front','ir_left_up','ir_up','rgb_front','rgb_left_up','rgb_up']
    # camera_list = ['ir_left']

    src_dir_dict = {'beelab_test':'/data/weiyu.li/DMSData/FatigueView/beelab_test_video',
                    'beelab_train': '/data/weiyu.li/DMSData/FatigueView/beelab_train_video'

    }

    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']





    data_type = 'train'
    camera_id = 0

    for data_type in src_dir_dict.keys():

        for camera_id in range(len(camera_list)):

            src_dir = src_dir_dict[data_type]

            camera_type = camera_list[camera_id]

            dst_dir = './data/{}/{}'.format(data_type,camera_type)
            dst_dir = '/mnt/data-1/zhenyu.yang/FatigueViewBaseline/Reddy2017RDD/data_3/{}/{}'.format(data_type,camera_type)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            video_list = getFiles(src_dir, '.mp4', camera_type)
            video_list += getFiles(src_dir, '.avi', camera_type)

            # if data_type == 'test':
            #     video_list = [v for v in video_list if 'fengchunshen' not in v and 'panbijia' not in v and 'basic' not in v]
            #     random_ratio = 0.005
            #     all_num = 30000
            #
            # if data_type == 'train':
            #     all_num = 100000
            #     video_list = [v for v in video_list if 'zhaoxinmei' not in v and 'basic' not in v]

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
