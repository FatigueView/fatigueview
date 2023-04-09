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


def getFiles(path, suffix,prefix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix) and file.startswith(prefix)]



def blink2numlist(blink_list,status_name):
    blink_list = [v == status_name for v in blink_list]
    blink_list.insert(0,False)
    blink_list.insert(-1,False)

    start_num = [i-1 for i in range(1,len(blink_list)) if not blink_list[i-1] and blink_list[i]]

    blink_list = blink_list[::-1]
    end_num = [len(blink_list) - i+1 for i in range(1,len(blink_list)) if not blink_list[i-1] and blink_list[i]]
    end_num = end_num[::-1]

    num_list = [abs(end-start)+1 for  start,end in zip(start_num,end_num)]


    return num_list


def get_ear(ldmk):
    eps = 1e-5
    get_distance = lambda x,y:((x[0]-y[0])**2 + (x[1]-y[1])**2 + eps)**0.5

    w = get_distance(ldmk[0],ldmk[4])

    h = [get_distance(ldmk[2],ldmk[6]),get_distance(ldmk[1],ldmk[7]),get_distance(ldmk[3],ldmk[5])]
    h = np.mean(h)
    ear = h/w
    ear = min(ear,0.7)

    return ear


def get_mouth_ear(ldmk):
    eps = 1e-5
    get_distance = lambda x,y:((x[0]-y[0])**2 + (x[1]-y[1])**2 + eps)**0.5

    w = get_distance(ldmk[60],ldmk[64])
    h = [get_distance(ldmk[62],ldmk[66]),get_distance(ldmk[61],ldmk[67]),get_distance(ldmk[63],ldmk[65])]
    h = np.mean(h)
    ear = h/w
    ear = min(ear,1)

    return ear



def get_cof_by_headpose(head_pose):

    yaw,pitch,roll = head_pose
    pitch = pitch * np.pi / 180
    yaw = -(yaw * np.pi / 180)
    roll = roll * np.pi / 180

    cof = max(np.cos(yaw), 0.4)/ max(np.cos(pitch),0.4)
    cof = min(max(cof,0.6),1.6)

    return cof

def get_fea_label(img_info):

    cof = 1
    if 'head_pose' in img_info and img_info['head_pose'] is not None and  len(img_info['head_pose']) > 1:
        cof = get_cof_by_headpose(img_info['head_pose'])


    eye_ears = []
    if 'left_eye_ldmk' in img_info and img_info['left_eye_ldmk'] is not None and  len(img_info['left_eye_ldmk']) > 4:
        eye_ears.append(get_ear(img_info['left_eye_ldmk']) * cof)
    else:
        eye_ears.append(-1)
    if 'right_eye_ldmk' in img_info and img_info['right_eye_ldmk'] is not None and len(img_info['right_eye_ldmk']) > 4:
        eye_ears.append(get_ear(img_info['right_eye_ldmk']) * cof)
    else:
        eye_ears.append(-1)


    if eye_ears[0] == -1 and eye_ears[1] == -1:
        eye_ear =  -1
    elif eye_ears[0] == -1:
        eye_ear =  eye_ears[1]
    elif eye_ears[1] == -1:
        eye_ear =  eye_ears[0]
    else:
        eye_ear = np.mean(eye_ears)

    mouth_ear = -1
    if 'ldmk' in img_info and img_info['ldmk'] is not None and  len(img_info['ldmk']) > 4:
        mouth_ear = get_mouth_ear(img_info['ldmk']) * cof

    return eye_ear,mouth_ear



def get_yawn_info(yawn_list):
    is_yawn = []
    for v in yawn_list:
        if v == -1:
            is_yawn.append(v)
            continue
        if v > 0.6:
            is_yawn.append(1)
        else:
            is_yawn.append(0)

    is_yawn = blink2numlist(is_yawn,1)

    yawn_rate = len(is_yawn) / (sum(v != -1 for v in yawn_list) + 0.001)

    return yawn_rate


def get_statics(data_list):
    if len(data_list) == 0:
        data_list = [-1]
    data_list.sort()
    data_len = len(data_list)
    return [np.mean(data_list),data_list[0],data_list[-1],data_list[int(data_len*0.25)],data_list[int(data_len*0.5)],data_list[int(data_len*0.75)]]

def get_blink_info(ear_list):
    is_close = []

    reopen_list = []
    close_list = []

    for v in ear_list:
        if v == -1:
            is_close.append(v)
            continue
        if v < 0.16:
            is_close.append(1)
        else:
            is_close.append(0)


    blink_duration =  blink2numlist(is_close,1)

    for i in range(len(ear_list) -1):
        if is_close[i] == 0 and is_close[i+1] == 1:
            close_list.append(abs(ear_list[i+1] - ear_list[i]))

        if is_close[i] == 1 and is_close[i+1] == 0:
            reopen_list.append(abs(ear_list[i+1] - ear_list[i]))


    preclose = sum(blink_duration) / (sum( v != -1 for v in ear_list) + 0.001)
    blink_rate = len(blink_duration) / (sum( v != -1 for v in ear_list) + 0.001)


    feas = [preclose,blink_rate] + get_statics(blink_duration) + get_statics(close_list) + get_statics(reopen_list)

    return feas



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


def get_batch_data(video_list,suffix,batch_size,start_id,dst_dir,stage= 'train',time_len = 60,random_ratio = 0.01):
    random.shuffle(video_list)
    half_frame_len = time_len * 25 // 2

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



        awake_path = video_path.replace(os.path.basename(video_path), 'awake.json')
        poor_acting_path = video_path.replace(os.path.basename(video_path), 'poor_acting.json')
        fatigue_path = video_path.replace(os.path.basename(video_path),'fatigue.json')

        awake_list = []
        poor_action_list = []
        fatigue_list = []

        if 'train' in stage  :
            if os.path.exists(awake_path):
                with open(awake_path, 'r') as f:
                    temp_awake_list = json.load(f)
                awake_list = list2num(temp_awake_list)

            if os.path.exists(poor_acting_path):
                with open(poor_acting_path, 'r') as f:
                    temp_poor_action_list = json.load(f)
                poor_action_list = list2num(temp_poor_action_list)


            fatigue_list = [i for i in range(len(height_list)) if i not in awake_list and i not in poor_action_list]


        if 'test' in stage:
            if os.path.exists(fatigue_path):
                with open(fatigue_path, 'r') as f:
                    temp_fatigue_list = json.load(f)
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

            if frame_id < half_frame_len or frame_id >= len(height_list) - (half_frame_len+1):
                continue


            awake_time = 0
            fatigue_time = 0
            for i in range(frame_id-half_frame_len,frame_id+half_frame_len):
                if i in awake_list:
                    awake_time += 1
                if i in fatigue_list:
                    fatigue_time += 1

            if awake_time < half_frame_len*1.2 and fatigue_time < half_frame_len*1.2:
                continue

            if awake_time >= half_frame_len*1.2 :
                label = 0
            else:
                label = 1

            temp_heights = height_list[frame_id - half_frame_len:frame_id + half_frame_len]
            ear_list = [v[0] for v in temp_heights]
            mouth_list = [v[1] for v in temp_heights]

            temp_feas = get_blink_info(ear_list)
            yawn = get_yawn_info(mouth_list)


            data_id += 1

            temp_feas = temp_feas + [yawn,label]

            temp_data = list(map(str, temp_feas))
            temp_data = ' '.join(temp_data)

            temp_data_name = '_'.join(video_path.split(os.sep)[-4:]).replace(video_suffix, '')
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
                    'test':'/data/weiyu.li/DMSData/FatigueView/test_video'

    }


    src_dir_dict = {'test':'/data/weiyu.li/DMSData/FatigueView/test_video'
    }

    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']



    src_dir_dict = {'beelab_test':'/data/weiyu.li/DMSData/FatigueView/beelab_test_video',
                    # 'beelab_train': '/data/weiyu.li/DMSData/FatigueView/beelab_train_video'

    }

    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']



    data_type = 'train'
    camera_id = 0

    for data_type in src_dir_dict.keys():
        for camera_id in range(len(camera_list)):

            src_dir = src_dir_dict[data_type]

            camera_type = camera_list[camera_id]

            dst_dir = '/mnt/data-1/zhenyu.yang/FatigueViewBaseline/Cheng2019AOD/data/{}/{}'.format(data_type,camera_type)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            video_list = getFiles(src_dir, '.mp4', camera_type)
            video_list += getFiles(src_dir, '.avi', camera_type)


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



            process_list = []
            for i in range(running_num):
                temp_p = Process(target=get_batch_data,args=(split_videos[i],suffix,batch_size,batch_size*i,dst_dir,data_type))
                process_list.append(temp_p)

            for temp_p in process_list:
                temp_p.start()

            for temp_p in process_list:
                temp_p.join()


            print('END')
