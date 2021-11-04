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

try:
    from .skeleton_vis_tools import  draw_points_and_skeleton,joints_dict
except:
    from skeleton_vis_tools import draw_points_and_skeleton,joints_dict


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
    skeleton = np.zeros((17,3))
    if 'skeleton' in img_info and img_info['skeleton'] is not None and  len(img_info['skeleton']) > 4:
        for i ,pt in enumerate(img_info['skeleton']):
            if i >= 11:
                break
            if pt[1] > 0:
                skeleton[i] = [pt[0],pt[1],1]
    return skeleton



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



        json_path = video_path.replace('.mp4', suffix)
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

        with open(stretch_path, 'r') as f:
            stretch_list = json.load(f)


        stretch_index_list = []
        normal_index_list = []
        cap = cv2.VideoCapture(video_path)
        for i in range(len(skeleton_list)):
            ret,frame = cap.read()
            if not ret:
                break

            if i % 5 != 0 :
                continue

            if i < half_frame_len or i >= len(skeleton_list) - (half_frame_len+1):
                continue

            temp_stretch = is_stretch(stretch_list,i-half_frame_len,i+half_frame_len)
            if temp_stretch == 1:
                stretch_index_list.append(i)

                npy_name = '_'.join(video_path.split(os.sep)[-4:]).replace('.mp4','')
                npy_name = '{}__{}__{}.jpg'.format(1,npy_name,i)

                pt = skeleton_list[i]
                pt[:, :2] = pt[:, 1::-1]

                frame = draw_points_and_skeleton(frame, pt, joints_dict()[
                    'coco']['skeleton'], person_index=0,
                                                 points_color_palette='gist_rainbow',
                                                 skeleton_color_palette='jet',
                                                 points_palette_samples=10)

                cv2.imwrite(os.path.join(dst_dir,npy_name),frame)

        cap.release()


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

    camera_list = ['ir_left_up']

    # camera_list = ['rgb_left_up']



    data_type = 'train'
    camera_id = 0

    for data_type in src_dir_dict.keys():
        for camera_id in range(len(camera_list)):

            src_dir = src_dir_dict[data_type]

            camera_type = camera_list[camera_id]

            dst_dir = './vis/{}/{}'.format(data_type,camera_type)
            if not os.path.exists(dst_dir):
                os.makedirs(dst_dir)

            video_list = getFiles(src_dir, '.mp4', camera_type)

            if data_type == 'test':
                video_list = [v for v in video_list if 'fengchunshen' not in v and 'panbijia' not in v]


            if data_type == 'train':
                video_list = [v for v in video_list if 'zhaoxinmei' not in v]


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
