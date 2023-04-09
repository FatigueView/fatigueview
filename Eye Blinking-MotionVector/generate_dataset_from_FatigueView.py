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
    from .tools import crop_img,eye_crop,get_box
except:
    from tools import crop_img,eye_crop,get_box


def cal_Flow(img1,img2):
    scale = 0.5
    levels = 1
    winsize = 5
    iterations = 10
    poly_n = 5
    poly_sigma = 1.1
    flow = cv2.calcOpticalFlowFarneback(img1, img2,
                                        flow=None, pyr_scale=scale,
                                        levels=levels, iterations=iterations,
                                        winsize=winsize, poly_n=poly_n,
                                        poly_sigma=poly_sigma, flags=0)
    return flow


def scale_height(bbox,scale=1.5):
    w,h = bbox[2]-bbox[0],bbox[3]-bbox[1]
    x,y = (bbox[2]+bbox[0])/2,(bbox[3]+bbox[1])/2
    h = h*scale
    rect = [x-w/2,y-h/2,x+w/2,y+h/2]
    rect = list(map(int,rect))
    return rect


def cellMotion(flow_y):
    h,w = flow_y.shape[:2]
    sep_h = h/3
    sep_w = w/3

    cell_motion = np.zeros((3,3))

    for i in range(3):
        for j in range(3):
            yl = int(i*sep_h)
            yr = int(i * sep_h+sep_h)

            xl = int(j*sep_w)
            xr = int(j * sep_w+sep_w)

            cell_motion[i,j] = np.mean(flow_y[yl:yr,xl:xr])

    return cell_motion

def cal_distance(rect_1,rect_2):
    x_1,y_1 = (rect_1[0]+rect_1[2])/2,(rect_1[1]+rect_1[3])/2
    x_2, y_2 = (rect_2[0] + rect_2[2]) / 2, (rect_2[1] + rect_2[3]) / 2
    return ((x_1-x_2)**2+(y_1-y_2)**2)**0.5

def get_treshold(distance):
    return 0.02*distance/25



def get_single_status(flow,rect,thres):
    eye_rect = scale_height(rect,2)
    eye_flow = eye_crop(flow,eye_rect,1)

    cell_motion = cellMotion(eye_flow)
    cell_motion = cell_motion[:2].reshape(-1)
    average = np.mean(cell_motion)
    var = np.var(cell_motion)

    if var < thres:
        return 0

    if cell_motion[4] > 0 and cell_motion[4] > cell_motion[1]:
        return -1 # close

    if cell_motion[4] < 0 and cell_motion[4] < cell_motion[1]:
        return 1 # open

    return 0

def get_doube_eye_status(flow,info):
    left_eye = get_single_status(flow,info[0],info[-1])
    right_eye = get_single_status(flow,info[1],info[-1])
    return left_eye,right_eye




def getFiles(path, suffix,prefix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix) and file.startswith(prefix)]



def blinklist2num(blink_list):
    num_list = []
    for blink in blink_list:
        num_list.extend(list(range(blink[0], blink[1] + 1)))
    return num_list


def is_good_jsons(img_info):
    ldmks = []
    if 'left_eye_ldmk' in img_info and img_info['left_eye_ldmk'] is not None and  len(img_info['left_eye_ldmk']) > 4:
        ldmks.append([img_info['left_eye_ldmk'],img_info['left_status']])
    if 'right_eye_ldmk' in img_info and img_info['right_eye_ldmk'] is not None and len(img_info['right_eye_ldmk']) > 4:
        ldmks.append([img_info['right_eye_ldmk'],img_info['right_status']])
    if len(ldmks) != 2:
        return False
    return True

def get_batch_data(video_list,suffix,batch_size,start_id,dst_dir,random_ratio = 0.2):
    random.shuffle(video_list)
    while True:
        video_path = video_list.pop()


        json_path = video_path.replace('.mp4', suffix)
        with open(json_path, 'r') as f:
            big_json = f.readlines()

        jsons_dict = {}
        for json_info in big_json:
            try:
                json_info = json.loads(json_info.strip())
            except:
                break
            jsons_dict[int(json_info['frame'])] = json_info



        blink_path = video_path.replace(os.path.basename(video_path), suffix)

        with open(blink_path, 'r') as f:
            jsons = json.load(f)

        close_list = blinklist2num(jsons['blink_close'])
        open_list = blinklist2num(jsons['blink_open'])



        cap = cv2.VideoCapture(video_path)
        frame_id = -1
        data_id = -1

        frames = []
        jsons = []

        while True:
            frame_id += 1
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)
            jsons.append(jsons_dict[frame_id])
            if len(frames) > 2:
                _ = frames.pop(0)
                _ = jsons.pop(0)

            if random.random() > random_ratio:
                continue


            if frame_id not in close_list and frame_id not in open_list:
                if random.random() < 0.6:
                    continue

            if not is_good_jsons(jsons[0]):
                continue


            left_eye = get_box(jsons[0]['left_eye_ldmk'])
            right_eye = get_box(jsons[0]['right_eye_ldmk'])

            thres = get_treshold(cal_distance(left_eye, right_eye))

            flow = cal_Flow(frames[0], frames[1])
            flow_y = flow[:, :, 0]


            left_eye_label, right_eye_label = get_doube_eye_status(flow_y, [left_eye,right_eye,thres])

            label = 0
            if frame_id in close_list:
                label = -1
            if frame_id in open_list:
                label = 1

            temp_info = {'thres': thres,
                         'left_eye': left_eye,
                         'right_eye': right_eye,
                         'left_eye_label':left_eye_label,
                         'right_eye_label':right_eye_label,
                         'label':label}


            data_id += 1


            temp_data_name = '{}.json'.format(start_id +data_id )
            with open(os.path.join(dst_dir,temp_data_name),'w') as f:
                json.dump(temp_info,f)

            flow_name = '{}.npy'.format(start_id +data_id )
            np.save(os.path.join(dst_dir,flow_name),flow_y)



            if data_id >= batch_size:
                return True


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
    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']



    data_type = 'test'
    for camera_id in range(len(camera_list)):

        src_dir = src_dir_dict[data_type]

        camera_type = camera_list[camera_id]

        dst_dir = './data/{}/{}'.format(data_type,camera_type)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        video_list = getFiles(src_dir, '.mp4', camera_type)


        all_num = 10000
        running_num = 32
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
