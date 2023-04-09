# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/11/5 11:37 AM
=================================================='''

import sys
sys.path.append('./')
sys.path.insert(0,'/data/zhenyu.yang/modules')
sys.path.append('./RAFT/core')

import cv2
import json
import numpy as np
import random
import copy
from multiprocessing import Process
import os
import argparse
import torch

try:
    from .tools import crop_img,eye_crop,get_box
    from .RAFT.core.raft import RAFT
    from .RAFT.core.utils.utils import InputPadder
    from .flow_vis import flow_to_image

except:
    from tools import crop_img,eye_crop,get_box
    from RAFT.core.raft import RAFT
    from RAFT.core.utils.utils import InputPadder
    from flow_vis import flow_to_image



def load_image(img):
    img = img.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img

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



# def cal_Flow(img1,img2):
#     scale = 0.8
#     levels = 2
#     winsize = 5
#     iterations = 10
#     poly_n = 5
#     poly_sigma = 1.1
#     img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
#     img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
#     flow = cv2.calcOpticalFlowFarneback(img1, img2,
#                                         flow=None, pyr_scale=scale,
#                                         levels=levels, iterations=iterations,
#                                         winsize=winsize, poly_n=poly_n,
#                                         poly_sigma=poly_sigma, flags=0)
#     return flow


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

def get_batch_data(video_list,suffix,batch_size,start_id,dst_dir,gpu_id=0,random_ratio = 0.05):

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='./RAFT/temp.pth',
                        help="restore checkpoint")
    parser.add_argument('--path',
                        default='/home/zhenyu/data/amap//train_0712/videos/',
                        help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficent correlation implementation')
    args = parser.parse_args()


    DEVICE = 'cuda:{}'.format(gpu_id)
    model = torch.nn.DataParallel(RAFT(args))
    model_param = torch.load(args.model,map_location=DEVICE)
    model.load_state_dict(model_param)
    model = model.module
    model.to(DEVICE)
    model.eval()



    random.shuffle(video_list)
    data_id = -1

    while True:
        if len(video_list) == 0:
            break

        video_path = video_list.pop()

        cap = cv2.VideoCapture(video_path)
        frame_id = -1


        frames = []
        jsons = []

        while True:
            frame_id += 1
            ret, frame = cap.read()

            if not ret:
                break

            frames.append(frame)
            if len(frames) < 2:
                continue

            if len(frames) > 2:
                _ = frames.pop(0)



            flow = cal_Flow(frames[0], frames[1])
            flow = flow_to_image(flow)


            pair_imgs = frames
            pair_imgs = list(map(load_image,pair_imgs))
            pair_imgs = torch.stack(pair_imgs, dim=0)
            pair_imgs = pair_imgs.to(DEVICE)

            padder = InputPadder(pair_imgs.shape)
            pair_imgs = padder.pad(pair_imgs)[0]

            image1 = pair_imgs[0, None]
            image2 = pair_imgs[1, None]


            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            raft_flow = flow_up[0].permute(1, 2, 0).detach().cpu().numpy()
            raft_flow = flow_to_image(raft_flow)

            img = np.concatenate([frames[0],flow,raft_flow],axis = 0)
            cv2.imwrite(os.path.join(dst_dir,'{}.jpg'.format(str(frame_id).zfill(6))),img)

            if frame_id > batch_size:
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

    dst_dir = './vis_flow'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    videos = ['/data/weiyu.li/DMSData/FatigueView/test_video/fatigue/shileilin/1603382315/rgb_front.mp4']

    get_batch_data(videos,suffix,1000,0,dst_dir,1)