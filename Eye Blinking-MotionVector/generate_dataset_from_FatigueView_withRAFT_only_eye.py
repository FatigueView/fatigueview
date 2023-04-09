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
    from .tools import crop_img,eye_crop,get_box,eye_scale_2
    from .RAFT.core.raft import RAFT
    from .RAFT.core.utils.utils import InputPadder

except:
    from tools import crop_img,eye_crop,get_box,eye_scale_2
    from RAFT.core.raft import RAFT
    from RAFT.core.utils.utils import InputPadder



def load_image(img):
    img = img.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def cal_Flow(img1,img2):
    scale = 0.8
    levels = 2
    winsize = 5
    iterations = 10
    poly_n = 5
    poly_sigma = 1.1
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
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



def get_single_status(flow,thres):
    # eye_rect = scale_height(rect,2)
    # eye_flow = eye_crop(flow,eye_rect,1)

    cell_motion = cellMotion(flow)
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


def finetune_two_frame_rect(rect_1,rect_2):
    w = max([rect_1[2] - rect_1[0],rect_2[2]-rect_2[0]])

    if w > rect_1[2] - rect_1[0]:
        rect_1 = eye_scale_2(rect_1,w/(rect_1[2] - rect_1[0]))

    if w > rect_2[2] - rect_2[0]:
        rect_2 = eye_scale_2(rect_2,w/(rect_2[2] - rect_2[0]))

    return rect_1,rect_2




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

        json_path = video_path.replace('.mp4', suffix)
        with open(json_path, 'r') as f:
            big_json = f.readlines()

        jsons_dict = {}
        for json_info in big_json:
            try:
                json_info = json.loads(json_info.strip())
            except:
                continue
            jsons_dict[int(json_info['frame'])] = json_info



        blink_path = video_path.replace(os.path.basename(video_path), 'blink.json')
        if not os.path.exists(blink_path):
            continue

        with open(blink_path, 'r') as f:
            jsons = json.load(f)

        close_list = blinklist2num(jsons['blink_close'])
        open_list = blinklist2num(jsons['blink_open'])



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
            jsons.append(jsons_dict[frame_id])
            if len(frames) > 2:
                _ = frames.pop(0)
                _ = jsons.pop(0)

            if len(frames) < 2:
                continue

            if frame_id not in close_list and frame_id not in open_list:
                if random.random() > random_ratio:
                    continue

            if not is_good_jsons(jsons[0]):
                continue

            try:
                left_eye = get_box(jsons[0]['left_eye_ldmk'])
                right_eye = get_box(jsons[0]['right_eye_ldmk'])

                left_eye_1 = get_box(jsons[1]['left_eye_ldmk'])
                right_eye_1 = get_box(jsons[1]['right_eye_ldmk'])
                thres = get_treshold(cal_distance(left_eye, right_eye))
            except:
                continue


            left_eye , left_eye_1 = finetune_two_frame_rect(left_eye,left_eye_1)
            right_eye, right_eye_1 = finetune_two_frame_rect(right_eye,right_eye_1)

            left_frame_0 = eye_crop(frames[0],left_eye,ratio=1.2)
            right_frame_0 = eye_crop(frames[0], right_eye, ratio=1.2)

            left_frame_1 = eye_crop(frames[-1],left_eye_1,ratio=1.2)
            right_frame_1 = eye_crop(frames[-1], right_eye_1, ratio=1.2)

            temp_frames = [left_frame_0,left_frame_1]
            if random.random() > 0.5:
                temp_frames = [right_frame_0, right_frame_1]


            for i in range(len(temp_frames)):
                temp_frames[i] = np.uint8(cv2.resize(temp_frames[i],(60,30)))

            flow = cal_Flow(temp_frames[0], temp_frames[1])
            flow_y = flow[:, :, 0]
            eye_label = get_single_status(flow_y,thres)


            for i in range(len(temp_frames)):
                temp_frames[i] = np.uint8(cv2.resize(temp_frames[i],(300,150)))

            pair_imgs = temp_frames
            pair_imgs = list(map(load_image,pair_imgs))
            pair_imgs = torch.stack(pair_imgs, dim=0)
            pair_imgs = pair_imgs.to(DEVICE)

            padder = InputPadder(pair_imgs.shape)
            pair_imgs = padder.pad(pair_imgs)[0]

            image1 = pair_imgs[0, None]
            image2 = pair_imgs[1, None]


            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            raft_flow_y = flow_up[0,1,:,:].detach().cpu().numpy()
            raft_flow_y  = cv2.resize(raft_flow_y,(40,20))


            raft_eye_label = get_single_status(raft_flow_y,thres)


            label = 0
            if frame_id in close_list:
                label = -1
            if frame_id in open_list:
                label = 1

            temp_info = {'thres': thres,
                         'left_eye': list(map(np.float,left_eye)),
                         'right_eye': list(map(np.float,right_eye)),
                         'eye_label':np.float(eye_label),
                         'raft_eye_label':np.float(raft_eye_label),
                         'label':np.float(label)}


            data_id += 1

            base_name = '_'.join(video_path.split(os.sep)[-4:]).replace('.mp4', '')
            temp_data_name = '{}__{}.json'.format(base_name, frame_id)
            with open(os.path.join(dst_dir,temp_data_name),'w') as f:
                json.dump(temp_info,f)

            flow_name = '{}__{}.npy'.format(base_name, frame_id)
            np.save(os.path.join(dst_dir,flow_name),flow_y)

            raft_flow_name = 'raft__{}__{}.npy'.format(base_name, frame_id)
            np.save(os.path.join(dst_dir,raft_flow_name),raft_flow_y)


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
    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']



    data_type = 'test'
    for camera_id in range(len(camera_list)):

        src_dir = src_dir_dict[data_type]

        camera_type = camera_list[camera_id]

        dst_dir = './data_only_eye_new/{}/{}'.format(data_type,camera_type)
        if not os.path.exists(dst_dir):
            os.makedirs(dst_dir)

        video_list = getFiles(src_dir, '.mp4', camera_type)


        all_num = 10000
        running_num = 18
        batch_size = all_num//running_num
        split_videos = split(video_list, running_num)



        process_list = []
        for i in range(running_num):
            temp_p = Process(target=get_batch_data,args=(split_videos[i],suffix,batch_size,batch_size*i,dst_dir,i%3+1))
            process_list.append(temp_p)

        for temp_p in process_list:
            temp_p.start()

        for temp_p in process_list:
            temp_p.join()


        print('END')
