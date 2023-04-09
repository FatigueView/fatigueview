# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/9/1 4:17 PM
=================================================='''
import cv2
import os
import numpy as np
from small_tools import get_single_eye_rect,crop



def cal_Flow(img1,img2):
    scale = 0.5
    levels = 1
    winsize = 5
    iterations = 5
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
    eye_flow = crop(flow,eye_rect,1)

    cell_motion = cellMotion(eye_flow)
    cell_motion = cell_motion[:2].reshape(-1)
    average = np.mean(cell_motion)
    var = np.var(cell_motion)

    if var < thres:
        return 0

    if cell_motion[4] > 0 and cell_motion[4] > cell_motion[1]:
        return 1

    if cell_motion[4] < 0 and cell_motion[4] < cell_motion[1]:
        return -1

    return 0

def get_doube_eye_status(flow,info):
    left_eye = get_single_status(flow,info[0],info[-1])
    right_eye = get_single_status(flow,info[1],info[-1])
    return left_eye,right_eye

if __name__ == '__main__':

    mp4_dir = '/home/users/zhenyu.yang/big_data/eye_origin_pack/eye_blink_s202'
    mp4_list = os.listdir(mp4_dir)
    mp4_list = [os.path.join(mp4_dir,v) for v in mp4_list if v.endswith('.mp4')]

    video_path = mp4_list[1]
    json_path = video_path.replace('.mp4', '.json')
    json_file = open(json_path, 'r')

    video = cv2.VideoCapture(video_path)

    sz = (1280, 720)
    fps = 30
    mp4_path = './vis.mp4'
    video_writer = cv2.VideoWriter(mp4_path,
                                   cv2.VideoWriter_fourcc('m', 'p', '4', 'v'),
                                   fps, sz)

    imgs = []
    json_infos = []
    ret, fame = video.read()

    imgs.append(cv2.cvtColor(fame,cv2.COLOR_BGR2GRAY))
    info = json_file.readline().strip()
    rects = get_single_eye_rect(info)
    thres = get_treshold(cal_distance(rects[0], rects[1]))
    rects += [thres]
    json_infos.append(rects)


    while ret:
        ret,fame = video.read()
        if not ret:
            break

        imgs.append(cv2.cvtColor(fame, cv2.COLOR_BGR2GRAY))
        info = json_file.readline().strip()
        rects = get_single_eye_rect(info)
        thres = get_treshold(cal_distance(rects[0],rects[1]))
        rects += [thres]
        json_infos.append(rects)

        flow = cal_Flow(imgs[0], imgs[1])
        flow_y = flow[:, :, 0]

        left_eye,right_eye = get_doube_eye_status(flow_y,json_infos[-2])

        txt_info = '{}    {}'.format(left_eye,right_eye)

        img = imgs.pop(0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.putText(img,txt_info,(20,20),cv2.FONT_HERSHEY_COMPLEX,0.8, (0,255,0), 1)

        video_writer.write(img)

        # if len(imgs) ==2:
        #     break

    video_writer.release()

    # imgs = [cv2.cvtColor(v,cv2.COLOR_BGR2GRAY) for v in imgs]
    #
    # flow = cal_Flow(imgs[0],imgs[1])
    # flow_y = flow[:,:,1]
    #
    # left_eye_rect = scale_height(json_infos[0][0],2)
    # left_eye_flow_y = crop(flow_y,left_eye_rect,1)
    # left_eye = crop(imgs[0],left_eye_rect,1)
    #
    # cell_motion = cellMotion(left_eye_flow_y)
    # cell_motion = cell_motion[:2].reshape(-1)
    # average = np.mean(cell_motion)
    # var = np.var(cell_motion)
    #
    #
    # cv2.imwrite('test.jpg',left_eye)
    # cv2.imwrite('test_2.jpg', imgs[0])
    # cv2.imwrite('test_3.jpg', imgs[1])



