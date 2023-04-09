# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/11/5 12:51 AM
=================================================='''

import numpy as np
import cv2
import random



def get_box(points):
    ldmk = np.array(points)
    x_min,y_min = np.min(ldmk,0)
    x_max, y_max = np.max(ldmk, 0)
    box =  [x_min,y_min,x_max, y_max]
    return box


def eye_scale_2(rect,ratio=1.25):
    x,y = (rect[0]+rect[2])/2,(rect[1]+rect[3])/2
    w,h = rect[2] - rect[0],rect[3] - rect[1]

    w = w*ratio
    h = w/2

    rect = [x-w/2,y-h/2,x + w / 2,y+h/2]
    return rect


def eye_crop(img,rect,ratio=1.2,out_drop = False):
    img = img.copy()
    img_h, img_w = img.shape[:2]
    rect = eye_scale_2(rect,ratio)
    rect = np.array(rect).astype(np.int)

    img_rect = np.clip(rect, 0, max(img_h, img_w))
    img_rect[2], img_rect[3] = min(rect[2], img_w), min(rect[3], img_h)

    img_croped = img[img_rect[1]:img_rect[3], img_rect[0]:img_rect[2]]

    if not out_drop:
        w, h = img_rect[2] - img_rect[0], img_rect[3] - img_rect[1]
        big_w, big_h = rect[2] - rect[0], rect[3] - rect[1]
        x,y = -min(0,rect[0]),-min(0,rect[1])
        blank_img = np.zeros(shape=(big_h,big_w,3))
        blank_img[y:y+h,x:x+w] = img_croped[:,:]
        img_croped = blank_img


    return img_croped


def eye_crop_single_channel(img,rect,ratio=1.2,out_drop = False):
    img = img.copy()
    img_h, img_w = img.shape[:2]
    rect = eye_scale_2(rect,ratio)
    rect = np.array(rect).astype(np.int)

    img_rect = np.clip(rect, 0, max(img_h, img_w))
    img_rect[2], img_rect[3] = min(rect[2], img_w), min(rect[3], img_h)

    img_croped = img[img_rect[1]:img_rect[3], img_rect[0]:img_rect[2]]

    if not out_drop:
        w, h = img_rect[2] - img_rect[0], img_rect[3] - img_rect[1]
        big_w, big_h = rect[2] - rect[0], rect[3] - rect[1]
        x,y = -min(0,rect[0]),-min(0,rect[1])
        blank_img = np.zeros(shape=(big_h,big_w))
        blank_img[y:y+h,x:x+w] = img_croped[:,:]
        img_croped = blank_img

    return img_croped



def crop_img(img,eye_ldmk):
    eye_box = get_box(eye_ldmk)
    eye = eye_crop(img,eye_box)


    return eye



