# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/11/5 12:51 AM
=================================================='''

import numpy as np
import cv2
import random



def HOG(img):
    winSize = (6, 6)
    blockSize = (6, 6)
    blockStride = (1, 1)
    cellSize = (3, 3)
    nbins = 9
    winStride = (1, 1)
    padding = (1, 1)

    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)

    hog_ans = hog.compute(img, winStride, padding)
    return hog_ans


def get_cov_matrix(gradient):
    cov_matrix = [gradient[0]**2,gradient[0]*gradient[1],gradient[0]*gradient[1],gradient[1]**2]
    return np.array(cov_matrix)

def HPOG(img,size):
    if not isinstance(size,tuple):
        size = (size,size)

    x_gradient = cv2.Sobel(img, cv2.CV_64F, 1, 0)
    y_gradient = cv2.Sobel(img, cv2.CV_64F, 0, 1)
    gradient = np.stack([x_gradient,y_gradient],axis=-1)

    hog_ans = gradient

    big_cov_matrix = np.zeros((len(hog_ans),len(hog_ans[0]),4))

    for i in range(len(hog_ans)):
        for j in range(len(hog_ans[0])):
            big_cov_matrix[i,j,:] = get_cov_matrix(hog_ans[i,j])

    big_cov_matrix = cv2.blur(big_cov_matrix, size)


    ans = []
    for i in range(len(big_cov_matrix)):
        for j in range(len(big_cov_matrix[0])):
            cov_matrix = big_cov_matrix[i,j].reshape((2,2))
            eigenvalue, featurevector = np.linalg.eig(cov_matrix)
            if abs(eigenvalue[0]) > abs(eigenvalue[-1]):
                featurevector = featurevector[0]
            else:
                featurevector = featurevector[-1]
            ans.extend([featurevector[0],featurevector[1]])

    return ans






def origin_LBP(img):
    dst = np.zeros(img.shape,dtype=img.dtype)
    h,w=img.shape
    start_index=1
    for i in range(start_index,h-1):
        for j in range(start_index,w-1):
            center = img[i][j]
            code = 0
#             顺时针，左上角开始的8个像素点与中心点比较，大于等于的为1，小于的为0，最后组成8位2进制
            code |= (img[i-1][j-1] >= center) << (np.uint8)(7)
            code |= (img[i-1][j  ] >= center) << (np.uint8)(6)
            code |= (img[i-1][j+1] >= center) << (np.uint8)(5)
            code |= (img[i  ][j+1] >= center) << (np.uint8)(4)
            code |= (img[i+1][j+1] >= center) << (np.uint8)(3)
            code |= (img[i+1][j  ] >= center) << (np.uint8)(2)
            code |= (img[i+1][j-1] >= center) << (np.uint8)(1)
            code |= (img[i  ][j-1] >= center) << (np.uint8)(0)
            dst[i-start_index][j-start_index]= code
    return dst


def LTP(img):

    LBP = origin_LBP(img)


    return LBP


def gabor_wavelet(rows, cols, kmax, f, orientation, scale, delt2):

    k = (kmax / (f ** scale)) * np.exp(1j * orientation * np.pi / 8)
    kn2 = np.abs(k) ** 2

    gw = np.zeros((rows, cols), np.complex128)

    for m in range(int(-rows/2) + 1, int(rows / 2) + 1):
        for n in range(int(-cols/2) + 1, int(cols / 2) + 1):
            t1 = np.exp(-0.5 * kn2 * (m**2 + n**2) / delt2)
            t2 = np.exp(1j * (np.real(k) * m + np.imag(k) * n))
            t3 = np.exp(-0.5 * delt2)
            gw[int(m + rows/2 - 1),int(n + cols/2 - 1)] = (kn2 / delt2) * t1 * (t2 - t3)

    return gw


def gabor_on_image(image):
    R = 128
    C = 128

    kmax = np.pi / 2
    f = np.sqrt(2)
    delt2 = (2 * np.pi) ** 2
    u=v=2

    gw = gabor_wavelet(R, C, kmax, f, u, v, delt2)

    resultR = cv2.filter2D(image, cv2.CV_32F, np.real(gw))
    resultI = cv2.filter2D(image, cv2.CV_32F, np.imag(gw))
    result = np.hypot(resultR, resultI)

    return result




def get_box(points):
    x_min,y_min = np.min(points,0)
    x_max, y_max = np.max(points, 0)
    box =  [x_min,y_min,x_max, y_max]
    return box


def eye_scale_2_bak(rect,ratio=1.25):
    x,y = (rect[0]+rect[2])/2,(rect[1]+rect[3])/2
    w,h = rect[2] - rect[0],rect[3] - rect[1]

    w = w*ratio
    h = w/2

    rect = [x-w/2,y-h/2,x + w / 2,y+h/2]
    return rect

def eye_scale_2(rect,ratio=1.25):
    x,y = (rect[0]+rect[2])/2,(rect[1]+rect[3])/2
    w,h = rect[2] - rect[0],rect[3] - rect[1]

    w = w*ratio
    h = h*ratio

    rect = [x-w/2,y-h/2,x + w / 2,y+h/2]
    return rect



def img_pad(img):
    h, w = img.shape[0:2]
    l = max(h, w)
    img = np.pad(img, (((l - h) // 2 + (l - h) % 2, (l - h) // 2),
                       ((l - w) // 2 + (l - w) % 2, (l - w) // 2), (0, 0)),
                 'constant')
    return img


#
# def eye_crop(img,rect,ratio=1.2,out_drop = False):
#     img = img.copy()
#     img_h, img_w = img.shape[:2]
#     rect = eye_scale_origin(rect,ratio)
#     rect = np.array(rect).astype(np.int)
#
#     img_rect = np.clip(rect, 0, max(img_h, img_w))
#     img_rect[2], img_rect[3] = min(rect[2], img_w), min(rect[3], img_h)
#
#     img_croped = img[img_rect[1]:img_rect[3], img_rect[0]:img_rect[2], :]
#
#     if not out_drop:
#         w, h = img_rect[2] - img_rect[0], img_rect[3] - img_rect[1]
#         big_w, big_h = rect[2] - rect[0], rect[3] - rect[1]
#         x,y = -min(0,rect[0]),-min(0,rect[1])
#         blank_img = np.zeros(shape=(big_h,big_w,3))
#         blank_img[y:y+h,x:x+w,:] = img_croped[:,:,:]
#         img_croped = blank_img
#
#     return img_croped

def eye_crop(img,rect,ratio=1.2,out_drop = False):
    img = img.copy()
    img_h, img_w = img.shape[:2]
    rect = eye_scale_2(rect,ratio)
    rect = np.array(rect).astype(np.int)

    if rect[0] >= img_w or  rect[1] >= img_h:
        return np.zeros((10,10,3))

    img_rect = np.clip(rect, 0, max(img_h, img_w))
    img_rect[2], img_rect[3] = min(rect[2], img_w), min(rect[3], img_h)

    img_croped = img[img_rect[1]:img_rect[3], img_rect[0]:img_rect[2], :]


    if not out_drop:
        w, h = img_rect[2] - img_rect[0], img_rect[3] - img_rect[1]
        big_w, big_h = rect[2] - rect[0], rect[3] - rect[1]
        x,y = -min(0,rect[0]),-min(0,rect[1])
        blank_img = np.zeros(shape=(big_h,big_w,3))
        try:
            blank_img[y:y+h,x:x+w,:] = img_croped[:,:,:]
        except:
            debug = 0
        img_croped = blank_img

    img_croped = img_pad(img_croped)

    return img_croped



def eye_crop_with_ldmk(img,ldmk,ratio=1.2):
    if ldmk is None:
        return np.zeros((10,10,3))
    bbox = get_box(ldmk)
    return eye_crop(img,bbox,ratio)


def crop_and_align(img,eye_ldmk):
    eye_box = get_box(eye_ldmk)
    eye = eye_crop(img,eye_box)
    theta = np.arctan2(eye_ldmk[4][1]-eye_ldmk[0][1],eye_ldmk[4][0]-eye_ldmk[0][0])
    h,w = eye.shape[:2]

    M = cv2.getRotationMatrix2D((w * 0.5, h * 0.5), -theta,1)
    eye = cv2.warpAffine(eye, M, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT,
                         borderValue=(0, 0, 0))

    return eye



