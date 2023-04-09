import sys

import sys
sys.path.insert(0,'/data/zhenyu.yang/modules')

sys.path.append('./')
sys.path.append('core')

import argparse
import torch

import os
import cv2
import glob
import numpy as np
from tqdm import tqdm

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(img):
    img = img.astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


# def load_image_list(image_files):
#     images = []
#     for imfile in sorted(image_files):
#         images.append(load_image(imfile))
#     return images


def viz(img, flo, origin_img_path=None):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    flo = flo[0].permute(1, 2, 0).cpu().numpy()

    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    if origin_img_path is not None:

        img_name = os.path.basename(origin_img_path)
        video_name = origin_img_path.split(os.sep)[-2]

        dst_dir = './outputs'

        temp_dir = os.path.join(dst_dir, video_name)
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)

        img = img_flo[:, :, [2, 1, 0]]
        cv2.imwrite(os.path.join(temp_dir, img_name), img)



    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()


def demo(args):
    model = torch.nn.DataParallel(RAFT(args))
    model_param = torch.load(args.model)
    model.load_state_dict(model_param)

    model = model.module
    model.to(DEVICE)
    model.eval()

    video = cv2.VideoCapture(args.path)

    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))  # float
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float

    height = height*2

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('/data/zhenyu.yang/output/raft/blink.avi', fourcc,25, (width, height))



    images_list = []
    for i in range(100):
        ret,frame = video.read()
        images_list.append(load_image(frame))

    with torch.no_grad():

        for i in tqdm(range(len(images_list) - 1)):
            pair_imgs = images_list[i:i+2]
            try:
                pair_imgs = torch.stack(pair_imgs, dim=0)
            except:
                continue
            pair_imgs = pair_imgs.to(DEVICE)

            padder = InputPadder(pair_imgs.shape)
            pair_imgs = padder.pad(pair_imgs)[0]

            image1 = pair_imgs[0, None]
            image2 = pair_imgs[1, None]


            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)


            img = image1
            flo = flow_up
            img = img[0].permute(1, 2, 0).cpu().numpy()
            flo = flo[0].permute(1, 2, 0).cpu().numpy()

            flo = flow_viz.flow_to_image(flo)
            img_flo = np.concatenate([img, flo], axis=0)
            img_flo = np.uint8(img_flo)
            out.write(img_flo)

        out.release()
        video.release()



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='temp.pth',
                        help="restore checkpoint")
    parser.add_argument('--path',
                        default='/data/weiyu.li/DMSData/FatigueView/raw_video/na/scenario_glass/1592211867/rgb_front.mp4',
                        help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
