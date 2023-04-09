import sys

sys.path.insert(0, '/data/site-packages/site-packages')
sys.path.insert(0, '/home/zhenyu/env/pytorch/')

sys.path.append('./')
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder

DEVICE = 'cuda'


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img


def load_image_list(image_files):
    images = []
    for imfile in sorted(image_files):
        images.append(load_image(imfile))

    images = torch.stack(images, dim=0)
    images = images.to(DEVICE)

    padder = InputPadder(images.shape)
    return padder.pad(images)[0]


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

    with torch.no_grad():
        images_list = glob.glob(os.path.join(args.path, '*.png')) + \
                      glob.glob(os.path.join(args.path, '*.jpg'))
        images_list.sort()

        images = load_image_list(images_list)
        for i in tqdm(range(images.shape[0] - 1)):
            image1 = images[i, None]
            image2 = images[i + 1, None]

            img_path_1 = images_list[i].split(os.sep)[-2]
            img_path_2 = images_list[i + 1].split(os.sep)[-2]
            if img_path_1 != img_path_2:
                continue

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            viz(image1, flow_up, images_list[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='temp.pth',
                        help="restore checkpoint")
    parser.add_argument('--path',
                        default='/home/zhenyu/data/amap//train_0712/videos/000687',
                        help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true',
                        help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true',
                        help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
