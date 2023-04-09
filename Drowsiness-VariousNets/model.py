# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/11/11 8:35 PM
=================================================='''

import sys
sys.path.insert(0,'/home/zhenyu/env/pytorch')
import torch
from torch import nn
import torch.nn.functional as F
import random
import cv2
import numpy as np
import os
from PIL import Image
import time

from torchvision import datasets, models, transforms
import torch.optim as optim
from torch.optim import lr_scheduler

try:
    from .Vgg_face_dag import vgg_face_dag
except:
    from Vgg_face_dag import vgg_face_dag


class MultiModel(nn.Module):
    def __init__(self):
        """
        PANnet
        :param model_config: 单帧图片（with more info） 加上3D的多帧信息
        """
        super().__init__()


        self.rgb_alex = models.alexnet(pretrained=True)
        num_ftrs = self.rgb_alex.classifier[6].in_features
        self.rgb_alex.classifier[6] = nn.Linear(num_ftrs, 2)

        self.flow_alex = models.alexnet(pretrained=True)
        num_ftrs = self.flow_alex.classifier[6].in_features
        self.flow_alex.classifier[6] = nn.Linear(num_ftrs, 2)


        self.rgb_vgg = vgg_face_dag(weights_path='/home/users/zhenyu.yang/data/FatigueBaseLine/Park2016DDD/params/vgg_face_dag.pth')
        num_ftrs = self.rgb_vgg .fc8.in_features
        self.rgb_vgg.fc8 = nn.Linear(num_ftrs, 2)


    def forward(self, rgb_img,flow):
        ans = self.rgb_alex(rgb_img) + self.flow_alex(rgb_img) + self.rgb_alex(flow)
        ans = ans/3
        return ans






def random_rotate(imgs):
    max_angle = 7
    angle = random.random() * 2 * max_angle - max_angle
    for i in range(len(imgs)):
        img = imgs[i]
        w, h = img.shape[:2]
        rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
        img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
        imgs[i] = img_rotation
    return imgs

def random_crop(imgs, min_ratio):
    if random.random() < 0.05:
        return imgs
    ratio = random.uniform(min_ratio, 1)
    h, w = imgs[0].shape[0:2]

    th, tw = int(h * ratio), int(w * ratio)

    if w == tw and h == th:
        return imgs

    i = random.randint(0, h - th)
    j = random.randint(0, w - tw)

    # return i, j, th, tw
    for idx in range(len(imgs)):
        if len(imgs[idx].shape) == 3:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw, :]
        else:
            imgs[idx] = imgs[idx][i:i + th, j:j + tw]
    return imgs


def resizes(imgs, dst_size):
    try:
        imgs = [cv2.resize(v, dsize=dst_size) for v in imgs]
    except:
        debug = 0
    return imgs


transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_list, label_list,stage):
        self.data_list = data_list
        self.label_list = label_list
        self.stage = stage

        self.blank_img = torch.FloatTensor(np.zeros((3,224,224)))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        try:
            img = cv2.imread(self.data_list[index])
            flow = np.load(self.data_list[index].replace('.jpg','.npy'))
            if len(img) > 0 and len(flow) > 0:
                pass
            else:
                return self.blank_img,self.blank_img,-1,index
        except:
            return self.blank_img, self.blank_img, -1, index


        flow = flow.astype('float')/100
        temp_flow = np.zeros((flow.shape[0],flow.shape[1],3))
        temp_flow[:,:,:2] = flow
        flow = temp_flow

        imgs = [img,flow]

        if self.stage == 'train':
            imgs = random_rotate(imgs)
            imgs = random_crop(imgs,0.8)

        imgs = resizes(imgs,(224,224))


        img, flow = imgs
        img = img[:,:,::-1]
        img =  Image.fromarray(img.astype('uint8'))

        flow = np.swapaxes(flow, 0, 2)
        flow = np.swapaxes(flow, 1, 2)
        flow = torch.FloatTensor(flow)/10


        label = int(self.label_list[index])
        img = transform(img)
        return img,flow, label,index


def train(model,data_list,label_list,epoch,model_prefix):

    data = Dataset(data_list,label_list,'train')
    dataloader = torch.utils.data.DataLoader(data,batch_size=256,shuffle=True,num_workers=32,pin_memory=True)



    dst_dir = './temp_weights'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    optimizer_ft = optim.SGD(model.parameters(), lr=0.01, momentum=0.9,weight_decay=0.0001)

    # scheduler = lr_scheduler.CosineAnnealingLR(optimizer_ft,5000,1e-6)
    scheduler = lr_scheduler.OneCycleLR(optimizer_ft, max_lr=0.01, steps_per_epoch=len(dataloader), epochs=15)


    # scheduler = lr_scheduler.MultiStepLR(optimizer_ft,[5,10,12,14], gamma=0.5)

    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    model.to(device)
    model = nn.DataParallel(model)
    model.train()



    print('start train')
    total_iter = 0
    running_loss = []
    for epoch_id  in range(epoch):

        iter_id = -1
        for imgs,flows,labels,index_list in dataloader:
            imgs = imgs.to(device)
            flows = flows.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer_ft.zero_grad()

            iter_id += 1
            total_iter += 1

            with torch.set_grad_enabled(True):
                outputs = model(imgs,flows)
                loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)

                loss.backward()
                optimizer_ft.step()

            lr = optimizer_ft.param_groups[0]['lr']
            running_loss.append(loss.item())
            if len(running_loss) > 150:
                _ = running_loss.pop()

            if iter_id % 20 == 0:
                time_info = time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))
                print_info = '{}  epoch : {}; lr: {:.6f} iter : {}/{} ; loss : {:.4f}'.format(time_info,epoch_id,lr,iter_id,len(dataloader),loss.item())

                print(print_info)

            scheduler.step()

            if total_iter % 150 == 0 and total_iter > 0:
                epoch_loss = np.mean(running_loss)

        # scheduler.step(epoch_id)
        if epoch_id % 3 == 0:
            model_name = '{}_{}.params'.format(model_prefix,epoch_id)
            torch.save(
                model.state_dict(), os.path.join(dst_dir,model_name))

    return model



def inference(model,data_list,label_list):
    dst_dir = './temp_weights'
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model = nn.DataParallel(model)
    model.eval()

    data = Dataset(data_list,label_list,'test')
    dataloader = torch.utils.data.DataLoader(data,batch_size=128,shuffle=False,num_workers=32,pin_memory=True)

    all_ans = []
    for imgs,flows,labels,index_list in dataloader:
        imgs = imgs.to(device)
        flows = flows.to(device)
        labels = labels.detach().cpu().numpy()

        with torch.set_grad_enabled(False):
            outputs = model(imgs,flows)
            outputs = outputs.detach().cpu().numpy()

            for index,output in zip(index_list,outputs):
                data_path = dataloader.dataset.data_list[index]
                label = dataloader.dataset.label_list[index]
                pred_label = np.argmax(output)
                info = [data_path,label,output[0],output[1],pred_label]
                all_ans.append(info)

    return all_ans



if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.zeros(1,3,244,244).to(device)

    model = MultiModel().to(device)
    print(model(x,x))