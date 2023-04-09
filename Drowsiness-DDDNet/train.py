# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/11/7 12:29 AM
=================================================='''
import sys
sys.path.append('./')
sys.path.insert(0,'/data/zhenyu.yang/modules')

import pickle
import torch
import os
import numpy as np
from collections import Counter
import random
try:
    from .model import MultiModel,train,inference
except:
    from model import MultiModel, train, inference

os.environ["CUDA_VISIBLE_DEVICES"] = '1,2,3'


def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix)]



def load_data(data_list):

    all_data = []
    for data_path in data_list:
        with open(data_path,'r') as f:
            data = list(map(float,f.read().strip().split()))
        all_data.append(data)
    all_data  = np.array(all_data)
    x = all_data[:,:-2]
    y = all_data[:,-2]
    return x,y



def data_balance(x,y,dst_size=None):
    x = np.array(x)
    y = np.array(y)

    if dst_size is None:
        dst_size = 100000

    all_labels = y.tolist()
    label_dict = Counter(all_labels)
    # max_len = max(label_dict.values())
    max_len = dst_size//len(set(all_labels))


    new_data = []
    new_label = []
    for label in label_dict:
        temp_data = x[y==label]
        for _ in range(max_len//len(temp_data)):
            new_data.append(temp_data)
            new_label.extend([label for _ in range(len(temp_data))])

        leave_num = max_len%len(temp_data)
        if leave_num> 0:
            index_list = list(range(leave_num))
            random.shuffle(index_list)
            temp_leave_data = temp_data[index_list]
            new_data.append(temp_leave_data)
            new_label.extend([label for _ in range(leave_num)])


    new_data = np.concatenate(new_data,axis = 0)
    index = list(range(len(new_data)))
    random.shuffle(index)

    new_data = new_data[index]

    new_label = np.array(new_label)
    new_label = new_label[index]

    return new_data,new_label






if __name__ == '__main__':

    src_dir_dict = {'train':'/mnt/data-1/zhenyu.yang/FatigueViewBaseline/Reddy2017RDD/data_new/train',
                    'test':'/mnt/data-1/zhenyu.yang/FatigueViewBaseline/Reddy2017RDD/data/test'
    }
    weights = '/mnt/data-1/zhenyu.yang/FatigueViewBaseline/Reddy2017RDD/weights'
    camera_list = ['ir_up','rgb_front','rgb_left_up','rgb_up']

    # camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']


    #
    # src_dir_dict = {'train':'./data/train',
    #                 'test':'./data/test'
    # }
    # weights = './weights'
    # camera_list = ['rgb_down','rgb_left']




    if not os.path.exists(weights):
        os.makedirs(weights)



    stage = 'train'
    camera_id  = 0



    with open('ans.txt', 'w') as f:
        f.write('Fatigue View\n')



    for camera_id in range(len(camera_list)):

        train_data_dir = os.path.join(src_dir_dict['train'],camera_list[camera_id])
        train_data_list = getFiles(train_data_dir,'eye.jpg')
        label_list = [int(os.path.basename(v)[0]) for v in train_data_list]
        train_x,train_y = data_balance(train_data_list,label_list)
        model = MultiModel()
        model = train(model,train_x,train_y,25,camera_list[camera_id])


        model_name = '{}.parmas'.format(camera_list[camera_id])
        torch.save(model.state_dict(), os.path.join(weights, model_name))


        # test_data_dir = os.path.join(src_dir_dict['test'],camera_list[camera_id])
        # test_data_list = getFiles(test_data_dir,'.jpg')
        # label_list = [int(os.path.basename(v)[0]) for v in test_data_list]
        # test_x,test_y = data_balance(test_data_list,label_list)
        # pred_info = inference(model,test_x,test_y)
        #
        #
        # acc = sum(y_[-1] == y for y_, y in zip(pred_info, test_y)) / (len(test_y) + 0.0001)
        # precison = sum(y_ [-1]== y and y == 1for y_,y in zip(pred_info,test_y)) / (sum(y_[-1] == 1 for y_ in pred_info) + 0.0001)
        # recall = sum(y_[-1] == y and y == 1 for y_,y in zip(pred_info,test_y)) / (sum(y == 1 for y in test_y) + 0.0001)
        #
        #
        # with open('ans.txt','a') as f:
        #     info = '{} :acc {:.3f} ; precison:{:.3f}; recall: {:.3f},  data len : {} - {} \n'.format(camera_list[camera_id],acc,precison,recall,len(train_x),len(test_x))
        #     print(info)
        #     f.write(info)

