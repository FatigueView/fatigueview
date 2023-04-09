# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/11/7 12:29 AM
=================================================='''
import pickle
from sklearn.svm import LinearSVC,SVC
import os
import numpy as np
from collections import Counter
import random

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


def load_single_slice(data_path):
    all_data = []

    with open(data_path, 'r') as f:
        temp_all_data = f.readlines()

    max_len = 6000
    temp_all_data = temp_all_data[:min(max_len,len(temp_all_data))]


    for data in temp_all_data:
        data = data.strip().split()
        if len(data) < 2:
            continue
        data = list(map(float,data))
        all_data.append(data)

    all_data  = np.array(all_data)
    x = all_data[:,:-2]
    y = all_data[:,-2:]
    return  x,y


def finetune(pred,status=1):
    for i in range(1,len(pred)-1):
        if pred[i] != status and pred[i-1] == status and pred[i+1] == status:
            pred[i] = status
    return pred


def num2list(blink_list,status_name=1):
    blink_list = [v == status_name for v in blink_list]
    blink_list.insert(0,False)
    blink_list.insert(-1,False)

    start_num = [i-1 for i in range(1,len(blink_list)) if not blink_list[i-1] and blink_list[i]]

    blink_list = blink_list[::-1]
    end_num = [len(blink_list) - i+1 for i in range(1,len(blink_list)) if not blink_list[i-1] and blink_list[i]]
    end_num = end_num[::-1]
    num_list = [[start,end] for  start,end in zip(start_num,end_num)]
    return num_list


def eval_pr(iou,threshold):
    p_len,l_len = iou.shape[:2]
    iou = np.max(iou,axis = 1)
    TP = np.sum(iou >= threshold)

    return TP,p_len,l_len

def line_IOU(pred,label):
    p_len = len(pred)
    l_len = len(label)
    PR = np.zeros((9,3))
    if p_len == 0 or l_len == 0:
        PR[:,1] = p_len
        PR[:, 2] = l_len
        return PR

    iou = np.zeros((p_len,l_len))
    for i,pred_slice in enumerate(pred):
        for j,label_slice in enumerate(label):

            temp_left = max(pred_slice[0], label_slice[0])
            temp_right = min(pred_slice[1], label_slice[1])

            if temp_left >= temp_right:
                continue

            pred_slice_len = abs(pred_slice[1] - pred_slice[0])
            label_slice_len = abs(label_slice[1] - label_slice[0])

            iou[i,j] = (temp_right - temp_left) / (min(pred_slice_len,label_slice_len) + 1e-5)

    for i in range(1,10):
        threshold = i/10
        PR[i-1] = eval_pr(iou,threshold)

    return PR




def eval_single_slice(pred,label):
    label[label==2] = 1
    pred_slice = num2list(finetune(pred,1))
    label_slice = num2list(finetune(label,1))
    return line_IOU(pred_slice,label_slice)



def analysis_PR(PR):
    PR = PR.copy()
    PR = PR + 0.0001
    precison = PR[:,0]/PR[:,1]
    recall = PR[:,0]/PR[:,2]

    pr_list = [[precison[i],recall[i]] for i in range(len(PR))]
    pr_list.sort(key=lambda x:x[1])
    p_list = [v[0] for v in pr_list]
    pr_list = [[max(p_list[i:]),pr_list[i][1]] for i in range(len(pr_list))]
    pr_list.insert(0,[1,0])
    pr_list.insert(-1,[0, 1])


    # ap = 0
    # for i in range(1,len(pr_list)):
    #     ap += (pr_list[i][1] - pr_list[i-1][1])*(pr_list[i][0] - pr_list[i-1][0])
    # ap /= 2

    ans = []
    for i in range(5,10):
        threshold = i/10
        pre = precison[i-1]
        rec = recall[i-1]
        info = 'threshold: {:.2f}; precision: {:.3f}; recall: {:.3f}'.format(threshold,pre,rec)
        ans.append(info)

    pre = np.mean(precison[4:])
    rec = np.mean(recall[4:])
    info = 'threshold: ---; precision: {:.3f}; recall: {:.3f}'.format(pre, rec)
    ans.append(info)
    # info = 'mAP: {:.3f}'.format(ap)
    # ans.append(info)

    return ans


if __name__ == '__main__':

    src_dir_dict = {'train':'./data/train',
                    'test':'./data_new/test'
    }

    weights = './weights'
    if not os.path.exists(weights):
        os.makedirs(weights)

    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']


    with open('loc_ans.txt', 'w') as f:
        f.write('Fatigue View\n')

    test_data_list = getFiles(src_dir_dict['test'], '.txt')
    for camera_id in range(len(camera_list)):

        camera_type = camera_list[camera_id]
        temp_data_list = [v for v in test_data_list if camera_type == v.split('__')[-2]]
        model_name = '{}.model'.format(camera_list[camera_id])

        with open(os.path.join(weights,model_name), "rb") as f:
            clf = pickle.loads(f.read())


        PR = np.zeros((9, 3))
        for data_path in test_data_list:
            test_x, test_y = load_single_slice(data_path)
            pred = clf.predict(test_x)
            label = test_y[:,-1]

            PR += eval_single_slice(pred,label)

            info = analysis_PR(PR)
            print('\n'.join(info))


        info = analysis_PR(PR)
        info.insert(0,camera_list[camera_id])
        info = '\n'.join(info)

        print(info)

        with open('ans.txt','a') as f:
            f.write(info)
            f.write('\n')
            f.write('****************')
            f.write('\n')


