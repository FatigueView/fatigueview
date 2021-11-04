# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/11/7 12:29 AM
=================================================='''

from sklearn.svm import LinearSVC,SVC
import os
import numpy as np
from collections import Counter
import random
import pickle
import json

def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix)]


def min_is_nodding(x_list,threshold):
    if sum(v!=-1 for v in x_list) == 0 :
        return 0
    if x_list[len(x_list)//2] == -1:
        return 0
    _x = x_list[len(x_list)//2]
    x_list = [v for v in x_list if v != -1]

    if max(x_list) - min(x_list) > threshold and _x in [max(x_list) ,min(x_list)]:
        return 1

    return 0


def is_nodding(x_list,half_frame_len = 8,threshold = 4):
    ans = []
    for i in range(half_frame_len, len(x_list) - half_frame_len):
        ans.append(min_is_nodding(x_list[i-half_frame_len:i+half_frame_len],threshold))
    return sum(ans)



def load_data(data_list):
    all_data = []
    for data_path in data_list:
        with open(data_path,'r') as f:
            data = json.load(f)

            x_list = [v[0] for v in data[:-1]]
            y_list = [v[1] for v in data[:-1]]

            nodding_list = []
            for half_frame_len in range(8,16,2):
                for threshold in range(4,10):
                    nodding_list.append(is_nodding(x_list,half_frame_len,threshold))
                    nodding_list.append(is_nodding(y_list, half_frame_len, threshold))
            nodding_list.append(data[-1])
        all_data.append(nodding_list)
    all_data  = np.array(all_data)
    x = all_data[:,:-1]
    y = all_data[:,-1]
    return x,y


def data_balance(x,y):
    all_labels = y.tolist()
    label_dict = Counter(all_labels)
    max_len = max(label_dict.values())

    new_data = []
    new_label = []
    for label in label_dict:
        temp_data = x[y==label]
        for _ in range(max_len//len(temp_data)):
            new_data.append(temp_data)
            new_label.extend([label for _ in range(len(temp_data))])

        new_data.append(temp_data[:max_len%len(temp_data)])
        new_label.extend([label for _ in range(max_len%len(temp_data))])

    new_data = np.concatenate(new_data,axis = 0)
    index = list(range(len(new_data)))
    random.shuffle(index)

    new_data = new_data[index]

    new_label = np.array(new_label)
    new_label = new_label[index]

    return new_data,new_label






if __name__ == '__main__':

    src_dir_dict = {'train':'./data/train',
                    'sabao_test':'./data/sabao_test'
    }


    weights = '/home/users/zhenyu.yang/data/FatigueBaseLine/Kawato2000RDO/weights'
    if not os.path.exists(weights):
        os.makedirs(weights)

    camera_list = ['ir_down','ir_front','ir_left','ir_left_up','ir_up','rgb_down','rgb_front','rgb_left','rgb_left_up','rgb_up']
    # camera_list = ['ir_down','ir_front']


    stage = 'train'
    camera_id  = 0

    with open('sabao_test.txt', 'w') as f:
        f.write('Fatigue View\n')



    for camera_id in range(len(camera_list)):

        # train_data_dir = os.path.join(src_dir_dict['test'],camera_list[camera_id])
        # train_data_list = getFiles(train_data_dir,'.json')
        # # train_data_list = train_data_list[:5]
        # train_x,train_y = load_data(train_data_list)
        # train_x,train_y = data_balance(train_x,train_y)
        #
        #
        #
        #
        #
        # clf = SVC(kernel='linear',random_state=0, tol=1e-5)
        # clf.fit(train_x,train_y)



        model_name = '{}.model'.format(camera_list[camera_id])
        with open(os.path.join(weights, model_name), "rb") as f:
            clf = pickle.loads(f.read())


        cof = clf.coef_
        x_cof = [cof[0,i] for i in range(0,len(cof[0]),2)]
        y_cof = [cof[0, i] for i in range(1, len(cof[0]), 2)]
        # cof_sum = np.sum(cof)
        # clf.coef_ = clf.coef_ * 0
        #
        # clf.coef_[2*np.argmax(x_cof)] = cof_sum/2
        # clf.coef_[2 * np.argmax(y_cof) + 1] = cof_sum / 2
        # print(np.argmax(x_cof),np.argmax(y_cof))
        # print(train_x[:,np.argmax(x_cof)*2])

        x_index = np.argmax(x_cof)*2
        y_index = np.argmax(y_cof)*2 + 1

        # print(train_x[:,x_index])
        # print(train_x[:, y_index])


        test_data_dir = os.path.join(src_dir_dict['sabao_test'],camera_list[camera_id])
        test_data_list = getFiles(test_data_dir,'.json')
        test_x,test_y = load_data(test_data_list)
        test_x, test_y = data_balance(test_x, test_y)

        pred = []
        for x in test_x:
            if x[x_index] >= 1 or x[y_index] >= 1:
                pred.append(1)
            else:
                pred.append(0)



        # pred = clf.predict(test_x)

        # model_name = '{}.model'.format(camera_list[camera_id])
        # with open(os.path.join(weights,model_name), "wb+") as f:
        #     f.write(pickle.dumps(clf))

        acc = sum(y_ == y for y_, y in zip(pred, test_y)) / (len(test_y) + 0.0001)
        precison = sum(y_ == y and y_ == 1for y_,y in zip(pred,test_y)) / (sum(y_ == 1 for y_ in pred) + 0.0001)
        recall = sum(y_ == y and y_ == 1 for y_,y in zip(pred,test_y)) / (sum(y == 1 for y in test_y) + 0.0001)


        with open('sabao_test.txt','a') as f:
            info = '{} :acc {:.3f} ; precison:{:.3f}; recall: {:.3f},  data len : - {} \n'.format(camera_list[camera_id],acc,precison,recall,len(test_x))
            print(info)
            f.write(info)


