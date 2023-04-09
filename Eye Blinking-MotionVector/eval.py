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
import json

def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix)]



def load_data(data_list):
    all_data = []
    for data_path in data_list:
        with open(data_path,'r') as f:
            data = json.load(f)
        all_data.append([data['left_eye_label'],data['right_eye_label'],data['raft_left_eye_label'],data['raft_right_eye_label'],data['label']])
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
                    'sanbao_test':'./data_only_eye/sanbao_test'
    }

    weights = './weights'
    if not os.path.exists(weights):
        os.makedirs(weights)

    camera_list = ['ir_down']

    stage = 'sanbao_test'
    camera_id  = 0

    with open('ans.txt', 'w') as f:
        f.write('Fatigue View\n')



    for camera_id in range(len(camera_list)):


        test_data_dir = os.path.join(src_dir_dict['sanbao_test'],camera_list[camera_id])
        test_data_list = getFiles(test_data_dir,'.json')


        test_x,test_y = load_data(test_data_list)
        test_x, test_y = data_balance(test_x, test_y)

        clf = SVC(random_state=0, tol=1e-5)
        clf.fit(test_x,test_y)
        pred = clf.predict(test_x)

        # pred = test_x[:,0]

        acc = sum(y_ == y for y_, y in zip(pred, test_y)) / (len(test_y) + 0.0001)
        precison = sum(y_ == y and y_ == 1for y_,y in zip(pred,test_y)) / (sum(y_ == 1 for y_ in pred) + 0.0001)
        recall = sum(y_ == y and y_ == 1 for y_,y in zip(pred,test_y)) / (sum(y == 1 for y in test_y) + 0.0001)



        with open('ans.txt','a') as f:
            info = '{} :acc {:.3f} ; precison:{:.3f}; recall: {:.3f},  data len : {} - {} \n'.format(camera_list[camera_id],acc,precison,recall,len(test_x),len(test_x))
            print(info)
            f.write(info)


