# -*- coding: UTF-8 -*-
'''=================================================
@Author ：zhenyu.yang
@Date   ：2020/11/11 10:19 AM
=================================================='''

import os

def getFiles(path, suffix):
    return [os.path.join(root, file) for root, dirs, files in os.walk(path)
            for file in files if file.endswith(suffix)]



if __name__ == '__main__':
    src_dir = 'data/train/'
    dst_txt = './data/label_train.txt'

    src_dir = 'data/test'
    dst_txt = './data/label_test.txt'

    src_dir = 'data/beelab_train/'
    dst_txt = './data/beelab_train.txt'

    src_dir = 'data/beelab_test/'
    dst_txt = './data/beelab_test.txt'


    # src_dir = 'data/sanbao_test/'
    # dst_txt = './data/sanbao_test.txt'



    video_list =  getFiles(src_dir,'.jpg')

    video_list = [ os.path.join(*v.split(os.sep)[:-1]) for v in video_list]
    video_list = list(set(video_list))


    with open(dst_txt,'w') as f:
        for video in video_list:
            frame_len = str(len(os.listdir(video)))
            temp_path = video.split(src_dir)[-1]
            video_name = os.path.basename(video)
            label = video_name[0]

            info = ' '.join([temp_path,frame_len,label])
            f.write(info+'\n')








