import shutil
import os

src_dir = '/home/users/zhenyu.yang/projects/FatigueBaseLine/Wang2017TSN/data/beelab_test'
dst_dir = '/home/users/zhenyu.yang/projects/FatigueBaseLine/Wang2017TSN/data/beelab_test_bak'
if not os.path.exists(dst_dir):
    os.makedirs(dst_dir)


slim_ratio = 0.5

cameras = os.listdir(src_dir)
for camera in cameras:
    camera_path = os.path.join(src_dir,camera)
    video_list = os.listdir(camera_path)
    video_list = [os.path.join(camera_path,v) for v in video_list]
    video_list.sort()

    temp_dst_dir = os.path.join(dst_dir,camera)
    if not os.path.exists(temp_dst_dir):
        os.makedirs(temp_dst_dir)
    for video in video_list[:int(len(video_list)*slim_ratio/2)]:
        shutil.move(video,temp_dst_dir)

    for video in video_list[-int(len(video_list)*slim_ratio/2):]:
        shutil.move(video,temp_dst_dir)