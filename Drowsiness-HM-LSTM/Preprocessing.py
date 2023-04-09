import os
import numpy as np
import json
from tqdm import tqdm
from collections import OrderedDict

def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

def normalize_blinks(num_blinks, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur, u_Dur, sigma_Dur, Vel, u_Vel,
                     sigma_Vel):
    # input is the blinking features as well as their mean and std, output is a [num_blinksx4] matrix as the normalized blinks
    eps = 1e-8
    normalized_blinks = np.zeros([num_blinks, 4])
    normalized_Freq = (Freq[0:num_blinks] - u_Freq) / (sigma_Freq + eps)
    normalized_blinks[:, 0] = normalized_Freq
    normalized_Amp = (Amp[0:num_blinks]  - u_Amp) / (sigma_Amp + eps)
    normalized_blinks[:, 1] = normalized_Amp
    normalized_Dur = (Dur[0:num_blinks]  - u_Dur) / (sigma_Dur + eps)
    normalized_blinks[:, 2] = normalized_Dur
    normalized_Vel = (Vel[0:num_blinks]  - u_Vel) / (sigma_Vel + eps)
    normalized_blinks[:, 3] = normalized_Vel

    return normalized_blinks

def normalize_per_blink(num_blinks, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur, u_Dur, sigma_Dur, Vel, u_Vel,
                     sigma_Vel):
    # input is the blinking features as well as their mean and std, output is a [num_blinksx4] matrix as the normalized blinks
    eps = 1e-8
    normalized_blinks = np.zeros([num_blinks, 4])
    normalized_Freq = (Freq[0:num_blinks] - u_Freq) / (sigma_Freq + eps)
    normalized_blinks[:, 0] = normalized_Freq
    normalized_Amp = (Amp[0:num_blinks]  - u_Amp) / (sigma_Amp + eps)
    normalized_blinks[:, 1] = normalized_Amp
    normalized_Dur = (Dur[0:num_blinks]  - u_Dur) / (sigma_Dur + eps)
    normalized_blinks[:, 2] = normalized_Dur
    normalized_Vel = (Vel[0:num_blinks]  - u_Vel) / (sigma_Vel + eps)
    normalized_blinks[:, 3] = normalized_Vel

    return normalized_blinks


def unroll_in_time(in_data, window_size):
    # in_data is [n,4]            out_data is [1,Window_size,4]
    n = len(in_data)
    if n <= window_size:
        out_data = np.zeros([1, window_size, 4])
        out_data[0, -n:, :] = in_data
        return out_data
    else:
        out_data = np.zeros([1, window_size, 4])
        out_data[0, :, :] = in_data[:window_size, :]
        return out_data


def Preprocess(data,window_size, phase):
    output = None
    labels = None
    u_Freq, sigma_Freq = 0, 1
    u_Amp, sigma_Amp = 0, 1
    u_Dur, sigma_Dur = 0, 1
    u_Vel, sigma_Vel = 0, 1
    for name in tqdm(data.keys()):

        alert_blink_unrolled = None
        alert_labels = None
        if 0 in data[name].keys():
            alert_data = np.concatenate(data[name][0], axis=0)
            Freq = alert_data[:,3]
            Amp = alert_data[:,1]
            Dur = alert_data[:,0]
            Vel = alert_data[:,2]
            blink_num=len(Freq)
            bunch_size=blink_num // 3   #one third used for baselining
            remained_size=blink_num-bunch_size

            # Using the last bunch_size number of blinks to calculate mean and std
            u_Freq=np.mean(Freq[-bunch_size:])
            sigma_Freq=np.std(Freq[-bunch_size:])
            if sigma_Freq==0:
                sigma_Freq=np.std(Freq)
            u_Amp=np.mean(Amp[-bunch_size:])
            sigma_Amp=np.std(Amp[-bunch_size:])
            if sigma_Amp==0:
                sigma_Amp=np.std(Amp)
            u_Dur=np.mean(Dur[-bunch_size:])
            sigma_Dur=np.std(Dur[-bunch_size:])
            if sigma_Dur==0:
                sigma_Dur=np.std(Dur)
            u_Vel=np.mean(Vel[-bunch_size:])
            sigma_Vel=np.std(Vel[-bunch_size:])
            if sigma_Vel==0:
                sigma_Vel=np.std(Vel)

            # print('freq: %f, amp: %f, dur: %f, vel: %f \n' %(u_Freq,u_Amp,u_Dur,u_Vel))
            # normalized_blinks=normalize_blinks(remained_size, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur, u_Dur, sigma_Dur,
            #                     Vel, u_Vel, sigma_Vel)
            # print('Postfreq: %f, Postamp: %f, Postdur: %f, Postvel: %f \n' % (np.mean(normalized_blinks[:,0]),
            #                                                                     np.mean(normalized_blinks[:,1]),
            #                                                                     np.mean(normalized_blinks[:,2]),
            #                                                                     np.mean(normalized_blinks[:,3])))

            normalized = []
            for sample in data[name][0]:
                normalized.append(normalize_per_blink(sample.shape[0], sample[:,3], u_Freq, sigma_Freq, sample[:,1], u_Amp, sigma_Amp, sample[:,0], u_Dur, sigma_Dur,
                                sample[:,2], u_Vel, sigma_Vel))
            
            alert_blink_unrolled=np.concatenate([unroll_in_time(sample, window_size) for sample in normalized],axis=0)
            # sweep a window over the blinks to chunk
            alert_labels = 0 * np.ones([len(alert_blink_unrolled), 1])


        sleepy_blink_unrolled = None
        sleepy_labels = None
        if 1 in data[name].keys():
            # fatigue_data = np.concatenate(data[name][1], axis=0)
            # Freq = fatigue_data[:,3]
            # Amp = fatigue_data[:,1]
            # Dur = fatigue_data[:,0]
            # Vel = fatigue_data[:,2]
            # blink_num = len(Freq)

            # normalized_blinks = normalize_blinks(blink_num, Freq, u_Freq, sigma_Freq, Amp, u_Amp, sigma_Amp, Dur,
            #                                         u_Dur, sigma_Dur, Vel, u_Vel, sigma_Vel)
            # print('SLEEPYfreq: %f, SLEEPYamp: %f, SLEEPYdur: %f, SLEEPYvel: %f \n'  % (np.mean(normalized_blinks[:,0]),
            #                                                                     np.mean(normalized_blinks[:,1]),
            #                                                                     np.mean(normalized_blinks[:,2]),
            #                                                                     np.mean(normalized_blinks[:,3])))

            normalized = []
            for sample in data[name][1]:
                normalized.append(normalize_per_blink(sample.shape[0], sample[:,3], u_Freq, sigma_Freq, sample[:,1], u_Amp, sigma_Amp, sample[:,0], u_Dur, sigma_Dur,
                                sample[:,2], u_Vel, sigma_Vel))
            
            sleepy_blink_unrolled=np.concatenate([unroll_in_time(sample, window_size) for sample in normalized],axis=0)
            sleepy_labels=10*np.ones([len(sleepy_blink_unrolled),1])


        if alert_blink_unrolled is None:
            tempX=sleepy_blink_unrolled
            tempY=sleepy_labels
        elif sleepy_blink_unrolled is None:
            if phase == 'train':
                tempX=np.concatenate((alert_blink_unrolled,alert_blink_unrolled,alert_blink_unrolled),axis=0)
                tempY=np.concatenate((alert_labels,alert_labels,alert_labels),axis=0)
            else:
                tempX = alert_blink_unrolled
                tempY = alert_labels
        else:
            if phase == 'train':
                tempX = np.concatenate((alert_blink_unrolled,alert_blink_unrolled,alert_blink_unrolled,sleepy_blink_unrolled),axis=0)
                tempY = np.concatenate((alert_labels,alert_labels,alert_labels, sleepy_labels), axis=0)
            else:
                tempX = np.concatenate((alert_blink_unrolled,sleepy_blink_unrolled),axis=0)
                tempY = np.concatenate((alert_labels, sleepy_labels), axis=0)

        if output is None:
            output=tempX
            labels=tempY
        else:
            output=np.concatenate((output,tempX),axis=0)
            labels=np.concatenate((labels,tempY),axis=0)

    if phase == 'train':
        output,labels=unison_shuffled_copies(output,labels)
    print('We have %d datapoints!!!'%len(labels))
    return output,labels

def get_data(path):
    data = OrderedDict()
    path_rec = []
    files = os.listdir(path)
    files.sort()
    for file in tqdm(files):
        name = file.split('_')[0]
        if 'test' in name:
            label = name = file.split('_')[3]
        if 'fatigue' in name:
            label = name = file.split('_')[1]
        with open(os.path.join(path, file)) as f:
            info = json.loads(f.read())
        label = info['label']
        if label == -1:
            continue
        if sum(info['blink_fea'][0]) == 0:
            continue
        if name not in data.keys():
            data[name] = {}
        if label not in data[name].keys():
            data[name][label] = []
        data[name][label].append(np.array(info['blink_fea']))
        path_rec.append(file + ' ' + str(label))


    return data,path_rec

if __name__ == "__main__":
    # poses = [
    #     'ir_down', 'ir_front', 'ir_left', 'ir_left_up', 'ir_up', 'rgb_down',
    #     'rgb_front', 'rgb_left', 'rgb_left_up', 'rgb_up'
    # ]
    poses = [
        'sanbao'
    ]
    for pos in poses:
        basepath = '/mnt/data-1/data/weiyu.li/tmp_dataprocessing4Ghoddoosian2019ARD/data_sanbao/'
        train_path = basepath + 'train/' + pos
        test_path = basepath + 'test/' + pos

        test_data,path_rec = get_data(test_path)
        train_data,_ = get_data(train_path)

        window_size=10


        blinks,labels=Preprocess(train_data, window_size, 'test')
        blinksTest,labelsTest=Preprocess(test_data, window_size, 'test')
        np.save(open('{}_train_samples.npy'.format(pos),'wb'),blinks)
        np.save(open('{}_train_labels.npy'.format(pos),'wb'),labels)
        np.save(open('{}_test_samples.npy'.format(pos),'wb'),blinksTest)
        np.save(open('{}_test_labels.npy'.format(pos),'wb'),labelsTest)
        with open('{}_samples.txt'.format(pos),'w')as f:
            for sample in path_rec:
                f.write(sample + '\n')
        print(pos)

