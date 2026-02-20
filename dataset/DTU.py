import numpy as np
import scipy
from scipy import io
import os
import scipy.io.matlab.mio5_utils
from torch.utils.data import DataLoader, Subset, Dataset

# Sub = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
Sub = [1,2,3]
def data_read_feature_cs(nSub, feature_name='envelope', window=1, overlap=1, condition=None, eeg_data=None, stimulus_data=None, layer_list=None):
        test_Sub = [nSub]
        train_Sub = list(set(Sub) - set(test_Sub))

        # window = 1
        # sample_rate = 128
        # train_data = np.empty((0, 360, window*sample_rate, 64))
        # train_label = np.empty((0, 1, 360))
        train_data = {}
        train_label = {}
        stimulus_total = {}

        for i in range(len(train_Sub)):
            print(f"Sub:{train_Sub[i]}")
            data, label, stim= data_read_feature(train_Sub[i], feature_name, window, overlap, condition, eeg_data, stimulus_data, layer_list)
            #   train_data = np.concatenate((train_data, data), axis=0)
            #   train_label = np.concatenate((train_label, label), axis=0)
            train_data.update(data)
            train_label.update(label)
            stimulus_total.update(stim)

        print(test_Sub[0])
        test_data, test_label, stim = data_read_feature(test_Sub[0], feature_name, window, overlap, condition, eeg_data, stimulus_data, layer_list)
        stimulus_total.update(stim)
        return train_data, train_label, test_data, test_label, stimulus_total


def rename(original, name):

    # 找到最后一个下划线的位置
    last_underscore_index = original.rfind('.')

    # 提取最后一个下划线之前的内容
    part_before_last_underscore = original[:last_underscore_index]

    # 创建新的字符串，后缀替换为 'envelope.npy'
    new_string = part_before_last_underscore + '_' + name

    # 输出结果
    return new_string

def data_read_stimulus_path(stimulus_name, stimulus_path, layer_list = None):
    # path = '/media/nchen/CYB/EEGViT/stimulus/'
    path = stimulus_path
    stimulus_path = os.path.join(path, stimulus_name + '.npy')
    data = np.load(stimulus_path)
    if layer_list is not None:
        data = data[:, layer_list, :]
    return data

def data_read_feature(nSub, feature_name='envelope', window_length=1, overlap=1, condition=None, eeg_data=None, stimulus_data=None, layer_list=None):
    # if condition == 'dry' or condition == 'hrtf':
    #     trials = 20
    # elif condition == 'all':
    #     trials = 8
    trials = 60
    channels = 64
    window = window_length
    sample_rate = 128
    overlap = overlap  # 0.5
    # data = np.empty((0, 360, window*sample_rate, 64))
    data = {}
    # root = '/media/nchen/CYB/KUL/preprocessed_data_50hz_mean/'
    root = eeg_data
    stimulus_path = stimulus_data
    label = {}
    stimulus_total = {}
    nSub = nSub
    condition = condition
    # train data
    data_mat = scipy.io.loadmat(root + 'S%d_data_preproc.mat'% nSub)
    label_data = scipy.io.loadmat(root + 'S%d_label.mat' % nSub)
    for k_tra in range(trials):
        # if data_mat['preproc_trials'][0, k_tra]['condition'] == condition or condition == 'all':
        label_direction = []
        label_stimulus = []
        mat_data = data_mat['data']['eeg'][0, 0][0, k_tra][:, :64]
        stimulus = [label_data['wav_male'][k_tra, 0], label_data['wav_female'][k_tra, 0]]
        table = str.maketrans('', '', "'[]")
        stimulus1 = str(stimulus[0]).translate(table)
        stimulus2 = str(stimulus[1]).translate(table)
        stimulus1_name = rename(stimulus1, feature_name)
        stimulus2_name = rename(stimulus2, feature_name)
        trial_data = mat_data.reshape((1, -1, channels))
        segm_data = np.empty((0, int(window * sample_rate), 64))
        segment = int(((trial_data.shape[1] / sample_rate)-(window_length-1)) / (window * overlap))
        print(segment)
        # trail_data = np.zeros((1, mat_data.shape[0], channels))
        for j in range(segment-1):
            if overlap != 1:
                    start = int(overlap*sample_rate*j)
            else: 
                    start = int(j*sample_rate*window)
            # start = int(overlap * sample_rate * j)
            end = int(start + sample_rate * window)
            segment_data = trial_data[0][start:end][:].reshape((1, -1, channels))
            segm_data = np.concatenate((segm_data, segment_data), axis=0)

        mat_label = int(data_mat['data']['event'][0,0][0,0][0][0, k_tra][1])#[2][1][1]#[k_tra]#['value']
        # stimulus_label = int(data_mat['preproc_trials'][0, k_tra]['attended_track'][0, 0])
        # if mat_label[0] == 'L':
        #     label_direction.append([0] * segment)
        # elif mat_label[0] == 'R':
        #     label_direction.append([1] * segment)
        label_stimulus.append([mat_label] * segment)

        segm_data = np.expand_dims(segm_data, axis=0)
        # data = np.concatenate((data, segm_data), axis=0)
        data[f'Sub{nSub}Trial{k_tra + 1}'] = [segm_data, stimulus1_name, stimulus2_name]
        # label[f'Sub{nSub}Trial{k_tra + 1}_direction'] = np.array(label_direction)
        label[f'Sub{nSub}Trial{k_tra + 1}_stimulus'] = np.array(label_stimulus)
        print(stimulus1_name)
        print(stimulus2_name)
        if stimulus1_name not in stimulus_total:
            stimulus1_data = data_read_stimulus_path(stimulus1_name, stimulus_path, layer_list=layer_list)
            stimulus_total[f'{stimulus1_name}'] = stimulus1_data
        if stimulus2_name not in stimulus_total:
            stimulus2_data = data_read_stimulus_path(stimulus2_name, stimulus_path, layer_list = layer_list)
            stimulus_total[f'{stimulus2_name}'] = stimulus2_data

            # data.append(segm_data)
            # label.append(label_data)
    return data, label, stimulus_total


def data_read_CL_feature_path_DTU(nSub, feature_name='envelope', window_length=1, overlap=1, condition=None, eeg_data=None, stimulus_data=None):
    # if condition == 'dry' or condition == 'hrtf':
    #     trials = 20
    # elif condition == 'all':
    #     trials = 8
    trials = 60
    channels = 64
    window = window_length
    sample_rate = 128
    overlap = overlap  # 0.5
    # data = np.empty((0, 360, window*sample_rate, 64))
    data = {}
    # root = '/media/nchen/CYB/KUL/preprocessed_data_50hz_mean/'
    root = eeg_data
    stimulus_path = stimulus_data
    label = {}
    stimulus_total = {}
    nSub = nSub
    condition = condition
    # train data
    data_mat = scipy.io.loadmat(root + 'S%d_data_preproc.mat'% nSub)
    label_data = scipy.io.loadmat(root + 'S%d_label.mat' % nSub)
    for k_tra in range(trials):
        # if data_mat['preproc_trials'][0, k_tra]['condition'] == condition or condition == 'all':
        label_direction = []
        label_stimulus = []
        mat_data = data_mat['data']['eeg'][0, 0][0, k_tra][:, :64]
        stimulus = [label_data['wav_male'][k_tra, 0], label_data['wav_female'][k_tra, 0]]
        table = str.maketrans('', '', "'[]")
        stimulus1 = str(stimulus[0]).translate(table)
        stimulus2 = str(stimulus[1]).translate(table)
        stimulus1_name = rename(stimulus1, feature_name)
        stimulus2_name = rename(stimulus2, feature_name)
        trial_data = mat_data.reshape((1, -1, channels))
        segm_data = np.empty((0, int(window * sample_rate), 64))
        segment = int(((trial_data.shape[1] / sample_rate)-(window_length-1)) / (window * overlap))
        print(segment)
        # trail_data = np.zeros((1, mat_data.shape[0], channels))
        for j in range(segment-1):
            if overlap != 1:
                    start = int(overlap*sample_rate*j)
            else: 
                    start = int(j*sample_rate*window)
            # start = int(overlap * sample_rate * j)
            end = int(start + sample_rate * window)
            segment_data = trial_data[0][start:end][:].reshape((1, -1, channels))
            segm_data = np.concatenate((segm_data, segment_data), axis=0)

        mat_label = int(data_mat['data']['event'][0,0][0,0][0][0, k_tra][1])#[2][1][1]#[k_tra]#['value']
        # stimulus_label = int(data_mat['preproc_trials'][0, k_tra]['attended_track'][0, 0])
        # if mat_label[0] == 'L':
        #     label_direction.append([0] * segment)
        # elif mat_label[0] == 'R':
        #     label_direction.append([1] * segment)
        label_stimulus.append([mat_label] * segment)

        segm_data = np.expand_dims(segm_data, axis=0)
        # data = np.concatenate((data, segm_data), axis=0)
        data[f'Sub{nSub}Trial{k_tra + 1}'] = [segm_data, stimulus1_name, stimulus2_name]
        # label[f'Sub{nSub}Trial{k_tra + 1}_direction'] = np.array(label_direction)
        label[f'Sub{nSub}Trial{k_tra + 1}_stimulus'] = np.array(label_stimulus)
        print(stimulus1_name)
        print(stimulus2_name)
        if stimulus1_name not in stimulus_total:
            stimulus1_data = data_read_stimulus_path(stimulus1_name, stimulus_path)
            stimulus_total[f'{stimulus1_name}'] = stimulus1_data
        if stimulus2_name not in stimulus_total:
            stimulus2_data = data_read_stimulus_path(stimulus2_name, stimulus_path)
            stimulus_total[f'{stimulus2_name}'] = stimulus2_data

            # data.append(segm_data)
            # label.append(label_data)
    return data, label, stimulus_total

if __name__ == '__main__':
    _, _, _ = data_read_CL_feature_path_DTU(1, 'wavlm_24layer_1s', eeg_data='E:/DTU/New_data1/', stimulus_data='D:/EEGViT/stimulus_DTU/')