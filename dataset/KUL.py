import numpy as np
import scipy
from scipy import io
import os
from torch.utils.data import DataLoader, Subset, Dataset

def rename(original, name):

    # 找到最后一个下划线的位置  
    last_underscore_index = original.rfind('_')  

    # 提取最后一个下划线之前的内容  
    part_before_last_underscore = original[:last_underscore_index]  

    # 创建新的字符串，后缀替换为 'envelope.npy'  
    new_string = part_before_last_underscore + '_' + name 

    # 输出结果  
    return new_string

def get_speaker(original, speak_dict):
    # 找到最后一个下划线的位置  
    last_underscore_index = original.rfind('_')  

    # 提取最后一个下划线之前的内容  
    part_before_last_underscore = original[:last_underscore_index]

    speaker = speak_dict[part_before_last_underscore]

    return speaker

def data_read(nSub):
        trials,channels = 8,64
        window = 1
        sample_rate = 128
        overlap = 1  #0.5
        data = np.empty((0, 360, window*sample_rate, 64))
        root = '/media/nchen/CYB/KUL/preprocessed_data_50hz_mean/'
        label = []
        nSub = nSub
        # train data
        data_mat = io.loadmat(root + 'S%d.mat' % nSub)

        for k_tra in range(trials):
            label_data = []
            mat_data = data_mat['preproc_trials'][0, k_tra]['RawData'][0, 0]['EegData']
            trial_data = mat_data[0][0][:46080,:].reshape((1,-1,channels))
            segm_data = np.empty((0, window * sample_rate, 64))
            segment = int(360/(window*overlap))
            #trail_data = np.zeros((1, mat_data.shape[0], channels))
            for j in range(segment):
                start = int(overlap*sample_rate*j)
                end = int(start+sample_rate*window)
                segment_data = trial_data[0][start:end][:].reshape((1,-1,channels))
                segm_data = np.concatenate((segm_data,segment_data),axis=0)
            mat_label = data_mat['preproc_trials'][0, k_tra]['attended_ear']
            if mat_label[0] == 'L':
                label_data.append([0]*segment)
            elif mat_label[0] == 'R':
                label_data.append([1]*segment)

            segm_data = np.expand_dims(segm_data, axis=0)
            data = np.concatenate((data, segm_data), axis=0)
            #label = np.concatenate((label, label_data), axis=0)
            label.append(label_data)

        return data, np.array(label)  #(trial, segmant, window, channel) 

        
Sub = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16]
def data_read_cs(nSub):
        test_Sub = [nSub]
        train_Sub = list(set(Sub) - set(test_Sub))

        window = 1
        sample_rate = 128
        train_data = np.empty((0, 360, window*sample_rate, 64))
        train_label = np.empty((0, 1, 360))

        for i in range(len(train_Sub)):
              data, label = data_read(train_Sub[i])
              train_data = np.concatenate((train_data, data), axis=0)
              train_label = np.concatenate((train_label, label), axis=0)
        
        test_data, test_label = data_read(test_Sub[0])

        return train_data, train_label, test_data, test_label

def data_read_condition(nSub, condition, trials=20):
        trials,channels = trials,64
        window = 1
        sample_rate = 128
        overlap = 1  #0.5
        #data = np.empty((0, 360, window*sample_rate, 64))
        data = []
        root = '/media/nchen/CYB/KUL/preprocessed_data_50hz_mean/'
        label = []
        nSub = nSub
        condition = condition
        # train data
        data_mat = scipy.io.loadmat(root + 'S%d.mat' % nSub)

        for k_tra in range(trials):
            if data_mat['preproc_trials'][0,k_tra]['condition'] == condition:
                label_data = []
                mat_data = data_mat['preproc_trials'][0, k_tra]['RawData'][0, 0]['EegData']
                trial_data = mat_data[0][0][:,:].reshape((1,-1,channels))
                segm_data = np.empty((0, window * sample_rate, 64))
                segment = int((trial_data.shape[1]/sample_rate)/(window*overlap))
                #trail_data = np.zeros((1, mat_data.shape[0], channels))
                for j in range(segment):
                    start = int(overlap*sample_rate*j)
                    end = int(start+sample_rate*window)
                    segment_data = trial_data[0][start:end][:].reshape((1,-1,channels))
                    segm_data = np.concatenate((segm_data,segment_data),axis=0)
                mat_label = data_mat['preproc_trials'][0, k_tra]['attended_ear']
                if mat_label[0] == 'L':
                    label_data.append([0]*segment)
                elif mat_label[0] == 'R':
                    label_data.append([1]*segment)

                segm_data = np.expand_dims(segm_data, axis=0)
                #data = np.concatenate((data, segm_data), axis=0)
                data.append(segm_data)
                label.append(label_data)
            else:
                 True
        return data, label

def data_read_feature_cs(nSub, feature_name='envelope', window=1, overlap=1, condition=None, eeg_data=None, stimulus_data=None, layer_list=None):
        test_Sub = [nSub]
        train_Sub = list(set(Sub) - set(test_Sub))

        # window = 1
        # sample_rate = 128
        # train_data = np.empty((0, 360, window*sample_rate, 64))
        # train_label = np.empty((0, 1, 360))
        train_data = {}
        train_label = {}

        for i in range(len(train_Sub)):
            print(f"Sub:{train_Sub[i]}")
            data, label, _= data_read_feature(train_Sub[i], feature_name, window, overlap, condition, eeg_data, stimulus_data, layer_list)
            #   train_data = np.concatenate((train_data, data), axis=0)
            #   train_label = np.concatenate((train_label, label), axis=0)
            train_data.update(data)
            train_label.update(label)

        print(test_Sub[0])
        test_data, test_label, stimulus_total = data_read_feature(test_Sub[0], feature_name, window, overlap, condition, eeg_data, stimulus_data, layer_list)

        return train_data, train_label, test_data, test_label, stimulus_total

def data_read_stimulus(stimulus_name, stimulus_path, layer_list = None):
    # path = '/media/nchen/CYB/EEGViT/stimulus/'
    path = stimulus_path
    stimulus_path = os.path.join(path, stimulus_name + '.npy')
    data = np.load(stimulus_path)
    if layer_list is not None:
        data = data[:, layer_list, :]
    return data

def data_read_feature(nSub, feature_name='envelope', window=1, overlap=1, condition=None, eeg_data=None, stimulus_data=None, layer_list=None):
    if condition == 'dry' or condition == 'hrtf':
        trials = 20
    elif condition == 'all':
        trials = 8
    trials=8
    channels = 64
    window = window
    sample_rate = 128
    overlap = overlap  #0.5
    #data = np.empty((0, 360, window*sample_rate, 64))
    data = {}
    # root = '/media/nchen/CYB/KUL/preprocessed_data_50hz_mean/'
    root = eeg_data
    stimulus_path = stimulus_data
    label = {}
    stimulus_total = {}
    nSub = nSub
    condition = condition
    # train data
    data_mat = scipy.io.loadmat(root + 'S%d.mat' % nSub)

    for k_tra in range(trials):
        if data_mat['preproc_trials'][0,k_tra]['condition'] == condition or condition == 'all':
            label_direction = []
            label_stimulus = []
            mat_data = data_mat['preproc_trials'][0, k_tra]['RawData'][0, 0]['EegData']
            stimulus = data_mat['preproc_trials'][0, k_tra]['stimuli'][0,0][:,0]
            table = str.maketrans('','', "'[]")
            stimulus1 = str(stimulus[0]).translate(table)
            stimulus2 = str(stimulus[1]).translate(table)
            stimulus1_name = rename(stimulus1, feature_name)
            stimulus2_name = rename(stimulus2, feature_name)
            trial_data = mat_data[0][0][:,:].reshape((1,-1,channels))
            segm_data = np.empty((0, window * sample_rate, 64))
            segment = int(((trial_data.shape[1]/sample_rate) - (window-1))/(window*overlap))
            #trail_data = np.zeros((1, mat_data.shape[0], channels))
            for j in range(segment):
                start = int(overlap*sample_rate*j)
                end = int(start+sample_rate*window)
                segment_data = trial_data[0][start:end][:].reshape((1,-1,channels))
                segm_data = np.concatenate((segm_data,segment_data),axis=0)

            mat_label = data_mat['preproc_trials'][0, k_tra]['attended_ear']
            stimulus_label = int(data_mat['preproc_trials'][0, k_tra]['attended_track'][0,0])
            if mat_label[0] == 'L':
                label_direction.append([0]*segment)
            elif mat_label[0] == 'R':
                label_direction.append([1]*segment)
            label_stimulus.append([stimulus_label] * segment)

            segm_data = np.expand_dims(segm_data, axis=0)
            #data = np.concatenate((data, segm_data), axis=0)
            data[f'Sub{nSub}Trial{k_tra+1}'] = [segm_data, stimulus1_name, stimulus2_name]
            label[f'Sub{nSub}Trial{k_tra+1}_direction'] = np.array(label_direction)
            label[f'Sub{nSub}Trial{k_tra+1}_stimulus'] = np.array(label_stimulus)
            print(stimulus1_name)
            print(stimulus2_name)
            if stimulus1_name not in stimulus_total:
                stimulus1_data = data_read_stimulus(stimulus1_name, stimulus_path, layer_list=layer_list)
                stimulus_total[f'{stimulus1_name}'] = stimulus1_data
            if stimulus2_name not in stimulus_total:
                stimulus2_data = data_read_stimulus(stimulus2_name, stimulus_path, layer_list=layer_list)
                stimulus_total[f'{stimulus2_name}'] = stimulus2_data
            
            # data.append(segm_data)
            # label.append(label_data)
        else:
            True
    return data, label, stimulus_total

def data_read_CL_feature(nSub, feature_name='envelope', window=1, overlap=1, condition=None, eeg_data=None, stimulus_data=None):
    if condition == 'dry' or condition == 'hrtf':
        trials = 20
    elif condition == 'all':
        trials = 8
    trials=8
    channels = 64
    window = window
    sample_rate = 128
    overlap = overlap  #0.5
    #data = np.empty((0, 360, window*sample_rate, 64))
    data = {}
    # root = '/media/nchen/CYB/KUL/preprocessed_data_50hz_mean/'
    root = eeg_data
    stimulus_path = stimulus_data
    label = {}
    stimulus_total = {}
    nSub = nSub
    condition = condition
    # train data
    data_mat = scipy.io.loadmat(root + 'S%d.mat' % nSub)

    for k_tra in range(trials):
        if data_mat['preproc_trials'][0,k_tra]['condition'] == condition or condition == 'all':
            label_direction = []
            label_stimulus = []
            mat_data = data_mat['preproc_trials'][0, k_tra]['RawData'][0, 0]['EegData']
            stimulus = data_mat['preproc_trials'][0, k_tra]['stimuli'][0,0][:,0]
            table = str.maketrans('','', "'[]")
            stimulus1 = str(stimulus[0]).translate(table)
            stimulus2 = str(stimulus[1]).translate(table)
            stimulus1_name = rename(stimulus1, feature_name)
            stimulus2_name = rename(stimulus2, feature_name)
            trial_data = mat_data[0][0][:,:].reshape((1,-1,channels))
            segm_data = np.empty((0, int(window * sample_rate), 64))
            segment = int(((trial_data.shape[1]/sample_rate) - (window-1))/(window*overlap))-1
            # print(segment)
            #trail_data = np.zeros((1, mat_data.shape[0], channels))
            for j in range(segment):
                start = int(window*sample_rate*j)
                end = start+int(sample_rate*window)
                segment_data = trial_data[0][start:end][:].reshape((1,-1,channels))
                # print(segm_data.shape)
                segm_data = np.concatenate((segm_data,segment_data),axis=0)

            mat_label = data_mat['preproc_trials'][0, k_tra]['attended_ear']
            stimulus_label = int(data_mat['preproc_trials'][0, k_tra]['attended_track'][0,0])
            if mat_label[0] == 'L':
                label_direction.append([0]*segment)
            elif mat_label[0] == 'R':
                label_direction.append([1]*segment)
            label_stimulus.append([stimulus_label] * segment)

            segm_data = np.expand_dims(segm_data, axis=0)
            #data = np.concatenate((data, segm_data), axis=0)
            data[f'Sub{nSub}Trial{k_tra+1}'] = [segm_data, stimulus1_name, stimulus2_name]
            label[f'Sub{nSub}Trial{k_tra+1}_direction'] = np.array(label_direction)
            label[f'Sub{nSub}Trial{k_tra+1}_stimulus'] = np.array(label_stimulus)
            print(stimulus1_name)
            print(stimulus2_name)
            if stimulus1_name not in stimulus_total:
                stimulus1_data = data_read_stimulus(stimulus1_name, stimulus_path)
                stimulus_total[f'{stimulus1_name}'] = stimulus1_data
            if stimulus2_name not in stimulus_total:
                stimulus2_data = data_read_stimulus(stimulus2_name, stimulus_path)
                stimulus_total[f'{stimulus2_name}'] = stimulus2_data
            
            # data.append(segm_data)
            # label.append(label_data)
        else:
            True
    return data, label, stimulus_total

def data_read_Restruction(nSub, feature_name='envelope', condition=None):
    if condition == 'dry' or condition == 'hrtf':
        trials = 20
    elif condition == 'all':
        trials = 8
    channels = 64
    window = 1
    sample_rate = 128
    overlap = 1  #0.5
    #data = np.empty((0, 360, window*sample_rate, 64))
    data = {}
    root = '/media/nchen/CYB/KUL/preprocessed_data_50hz_mean/'
    label = {}
    stimulus_total = {}
    nSub = nSub
    condition = condition
    # train data
    data_mat = scipy.io.loadmat(root + 'S%d.mat' % nSub)
    speaker_dict = {'part1_track1': 1, 'rep_part1_track1': 1, 'part2_track1': 1, 'rep_part2_track1': 1,
                    'part3_track1': 2, 'rep_part3_track1': 2, 'part4_track1': 2, 'rep_part4_track1': 2,
                    'part1_track2': 3, 'rep_part1_track2': 3, 'part2_track2': 3, 'rep_part2_track2': 3,
                    'part3_track2': 3, 'rep_part3_track2': 3, 'part4_track2': 3, 'rep_part4_track2': 3,}

    for k_tra in range(trials):
        if data_mat['preproc_trials'][0,k_tra]['condition'] == condition or condition == 'all':
            label_direction = []
            label_stimulus = []
            label_speaker = []
            mat_data = data_mat['preproc_trials'][0, k_tra]['RawData'][0, 0]['EegData']
            stimulus = data_mat['preproc_trials'][0, k_tra]['stimuli'][0,0][:,0]
            table = str.maketrans('','', "'[]")
            stimulus1 = str(stimulus[0]).translate(table)
            stimulus2 = str(stimulus[1]).translate(table)
            stimulus1_name = rename(stimulus1, feature_name)
            stimulus2_name = rename(stimulus2, feature_name)
            audio1_name = rename(stimulus1, 'audio')
            audio2_name = rename(stimulus2, 'audio')
            trial_data = mat_data[0][0][:,:].reshape((1,-1,channels))
            segm_data = np.empty((0, window * sample_rate, 64))
            segment = int((trial_data.shape[1]/sample_rate)/(window*overlap))
            #trail_data = np.zeros((1, mat_data.shape[0], channels))
            for j in range(segment):
                start = int(overlap*sample_rate*j)
                end = int(start+sample_rate*window)
                segment_data = trial_data[0][start:end][:].reshape((1,-1,channels))
                segm_data = np.concatenate((segm_data,segment_data),axis=0)

            mat_label = data_mat['preproc_trials'][0, k_tra]['attended_ear']
            stimulus_label = int(data_mat['preproc_trials'][0, k_tra]['attended_track'][0,0])
            if mat_label[0] == 'L':
                label_direction.append([0]*segment)
            elif mat_label[0] == 'R':
                label_direction.append([1]*segment)
            label_stimulus.append([stimulus_label] * segment)

            speaker_id = get_speaker(str(stimulus[stimulus_label-1]).translate(table), speaker_dict)
            label_speaker.append([speaker_id] * segment)

            segm_data = np.expand_dims(segm_data, axis=0)
            #data = np.concatenate((data, segm_data), axis=0)
            data[f'Sub{nSub}Trial{k_tra+1}'] = [segm_data, stimulus1_name, stimulus2_name, audio1_name, audio2_name]
            label[f'Sub{nSub}Trial{k_tra+1}_direction'] = np.array(label_direction)
            label[f'Sub{nSub}Trial{k_tra+1}_stimulus'] = np.array(label_stimulus)
            label[f'Sub{nSub}Trial{k_tra+1}_speaker'] = np.array(label_speaker)
            print(stimulus1_name)
            print(stimulus2_name)
            print(audio1_name)
            print(audio2_name)
            if stimulus1_name not in stimulus_total:
                stimulus1_data = data_read_stimulus(stimulus1_name)
                audio1_data = data_read_stimulus(audio1_name)
                stimulus_total[f'{stimulus1_name}'] = stimulus1_data
                stimulus_total[f'{audio1_name}'] = audio1_data
            if stimulus2_name not in stimulus_total:
                stimulus2_data = data_read_stimulus(stimulus2_name)
                audio2_data = data_read_stimulus(audio2_name)
                stimulus_total[f'{stimulus2_name}'] = stimulus2_data
                stimulus_total[f'{audio2_name}'] = audio2_data
            
            # data.append(segm_data)
            # label.append(label_data)
        else:
            True

    return data, label, stimulus_total

def data_read_CL_Restruct(nSub, feature_name='envelope', restruct_name='envelope', condition=None, window=1, overlap=1):
    if condition == 'dry' or condition == 'hrtf':
        trials = 20
    elif condition == 'all':
        trials = 8
    channels = 64
    window = window
    sample_rate = 128
    overlap = overlap  # 0.5
    # data = np.empty((0, 360, window*sample_rate, 64))
    data = {}
    root = '/media/nchen/CYB/KUL/preprocessed_data_50hz_mean/'
    label = {}
    stimulus_total = {}
    restruct_stim = {}
    nSub = nSub
    condition = condition
    # train data
    data_mat = scipy.io.loadmat(root + 'S%d.mat' % nSub)

    for k_tra in range(trials):
        if data_mat['preproc_trials'][0, k_tra]['condition'] == condition or condition == 'all':
            label_direction = []
            label_stimulus = []
            mat_data = data_mat['preproc_trials'][0, k_tra]['RawData'][0, 0]['EegData']
            stimulus = data_mat['preproc_trials'][0, k_tra]['stimuli'][0, 0][:, 0]
            table = str.maketrans('', '', "'[]")
            stimulus1 = str(stimulus[0]).translate(table)
            stimulus2 = str(stimulus[1]).translate(table)
            stimulus1_name = rename(stimulus1, feature_name)
            stimulus2_name = rename(stimulus2, feature_name)
            restruct_stim1_name = rename(stimulus1, restruct_name)
            restruct_stim2_name = rename(stimulus2, restruct_name)
            trial_data = mat_data[0][0][:, :].reshape((1, -1, channels))
            segm_data = np.empty((0, window * sample_rate, 64))
            segment = int((trial_data.shape[1] / sample_rate) / (window * overlap))
            # trail_data = np.zeros((1, mat_data.shape[0], channels))
            for j in range(segment):
                start = int(overlap * sample_rate * j)
                end = int(start + sample_rate * window)
                segment_data = trial_data[0][start:end][:].reshape((1, -1, channels))
                segm_data = np.concatenate((segm_data, segment_data), axis=0)

            mat_label = data_mat['preproc_trials'][0, k_tra]['attended_ear']
            stimulus_label = int(data_mat['preproc_trials'][0, k_tra]['attended_track'][0, 0])
            if mat_label[0] == 'L':
                label_direction.append([0] * segment)
            elif mat_label[0] == 'R':
                label_direction.append([1] * segment)
            label_stimulus.append([stimulus_label] * segment)

            segm_data = np.expand_dims(segm_data, axis=0)
            # data = np.concatenate((data, segm_data), axis=0)
            data[f'Sub{nSub}Trial{k_tra + 1}'] = [segm_data, stimulus1_name, stimulus2_name, restruct_stim1_name, restruct_stim2_name]
            label[f'Sub{nSub}Trial{k_tra + 1}_direction'] = np.array(label_direction)
            label[f'Sub{nSub}Trial{k_tra + 1}_stimulus'] = np.array(label_stimulus)
            print(stimulus1_name)
            print(stimulus2_name)
            if stimulus1_name not in stimulus_total:
                stimulus1_data = data_read_stimulus(stimulus1_name)
                stimulus_total[f'{stimulus1_name}'] = stimulus1_data
            if stimulus2_name not in stimulus_total:
                stimulus2_data = data_read_stimulus(stimulus2_name)
                stimulus_total[f'{stimulus2_name}'] = stimulus2_data

            print(restruct_stim1_name)
            print(restruct_stim2_name)
            if restruct_stim1_name not in restruct_stim:
                stimulus1_data = data_read_stimulus(restruct_stim1_name)
                restruct_stim[f'{restruct_stim1_name}'] = stimulus1_data
            if restruct_stim2_name not in restruct_stim:
                stimulus2_data = data_read_stimulus(restruct_stim2_name)
                restruct_stim[f'{restruct_stim2_name}'] = stimulus2_data
            # data.append(segm_data)
            # label.append(label_data)
        else:
            True
    return data, label, stimulus_total, restruct_stim

def data_read_CL_feature_cs(nSub, feature_name = 'envelope', condition = 'hrtf'):
    test_Sub = [nSub]
    train_Sub = list(set(Sub) - set(test_Sub))
    data_train = {}
    label_train = {}
    stimulus_total = {}

    for i in range(len(train_Sub)):
        data_sub, label_sub, stimulus = data_read_CL_feature(train_Sub[i], feature_name, condition)
        data_train.update(data_sub)
        label_train.update(label_sub)
        stimulus_total.update(stimulus)
    
    data_test, label_test, test_stimulus = data_read_CL_feature(test_Sub[0], feature_name, condition)
    stimulus_total.update(test_stimulus)

    return data_train, label_train, data_test, label_test, stimulus_total


class KULdataset(Dataset):
    def __init__(self, data, label,stimulus):
        eeg_dim, stimulus_dim = 64, 1
        final_data = np.empty((0, 128, eeg_dim + stimulus_dim*2))
        final_label = np.empty((0, 1))
        for key in data.keys():
            eeg = np.squeeze(data[key][0])
            att_stim_name = data[key][1]
            unatt_stim_name = data[key][2]
            att_stimulus = stimulus[att_stim_name]
            unatt_stimulus = stimulus[unatt_stim_name]
            data_sigle = np.concatenate((eeg, att_stimulus, unatt_stimulus), axis=-1)
            final_data = np.concatenate((final_data, data_sigle), axis=0)
            t = label[key + '_stimulus'][0].reshape(-1, 1)
            final_label = np.concatenate((final_label, label[key + '_stimulus'][0].reshape(-1, 1)), axis=0)
        self.data = final_data.transpose((0, 2, 1))
        self.label = final_label
        #for i in range():
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index], self.label[index]

def data_read_hrtf_Niu(condition='hrtf'):
        # test_Sub = [nSub]
        # train_Sub = list(set(Sub) - set(test_Sub))
        train_Sub = Sub
        window = 1
        sample_rate = 128
        # train_data = np.empty((0, 360, window*sample_rate, 64))
        # train_label = np.empty((0, 1, 360))
        data = []
        label = []

        for i in range(len(train_Sub)):
            sub_data, sub_label = data_read_condition(train_Sub[i], condition, trials=8)
            #   train_data = np.concatenate((train_data, data), axis=0)
            #   train_label = np.concatenate((train_label, label), axis=0)
            data.extend(sub_data)
            label.extend(sub_label)
        
        return data, label


if __name__ == '__main__':
    # #_, _, _, _, = data_read_cs(1)
    # # data, label = data_read_condition(1, 'hrtf')
    # # data_label = np.empty((1,0))
    # # for i in range(len(data)):
    # #      data_label = np.concatenate((data_label, np.array(label[i])), axis=1)
    # # data = data.stack(dim=1)
    # # label = label.stack(dim=1)
    # # print(data.shape, label.shape)
    # data, label, stimulus = data_read_CL(1, 'envelope', 'hrtf')
    # total = list(data.keys())
    # t = list(data.keys())[0:2]
    # l = list(map(lambda x: x + '_stimulus', t))
    # #train_trial_index = list(set(trial_index) - set(test_trial_index))
    # t_train = list(set(total) - set(t))
    # data_test = {key: data[key] for key in t }
    # da = data[t]
    # # eeg_dim, stimulus_dim = 64, 1
    # # final_data = np.empty((0, 128, eeg_dim + stimulus_dim*2))
    # # final_label = np.empty((0, 1))
    # # for key in data.keys():
    # #     eeg = np.squeeze(data[key][0])
    # #     att_stim_name = data[key][1]
    # #     unatt_stim_name = data[key][2]
    # #     att_stimulus = stimulus[att_stim_name]
    # #     unatt_stimulus = stimulus[unatt_stim_name]
    # #     data_sigle = np.concatenate((eeg, att_stimulus, unatt_stimulus), axis=-1)
    # #     final_data = np.concatenate((final_data, data_sigle), axis=0)
    # #     t = label[key + '_stimulus'][0].reshape(-1, 1)
    # #     final_label = np.concatenate((final_label, label[key + '_stimulus'][0].reshape(-1, 1)), axis=0)
    # a = KULdataset(data, label, stimulus)
    # data, label = data_read_hrtf_Niu()
    # print(data.shape)
    # print(label.shape)
    #data_tr, label_tr, data_te, label_te, stimulus = data_read_CL_cs(1, 'envelope', 'hrtf')
    #print(data_tr.keys())
    # data, label = data_read_hrtf_Niu()
    # _, _ = data_read_Restruction(1, condition='all')
    True