"""Code to calculate mel spectrograms."""
import os
import math
import glob
import librosa
import numpy as np
import scipy
from kornia.core import tensor
from pyarrow.dataset import dataset
from scipy.signal import resample
from scipy.ndimage import zoom
from scipy.io import wavfile
import torch
from sympy.core.random import sample
from tqdm import tqdm
from transformers import (Wav2Vec2FeatureExtractor, Wav2Vec2Model)
from sklearn.decomposition import PCA
from transformers import AutoFeatureExtractor, WhisperModel
import copy

os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

def rename(original, name, dataset='KUL'):

    # 找到最后一个下划线的位置
    if dataset == 'KUL':
        last_underscore_index = original.rfind('_')#KUL
    else:
        last_underscore_index = original.rfind('.')  # DTU

    # 提取最后一个下划线之前的内容  
    part_before_last_underscore = original[:last_underscore_index]  

    # 创建新的字符串，后缀替换为 'envelope.npy'  
    new_string = part_before_last_underscore + '_' + name + '.npy'  

    # 输出结果  
    return new_string

def rename_new(original, name, window, dataset="KUL"):

    if dataset == 'KUL':
        last_underscore_index = original.rfind('_')  # KUL
    else:
        last_underscore_index = original.rfind('.')  # DTU

    # 提取最后一个下划线之前的内容
    part_before_last_underscore = original[:last_underscore_index]

    # 创建新的字符串，后缀替换为 'envelope.npy'
    new_string = part_before_last_underscore + '_' + name + f'_{window}s' + '.npy'

    # 输出结果
    return new_string

def preemphasis(x, preemph):
    return scipy.signal.lfilter([1, -preemph], [1], x)

def calculate_mel_spectrogram(
    audio_path,
    target_fs=64,
    fmin=0,
    fmax=5000,
    nb_filters=10,
    hop_length=None,
    win_length=None,
):
    """Calculates mel spectrogram of a raw speech file. This function makes the same calucation as
    in the sparrKULee pipeline and is the regression objective for task 2.

    Parameters
    ---------
    audio_path: str
        audio file path
    target_fs: int
        Sampling frequency of the calculated mel spectrogram
    fmin: Union[float, int]
        Minimum center frequency used in mel filter matrix
    fmax: Union[float, int]
        Maximum center frequency used in mel filter matrix
    nb_filters: int
        Number of mel spectrogram frequency bands
    hop_length: int
        Hop length (in samples) used for calculation of the spectrogram
    win_length: int
        Window length (in samples) of each frame

    Returns
    -------
    numpy.ndarray
        Mel spectrogram
    """

    # unzip audio file


    # speech = dict(np.load(audio_path))
    # audio, fs = speech["audio"], speech["fs"]
    fs, audio = wavfile.read(audio_path)
    if not hop_length:
        hop_length = int((1 / target_fs) * fs)  # this will downsample the signal to target_fs Hz
    if not win_length:
        win_length = int(0.025 * fs)  # 25 milli seconds

    # Finds the closest power of 2
    # that is bigger than win_length
    n_fft = int(math.pow(2, math.ceil(math.log2(win_length))))

    # DC removal
    audio = audio - np.mean(audio)

    # mel_spectrogram = librosa.feature.melspectrogram(audio, window='hann',
    #                                    sr=fs, n_fft=n_fft, hop_length=hop_length,
    #                                    win_length=win_length, fmin=fmin, fmax=fmax, htk=False, norm='slaney',
    #                                    n_mels=nb_filters, center=False)

    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=fs, n_fft=n_fft, 
                                        hop_length=hop_length, win_length=win_length,
                                        fmin=fmin, fmax=fmax, n_mels=nb_filters,
                                        window='hann', center=False, htk=False, norm='slaney')

    log_fbank = librosa.power_to_db(mel_spectrogram, ref=np.max, amin=1e-12)

    # return mel_spectrogram
    return log_fbank

def calculate_logmel_spectrogram(wav_path, sr=160000, preemph=0.97, n_fft=2048, n_mels=80, hop_length=160,
                win_length=400, fmin=50, top_db=80, bits=8, offset=0.0, duration=None):
    wav, _ = librosa.load(wav_path, sr=sr,
                        offset=offset, duration=duration)
    wav = wav / np.abs(wav).max() * 0.999
    # wav = preemphasis(wav, preemph)

    mel = librosa.feature.melspectrogram(y=preemphasis(wav, preemph),
                                        sr=sr,
                                        n_fft=n_fft,
                                        n_mels=n_mels,
                                        hop_length=hop_length,
                                        win_length=win_length,
                                        fmin=fmin,
                                        power=1)
    logmel = librosa.amplitude_to_db(mel, top_db=top_db)
    logmel = logmel / top_db + 1

    return logmel

def split_feature(feature, window=1, sample_rate=128, overlap=1):
    if len(feature.shape) !=2:
        feature = np.expand_dims(feature, axis=-1)
    time, dim = feature.shape
    sample_num = int(time/(window*sample_rate*overlap))
    split_data = np.empty((0, int(window * sample_rate), dim))
    if overlap != 1:
        win_pap = window - 1
    else:
        win_pap = 1
    for i in range(sample_num-win_pap):
        start = int(overlap*sample_rate*window*i)
        end = int(start+sample_rate*window)
        segment_data = feature[start:end, :].reshape((1,-1,dim))
        split_data = np.concatenate((split_data,segment_data),axis=0)

    return split_data
# 'Center freqs' of mel bands - uniformly spaced between limits
# mel_f:  [   0.        ,  147.02442191,  324.92910187,  540.19997145,
#         800.6852341 , 1115.88148983, 1497.27995596, 1958.78540639,
#        2517.22310262, 3192.95219807, 4010.6079787 , 5000.        ]
def data_read_condition(nSub, condition):
        trials,channels = 20,64
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

def get_envelope(path='D:/EEGViT/stimulus/'):
    trials,channels = 8,64
    window = 0.5
    sample_rate = 128
    overlap = 1  #0.5
    #data = np.empty((0, 360, window*sample_rate, 64))
    data = []
    root = 'D:\KUL\preprocessed_data_50hz_mean/'
    label = []
    nSub = 1
    #condition = dry
    # train data
    data_mat = scipy.io.loadmat(root + 'S%d.mat' % nSub)
    if not os.path.exists(path):
        os.makedirs(path)

    for k_tra in range(trials):
        #if data_mat['preproc_trials'][0,k_tra]['condition'] == condition:
        #mat_data = data_mat['preproc_trials'][0, k_tra]['RawData'][0, 0]['EegData']
        stimulus = data_mat['preproc_trials'][0, k_tra]['stimuli'][0,0][:,0]
        envelope = data_mat['preproc_trials'][0, k_tra]['Envelope'][0, 0]['AudioData']
        table = str.maketrans('','', "'[]")
        stimulus1 = str(stimulus[0]).translate(table)
        stimulus2 = str(stimulus[1]).translate(table)
        stimulus1_name = rename(stimulus1, f'envelope_{window}s')
        stimulus2_name = rename(stimulus2, f'envelope_{window}s')
        file_path1 = os.path.join(path, f'{stimulus1_name}')
        file_path2 = os.path.join(path, f'{stimulus2_name}')
        if not os.path.exists(file_path1):
            #os.makedirs(stimulus1_name)
            envelope1 = np.sum(envelope[0][0][:,:,0], axis=1, keepdims=True)
            envelope1_split = split_feature(envelope1, window=window, sample_rate=sample_rate, overlap=overlap)
            np.save(file_path1, envelope1_split)
        if not os.path.exists(file_path2):
            #os.makedirs(stimulus2_name)
            envelope2 = np.sum(envelope[0][0][:,:,1], axis=1, keepdims=True)
            envelope2_split = split_feature(envelope2, window=window, sample_rate=sample_rate, overlap=overlap)
            np.save(file_path2, envelope2_split)

def get_envelope_DTU(path='E:/DTU/right_data/stimulus/'):
    trials,channels = 60,64
    window = 1
    sample_rate = 128
    overlap = 1 #0.5
    #data = np.empty((0, 360, window*sample_rate, 64))
    data = []
    root = 'E:/DTU/right_data/'
    label = []
    nSub = 4
    #condition = dry
    # train data
    data_mat = scipy.io.loadmat(root + 'S%d_data_preproc.mat' % nSub)
    label_data = scipy.io.loadmat(root + 'S%d_label.mat' % nSub)
    if not os.path.exists(path):
        os.makedirs(path)

    for k_tra in range(trials):
        #if data_mat['preproc_trials'][0,k_tra]['condition'] == condition:
        #mat_data = data_mat['preproc_trials'][0, k_tra]['RawData'][0, 0]['EegData']
        mat_data = data_mat['data']['eeg'][0, 0][0, k_tra][:, :64]
        stimulus = [label_data['wav_male'][k_tra, 0], label_data['wav_female'][k_tra, 0]]
        envelope_wavA = data_mat['data']['wavA'][0,0][0, k_tra]
        envelope_wavB = data_mat['data']['wavB'][0,0][0, k_tra]
        # stimulus = data_mat['preproc_trials'][0, k_tra]['stimuli'][0,0][:,0]
        # envelope = data_mat['preproc_trials'][0, k_tra]['Envelope'][0, 0]['AudioData']
        table = str.maketrans('','', "'[]")
        stimulus1 = str(stimulus[0]).translate(table)
        stimulus2 = str(stimulus[1]).translate(table)
        stimulus1_name = rename(stimulus1, f'envelope_{window}s', "DTU")
        stimulus2_name = rename(stimulus2, f'envelope_{window}s', "DTU")
        print(stimulus1_name)
        print(stimulus2_name)
        file_path1 = os.path.join(path, f'{stimulus1_name}')
        file_path2 = os.path.join(path, f'{stimulus2_name}')
        if not os.path.exists(file_path1):
            #os.makedirs(stimulus1_name)
            # envelope1 = np.sum(envelope[0][0][:,:,0], axis=1, keepdims=True)
            envelope1_split = split_feature(envelope_wavA, window=window, sample_rate=sample_rate, overlap=overlap)
            np.save(file_path1, envelope1_split)
        if not os.path.exists(file_path2):
            #os.makedirs(stimulus2_name)
            # envelope2 = np.sum(envelope[0][0][:,:,1], axis=1, keepdims=True)
            envelope2_split = split_feature(envelope_wavB, window=window, sample_rate=sample_rate, overlap=overlap)
            np.save(file_path2, envelope2_split)
import pandas as pd
import h5py
def get_envelope_NJU(path='E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/envelope/'):
    trials,channels = 32,64
    window = 1
    sample_rate = 128
    overlap = 1 #0.5
    #data = np.empty((0, 360, window*sample_rate, 64))
    data = []
    root = 'E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/NJUNCA_preprocessed_arte_removed/'
    label = []
    nSub = 3
    #condition = dry
    # train data
    data_mat = scipy.io.loadmat(root + f'S{nSub:02}_1.mat')
    # with h5py.File(root + f'S{nSub:02}.mat', 'r') as f:
    #     data_mat = {}
    #     # 递归函数来提取所有数据集
    #     def extract_data(obj, parent_key=''):
    #         if isinstance(obj, h5py.Dataset):
    #             # 如果是数据集，添加到字典
    #             data_mat[parent_key] = obj[()]
    #         elif isinstance(obj, h5py.Group):
    #             # 如果是组，递归处理其内容
    #             for key, value in obj.items():
    #                 extract_data(value, f"{parent_key}/{key}" if parent_key else key)
    #     extract_data(f)
    stim = pd.read_csv("E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/eeg_data/" + f'S{nSub:02}_expinfo.csv')
    if not os.path.exists(path):
        os.makedirs(path)

    for k_tra in range(trials):

        stimulus_left_name = stim['l_audio'][k_tra][:]
        stimulus_right_name = stim['r_audio'][k_tra][:]
        envelope_wavA = data_mat['leftEnv'][0, k_tra]
        envelope_wavB = data_mat['rightEnv'][0, k_tra]
        # stimulus = data_mat['preproc_trials'][0, k_tra]['stimuli'][0,0][:,0]
        # envelope = data_mat['preproc_trials'][0, k_tra]['Envelope'][0, 0]['AudioData']
        table = str.maketrans('','', "'[]")
        stimulus1 = str(stimulus_left_name).translate(table)
        stimulus2 = str(stimulus_right_name).translate(table)
        stimulus1_name = rename(stimulus1, f'envelope_{window}s', "NJU")
        stimulus2_name = rename(stimulus2, f'envelope_{window}s', "NJU")
        print(stimulus1_name)
        print(stimulus2_name)
        file_path1 = os.path.join(path, f'{stimulus1_name}')
        file_path2 = os.path.join(path, f'{stimulus2_name}')
        if not os.path.exists(file_path1):
            #os.makedirs(stimulus1_name)
            # envelope1 = np.sum(envelope[0][0][:,:,0], axis=1, keepdims=True)
            envelope1_split = split_feature(envelope_wavA, window=window, sample_rate=sample_rate, overlap=overlap)
            np.save(file_path1, envelope1_split)
        if not os.path.exists(file_path2):
            #os.makedirs(stimulus2_name)
            # envelope2 = np.sum(envelope[0][0][:,:,1], axis=1, keepdims=True)
            envelope2_split = split_feature(envelope_wavB, window=window, sample_rate=sample_rate, overlap=overlap)
            np.save(file_path2, envelope2_split)

def get_melspectrogram(wav_path, save_path, target_fs=128, number_filters=28, feat_name = 'mel'):
    file_list = glob.glob(os.path.join(wav_path, '*dry.wav'))
    for file in file_list:
        file_name = os.path.basename(file)
        mel_data = calculate_mel_spectrogram(audio_path=file, target_fs=target_fs, nb_filters=number_filters)
        data = split_feature(mel_data.T)
        data_mean = np.mean(data, axis=2, keepdims=True)
        # data_std = np.std(data, axis=1, keepdims=True)
        # data = (data-data_mean)/data_std
        data_max = np.max(data, axis=2, keepdims=True)
        data_min = np.min(data, axis=2, keepdims=True)
        data = (data-data_min) / (data_max-data_min + 1e-6)
        newname = rename(file_name, name=feat_name)
        np.save(os.path.join(save_path, newname), data)

def get_log_melspectrogram(wav_path, save_path, target_fs=128, number_filters=28, window=1, overlap=1, top_db=80, dataset="KUL"):
    if dataset == "KUL":
        file_list = glob.glob(os.path.join(wav_path, '*dry.wav'))#KUL
    else:
        file_list = glob.glob(os.path.join(wav_path, '*.wav'))  # DTU
    for file in file_list:
        file_name = os.path.basename(file)
        mel_data = calculate_mel_spectrogram(audio_path=file, target_fs=target_fs, nb_filters=number_filters)
        logmel = librosa.amplitude_to_db(mel_data, top_db=top_db)
        logmel = logmel / top_db + 1
        data = split_feature(logmel.T, window=window, sample_rate=target_fs, overlap=overlap)
        newname = rename(file_name, name=f'logmel{number_filters}_{window}s', dataset=dataset)
        print(newname)
        np.save(os.path.join(save_path, newname), data)

def get_wav2vec(wav_path, save_path, target_fs=128, device=1, layer=14, pca_dim=64, window=1, overlap=0.5):

    model_path = '/media/nchen/CYB/EEG-Stimulus-Match-Mismatch/PM/wav2vec2-large-xlsr-53-dutch'
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2Model.from_pretrained(model_path)
    model = model.to(device)
    model = model.half()
    model.eval()
    with torch.no_grad():
        file_list = glob.glob(os.path.join(wav_path, '*dry.wav'))
        for file in file_list:
            file_name = os.path.basename(file)
            sr, wav = wavfile.read(file)
            wav = resample(wav, int(wav.shape[0] * 16000/sr))
            sr = 16000
            feat_patch = []
            for sp_st in tqdm(range(0, len(wav)-sr*window, int(sr*window*overlap))): #5s
                wav_i = wav[sp_st:sp_st+sr*window+100]
                if len(wav_i) < 500:
                    continue
                wav_input = feature_extractor(wav_i, return_tensors='pt', sampling_rate=sr)
                
                input_values = wav_input.input_values.half().to(device)
                o = model(input_values, output_hidden_states=True)
    
                feat_patch.append(o[2][layer].detach().squeeze().cpu().numpy())
            wav_time = [len(feat_patch[i]) for i in range(len(feat_patch))]
            add_index = [index for index, item in enumerate(wav_time) if item != 50*window]
            if len(add_index) != 0:
                for i in range(len(add_index)):
                    if len(feat_patch[add_index[i]].shape) == 2:
                        if wav_time[add_index[i]] < 50*window:
                            feat_patch[add_index[i]] = np.concatenate((feat_patch[add_index[i]], np.zeros((50*window-wav_time[add_index[i]], 1024))), axis=0)
                        elif wav_time[add_index[i]] > 50*window:
                            feat_patch[add_index[i]] = feat_patch[add_index[i]][:50*window,:]
                        wav_time[add_index[i]] = 50*window
                    else:
                        feat_patch.pop(add_index[i])
                        wav_time.pop(add_index[i])
            print(f'PCA feat wav2vec2 layer {layer}')
            feat = np.concatenate(feat_patch)
            pca = PCA(n_components=pca_dim)
            feat_pca = pca.fit_transform(feat)
            feat_pca_split = np.split(feat_pca, np.cumsum(wav_time)[:-1])

            emb = np.array(feat_pca_split).astype(np.float64)

            zoom_factors = (1, target_fs/50, 1)
            emb_res = zoom(emb, zoom_factors)

            emb_res = emb_res.astype(np.float32)
            newname = rename(file_name, name=f'wav2vec_{window}s')
            print(newname)
            np.save(os.path.join(save_path, newname), emb_res)


def get_wav2vec_all(wav_path, save_path, target_fs=128, device=0, layer_num=1, pca_dim=1024, window=1, overlap=1, dataset='KUL'):
    model_path = 'D:\EEGViT/PM/wav2vec2-large-xlsr-53-dutch'
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2Model.from_pretrained(model_path)
    model = model.to(device)
    model = model.half()
    model.eval()
    # layer = [i for i in range(0, layer_num+1, 2)]
    layer = [19]
    with torch.no_grad():
        if dataset == 'KUL':
            file_list = glob.glob(os.path.join(wav_path, '*dry.wav'))
        else:
            file_list = glob.glob(os.path.join(wav_path, '*.wav'))
        for file in file_list:
            file_name = os.path.basename(file)
            sr, wav = wavfile.read(file)
            wav = resample(wav, int(wav.shape[0] * 16000 / sr))
            sr = 16000
            feat_patch = []
            for sp_st in tqdm(range(0, len(wav)-int(sr*window*overlap), int(sr*window*overlap))):  # 5s
                wav_i = wav[sp_st:sp_st + int(sr*window) + 100]
                if len(wav_i) < 500:
                    continue
                wav_input = feature_extractor(wav_i, return_tensors='pt', sampling_rate=sr)

                input_values = wav_input.input_values.half().to(device)
                o = model(input_values, output_hidden_states=True)
                feat_layer = []
                for l in range(len(layer)):
                    feat_layer.append(o[2][layer[l]].detach().squeeze().cpu().numpy())
                if len(feat_layer) == len(layer):
                    feat_patch.append(feat_layer)
            # wav_time = [len(feat_patch[i]) for i in range(len(feat_patch))]
            wav_time = []
            for data in feat_patch:
                data_dim = []
                End = True
                for dim in data:
                    if len(dim.shape) == 2:
                        data_dim.append(dim.shape[0])
                    else:
                        End = False
                if End :
                    wav_time.append(data_dim)
                else:
                    feat_patch = feat_patch[:-1]

            # add_index = [index for index, item in enumerate(wav_time) if item != 50]
            add_index = [[index for index, item in enumerate(wav_time[i]) if item !=int(50*window) ] for i in range(len(wav_time))]

            if len(add_index) != 0:
                for i in range(len(add_index)):
                    if add_index[i] != []:
                        dim_index = add_index[i]
                        # print(len(wav_time[i]))
                        # if len(dim_index) == len(wav_time[i]):
                        for d in dim_index:
                            # if len(feat_patch[add_index[i]].shape) == 2:
                            if len(feat_patch[i][d].shape) == 2:
                                if wav_time[i][d] < int(50*window):
                                    feat_patch[i][d] = np.concatenate(
                                        (feat_patch[i][d], np.zeros((int(50*window) - wav_time[i][d], 1024))), axis=0)
                                elif wav_time[i][d] > int(50*window):
                                    feat_patch[i][d] = feat_patch[i][d][:int(50*window), :]
                                wav_time[i][d] = int(50*window)
                            else:
                                feat_patch[i].pop(d)
                                wav_time[i].pop(d)
                        # else:
                        #     feat_patch.pop(i)
                        #     wav_time.pop(i)
            print(f'PCA feat wav2vec2 layer {layer}')
            sample = len(feat_patch)
            layer_number = len(feat_patch[0])
            time, dim = feat_patch[0][0].shape
            feat = np.concatenate(feat_patch).reshape((-1, dim))
            # pca = PCA(n_components=pca_dim)
            # feat_pca = pca.fit_transform(feat)
            feat_pca = feat
            t = np.cumsum(wav_time)[:-1]
            feat_pca_split = np.split(feat_pca, np.cumsum(wav_time)[:-1])

            emb = np.array(feat_pca_split).astype(np.float64)
            emb = emb.reshape((sample, layer_number, time, pca_dim))
            zoom_factors = (1, 1, target_fs / 50, 1)
            emb_res = zoom(emb, zoom_factors)

            emb_res = emb_res.astype(np.float32)
            newname = rename(file_name, name='wav2vec_' + f'{layer_num}layer'+f'_{window}s', dataset=dataset)
            print(newname)
            np.save(os.path.join(save_path, newname), emb_res)

def get_wav2vec_all_24(wav_path, save_path, target_fs=128, device=0, layer_num=24, pca_dim=64, window=1, overlap=1):
    model_path = 'D:\EEGViT/PM/wav2vec2-large-xlsr-53-dutch'
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_path)
    model = Wav2Vec2Model.from_pretrained(model_path)
    model = model.to(device)
    model = model.half()
    model.eval()
    layer = [i for i in range(0, layer_num+1)]
    with torch.no_grad():
        file_list = glob.glob(os.path.join(wav_path, '*dry.wav'))
        for file in file_list:
            file_name = os.path.basename(file)
            sr, wav = wavfile.read(file)
            wav = resample(wav, int(wav.shape[0] * 16000 / sr))
            sr = 16000
            feat_patch = []
            for sp_st in tqdm(range(0, len(wav)-sr*window, int(sr*window*overlap))):  # 5s
                wav_i = wav[sp_st:sp_st + sr*window + 100]
                if len(wav_i) < 500:
                    continue
                wav_input = feature_extractor(wav_i, return_tensors='pt', sampling_rate=sr)

                input_values = wav_input.input_values.half().to(device)
                o = model(input_values, output_hidden_states=True)
                feat_layer = []
                for l in range(len(layer)):
                    feat_layer.append(o[2][layer[l]].detach().squeeze().cpu().numpy())
                if len(feat_layer) == len(layer):
                    feat_patch.append(feat_layer)
            # wav_time = [len(feat_patch[i]) for i in range(len(feat_patch))]
            wav_time = []
            for data in feat_patch:
                data_dim = []
                End = True
                for dim in data:
                    if len(dim.shape) == 2:
                        data_dim.append(dim.shape[0])
                    else:
                        End = False
                if End :
                    wav_time.append(data_dim)
                else:
                    feat_patch = feat_patch[:-1]

            # add_index = [index for index, item in enumerate(wav_time) if item != 50]
            add_index = [[index for index, item in enumerate(wav_time[i]) if item !=50*window ] for i in range(len(wav_time))]

            if len(add_index) != 0:
                for i in range(len(add_index)):
                    if add_index[i] != []:
                        dim_index = add_index[i]
                        # print(len(wav_time[i]))
                        # if len(dim_index) == len(wav_time[i]):
                        for d in dim_index:
                            # if len(feat_patch[add_index[i]].shape) == 2:
                            if len(feat_patch[i][d].shape) == 2:
                                if wav_time[i][d] < 50*window:
                                    feat_patch[i][d] = np.concatenate(
                                        (feat_patch[i][d], np.zeros((50*window - wav_time[i][d], 1024))), axis=0)
                                elif wav_time[i][d] > 50*window:
                                    feat_patch[i][d] = feat_patch[i][d][:50*window, :]
                                wav_time[i][d] = 50*window
                            else:
                                feat_patch[i].pop(d)
                                wav_time[i].pop(d)
                        # else:
                        #     feat_patch.pop(i)
                        #     wav_time.pop(i)
            print(f'PCA feat wav2vec2 layer {layer}')
            sample = len(feat_patch)
            layer_number = len(feat_patch[0])
            time, dim = feat_patch[0][0].shape
            feat = np.concatenate(feat_patch).reshape((-1, dim))
            pca = PCA(n_components=pca_dim)
            feat_pca = pca.fit_transform(feat)
            t = np.cumsum(wav_time)[:-1]
            feat_pca_split = np.split(feat_pca, np.cumsum(wav_time)[:-1])

            emb = np.array(feat_pca_split).astype(np.float64)
            emb = emb.reshape((sample, layer_number, time, pca_dim))
            zoom_factors = (1, 1, target_fs / 50, 1)
            emb_res = zoom(emb, zoom_factors)

            emb_res = emb_res.astype(np.float32)
            newname = rename(file_name, name='wav2vec_' + f'{layer_num}layer'+f'_{window}s')
            print(newname)
            np.save(os.path.join(save_path, newname), emb_res)

import torch
from transformers import Wav2Vec2Processor, HubertForCTC

def get_hubert_all(wav_path, save_path, target_fs=128, device=0, layer_num=14, pca_dim=64, window=1, overlap=1):
    # load the pre-trained checkpoints
    model_path = 'D:\EEGViT\PM\hubert_large'
    processor = Wav2Vec2Processor.from_pretrained(model_path)
    model = HubertForCTC.from_pretrained(model_path)
    model.eval()

    layer = [i for i in range(0, layer_num+1)]
    with torch.no_grad():
        file_list = glob.glob(os.path.join(wav_path, '*dry.wav'))
        for file in file_list:
            file_name = os.path.basename(file)
            sr, wav = wavfile.read(file)
            wav = resample(wav, int(wav.shape[0] * 16000 / sr))
            sr = 16000
            feat_patch = []
            for sp_st in tqdm(range(0, len(wav)-sr*window, int(sr*window*overlap))):  # 5s
                wav_i = wav[sp_st:sp_st + sr*window + 100]
                if len(wav_i) < 500:
                    continue
                wav_input_16khz = torch.tensor(wav_i) #.unsqueeze(0)
                # wav_input = feature_extractor(wav_i, return_tensors='pt', sampling_rate=sr)
                # input_values = wav_input.input_values.half().to(device)
                # o = model(input_values, output_hidden_states=True)
                input_values = processor(wav_input_16khz, return_tensors="pt").input_values  # Batch size 1
                o = model(input_values, output_hidden_states=True)

                feat_layer = []
                for l in range(len(layer)):
                    feat_layer.append(o['hidden_states'][l].detach().squeeze().cpu().numpy())
                if len(feat_layer) == len(layer):
                    feat_patch.append(feat_layer)
            # wav_time = [len(feat_patch[i]) for i in range(len(feat_patch))]
            wav_time = []
            for data in feat_patch:
                data_dim = []
                End = True
                for dim in data:
                    if len(dim.shape) == 2:
                        data_dim.append(dim.shape[0])
                    else:
                        End = False
                if End :
                    wav_time.append(data_dim)
                else:
                    feat_patch = feat_patch[:-1]

            # add_index = [index for index, item in enumerate(wav_time) if item != 50]
            add_index = [[index for index, item in enumerate(wav_time[i]) if item !=50*window ] for i in range(len(wav_time))]

            if len(add_index) != 0:
                for i in range(len(add_index)):
                    if add_index[i] != []:
                        dim_index = add_index[i]
                        # print(len(wav_time[i]))
                        # if len(dim_index) == len(wav_time[i]):
                        for d in dim_index:
                            # if len(feat_patch[add_index[i]].shape) == 2:
                            if len(feat_patch[i][d].shape) == 2:
                                if wav_time[i][d] < 50*window:
                                    feat_patch[i][d] = np.concatenate(
                                        (feat_patch[i][d], np.zeros((50*window - wav_time[i][d], 1024))), axis=0)
                                elif wav_time[i][d] > 50*window:
                                    feat_patch[i][d] = feat_patch[i][d][:50*window, :]
                                wav_time[i][d] = 50*window
                            else:
                                feat_patch[i].pop(d)
                                wav_time[i].pop(d)
                        # else:
                        #     feat_patch.pop(i)
                        #     wav_time.pop(i)
            print(f'PCA feat hubert layer {layer}')
            sample = len(feat_patch)
            layer_number = len(feat_patch[0])
            time, dim = feat_patch[0][0].shape
            feat = np.concatenate(feat_patch).reshape((-1, dim))
            pca = PCA(n_components=pca_dim)
            feat_pca = pca.fit_transform(feat)
            t = np.cumsum(wav_time)[:-1]
            feat_pca_split = np.split(feat_pca, np.cumsum(wav_time)[:-1])

            emb = np.array(feat_pca_split).astype(np.float64)
            emb = emb.reshape((sample, layer_number, time, pca_dim))
            zoom_factors = (1, 1, target_fs / 50, 1)
            emb_res = zoom(emb, zoom_factors)

            emb_res = emb_res.astype(np.float32)
            newname = rename(file_name, name='hubert_' + f'{layer_num}layer'+f'_{window}s')
            print(newname)
            np.save(os.path.join(save_path, newname), emb_res)

from WavLM import WavLM, WavLMConfig

def get_wavlm_all(wav_path, save_path, target_fs=128, device=0, layer_num=14, pca_dim=64, window=1, overlap=1, dataset='KUL'):
    # load the pre-trained checkpoints
    model_path = 'D:\EEGViT\PM\WavLm-large\WavLM-Large.pt'
    checkpoint = torch.load(model_path)
    cfg = WavLMConfig(checkpoint['cfg'])
    model = WavLM(cfg)
    model.load_state_dict(checkpoint['model'])
    model.eval()
    dataset_name = dataset

    layer = [i for i in range(0, layer_num+1, 1)]
    with torch.no_grad():
        if dataset_name == 'KUL':
            file_list = glob.glob(os.path.join(wav_path, '*dry.wav')) #KUL
        else:
            file_list = glob.glob(os.path.join(wav_path, '*.wav'))  # DTU
        for file in file_list:
            file_name = os.path.basename(file)
            sr, wav = wavfile.read(file)
            wav = resample(wav, int(wav.shape[0] * 16000 / sr))
            sr = 16000
            feat_patch = []
            for sp_st in tqdm(range(0, len(wav), int(sr*window*overlap))):  # 5s
                wav_i = wav[sp_st:sp_st + int(sr*window)]
                if len(wav_i) < 500:
                    continue
                wav_input_16khz = torch.tensor(wav_i).unsqueeze(0)
                # wav_input = feature_extractor(wav_i, return_tensors='pt', sampling_rate=sr)
                # input_values = wav_input.input_values.half().to(device)
                # o = model(input_values, output_hidden_states=True)

                if cfg.normalize:
                    wav_input_16khz = torch.nn.functional.layer_norm(wav_input_16khz, wav_input_16khz.shape)
                rep, layer_results = model.extract_features(wav_input_16khz.float(), output_layer=model.cfg.encoder_layers, ret_layer_results=True)[0]
                layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

                feat_layer = []
                for l in range(len(layer)):
                    feat_layer.append(layer_reps[l].detach().squeeze().cpu().numpy())
                if len(feat_layer) == len(layer):
                    feat_patch.append(feat_layer)
            # wav_time = [len(feat_patch[i]) for i in range(len(feat_patch))]
            wav_time = []
            for data in feat_patch:
                data_dim = []
                End = True
                for dim in data:
                    if len(dim.shape) == 2:
                        data_dim.append(dim.shape[0])
                    else:
                        End = False
                if End :
                    wav_time.append(data_dim)
                else:
                    feat_patch = feat_patch[:-1]

            # add_index = [index for index, item in enumerate(wav_time) if item != 50]
            add_index = [[index for index, item in enumerate(wav_time[i]) if item !=int(50*window) ] for i in range(len(wav_time))]

            if len(add_index) != 0:
                for i in range(len(add_index)):
                    if add_index[i] != []:
                        dim_index = add_index[i]
                        # print(len(wav_time[i]))
                        # if len(dim_index) == len(wav_time[i]):
                        for d in dim_index:
                            # if len(feat_patch[add_index[i]].shape) == 2:
                            if len(feat_patch[i][d].shape) == 2:
                                if wav_time[i][d] < int(50*window):
                                    feat_patch[i][d] = np.concatenate(
                                        (feat_patch[i][d], np.zeros((int(50*window) - wav_time[i][d], 1024))), axis=0)
                                elif wav_time[i][d] > int(50*window):
                                    feat_patch[i][d] = feat_patch[i][d][:int(50*window), :]
                                wav_time[i][d] = int(50*window)
                            else:
                                feat_patch[i].pop(d)
                                wav_time[i].pop(d)
                        # else:
                        #     feat_patch.pop(i)
                        #     wav_time.pop(i)
            print(f'PCA feat wavlm layer {layer}')
            sample = len(feat_patch)
            layer_number = len(feat_patch[0])
            time, dim = feat_patch[0][0].shape
            feat = np.concatenate(feat_patch).reshape((-1, dim))
            pca = PCA(n_components=pca_dim)
            feat_pca = pca.fit_transform(feat)
            t = np.cumsum(wav_time)[:-1]
            feat_pca_split = np.split(feat_pca, np.cumsum(wav_time)[:-1])

            emb = np.array(feat_pca_split).astype(np.float64)
            emb = emb.reshape((sample, layer_number, time, pca_dim))
            zoom_factors = (1, 1, target_fs / 50, 1)
            emb_res = zoom(emb, zoom_factors)

            emb_res = emb_res.astype(np.float32)
            if overlap == 1:
                newname = rename(file_name, name='wavlm_' + f'{layer_num}layer'+ f'_{pca_dim}pca' +f'_{window}s', dataset=dataset_name)
            else:
                newname = rename(file_name, name='wavlm_' + f'{layer_num}layer' + f'_{pca_dim}pca' + f'_{window}s_overlap', dataset=dataset_name)
            print(newname)
            np.save(os.path.join(save_path, newname), emb_res)

def get_whisper_all(wav_path, save_path, target_fs=128, device=0, layer_num=14, pca_dim=64, window=1, overlap=1):
    # load the pre-trained checkpoints
    model_path = "C:\\Users\DontSoRry\Desktop\whisper/"
    feature_extractor = AutoFeatureExtractor.from_pretrained(model_path)
    model = WhisperModel.from_pretrained(model_path)
    embed_positions = copy.deepcopy(model.encoder.embed_positions.weight).to('cuda')
    model = model.to(device)
    model.eval()

    layer = [i for i in range(0, layer_num+1, 1)]
    with torch.no_grad():
        #file_list = glob.glob(os.path.join(wav_path, '*dry.wav')) #KUL
        file_list = glob.glob(os.path.join(wav_path, '*.wav'))  # DTU
        for file in file_list:
            file_name = os.path.basename(file)
            sr, wav = wavfile.read(file)
            wav = resample(wav, int(wav.shape[0] * 16000 / sr))
            sr = 16000
            feat_patch = []
            for sp_st in tqdm(range(0, len(wav), int(sr*window*overlap))):  # 5s
                wav_i = wav[sp_st:sp_st + sr*window]
                if len(wav_i) < 500:
                    continue

                wav_input = feature_extractor(wav_i, padding=False, return_tensors='pt', sampling_rate=sr)
                input_feature = wav_input.input_features
                max_len = input_feature.shape[-1]
                input_lengths = (max_len - 1) // 2 + 1
                input_feature = input_feature.to(device)
                model.encoder.embed_positions = model.encoder.embed_positions.from_pretrained(embed_positions[:input_lengths])
                feat = model.encoder(input_feature, output_hidden_states=True).hidden_states
                #stacked_feature = torch.stack(feat, dim=0)[1:]

                layer_reps = [x[0] for x in feat]
                #layer_reps = [x.transpose(0, 1) for x, _ in layer_results]

                feat_layer = []
                for l in range(len(layer)):
                    feat_layer.append(layer_reps[l].detach().squeeze().cpu().numpy())
                if len(feat_layer) == len(layer):
                    feat_patch.append(feat_layer)
            # wav_time = [len(feat_patch[i]) for i in range(len(feat_patch))]
            wav_time = []
            for data in feat_patch:
                data_dim = []
                End = True
                for dim in data:
                    if len(dim.shape) == 2:
                        data_dim.append(dim.shape[0])
                    else:
                        End = False
                if End :
                    wav_time.append(data_dim)
                else:
                    feat_patch = feat_patch[:-1]

            # add_index = [index for index, item in enumerate(wav_time) if item != 50]
            add_index = [[index for index, item in enumerate(wav_time[i]) if item !=50*window ] for i in range(len(wav_time))]

            if len(add_index) != 0:
                for i in range(len(add_index)):
                    if add_index[i] != []:
                        dim_index = add_index[i]
                        # print(len(wav_time[i]))
                        # if len(dim_index) == len(wav_time[i]):
                        for d in dim_index:
                            # if len(feat_patch[add_index[i]].shape) == 2:
                            if len(feat_patch[i][d].shape) == 2:
                                if wav_time[i][d] < 50*window:
                                    feat_patch[i][d] = np.concatenate(
                                        (feat_patch[i][d], np.zeros((50*window - wav_time[i][d], 1280))), axis=0)
                                elif wav_time[i][d] > 50*window:
                                    feat_patch[i][d] = feat_patch[i][d][:50*window, :]
                                wav_time[i][d] = 50*window
                            else:
                                feat_patch[i].pop(d)
                                wav_time[i].pop(d)
                        # else:
                        #     feat_patch.pop(i)
                        #     wav_time.pop(i)
            print(f'PCA feat whisper layer {layer}')
            sample = len(feat_patch)
            layer_number = len(feat_patch[0])
            time, dim = feat_patch[0][0].shape
            feat = np.concatenate(feat_patch).reshape((-1, dim))
            pca = PCA(n_components=pca_dim)
            feat_pca = pca.fit_transform(feat)
            t = np.cumsum(wav_time)[:-1]
            feat_pca_split = np.split(feat_pca, np.cumsum(wav_time)[:-1])

            emb = np.array(feat_pca_split).astype(np.float64)
            emb = emb.reshape((sample, layer_number, time, pca_dim))
            zoom_factors = (1, 1, target_fs / 50, 1)
            emb_res = zoom(emb, zoom_factors)

            emb_res = emb_res.astype(np.float32)
            newname = rename(file_name, name='whisper_' + f'{layer_num}layer'+f'_{window}s')
            print(newname)
            np.save(os.path.join(save_path, newname), emb_res)


def get_audio(path='./EEGViT/stimulus/'):
    trials,channels = 20,64
    window = 1
    sample_rate = 128
    sr = 16000
    overlap = 1  #0.5
    #data = np.empty((0, 360, window*sample_rate, 64))
    data = []
    root = '/media/nchen/CYB/KUL/preprocessed_data_50hz_mean/'
    audio_path = '/media/nchen/CYB/KUL/stimuli/'
    label = []
    nSub = 1
    #condition = dry
    # train data
    data_mat = scipy.io.loadmat(root + 'S%d.mat' % nSub)
    if not os.path.exists(path):
        os.makedirs(path)

    for k_tra in range(trials):
        #if data_mat['preproc_trials'][0,k_tra]['condition'] == condition:
        #mat_data = data_mat['preproc_trials'][0, k_tra]['RawData'][0, 0]['EegData']
        stimulus = data_mat['preproc_trials'][0, k_tra]['stimuli'][0,0][:,0]
        # envelope = data_mat['preproc_trials'][0, k_tra]['Envelope'][0, 0]['AudioData']
        table = str.maketrans('','', "'[]")
        stimulus1 = str(stimulus[0]).translate(table)
        stimulus2 = str(stimulus[1]).translate(table)
        fs1, audio1 = wavfile.read(os.path.join(audio_path, stimulus1))
        audio1 = resample(audio1, int(audio1.shape[0] * sr/fs1))
        fs2, audio2 = wavfile.read(os.path.join(audio_path, stimulus2))
        audio2 = resample(audio2, int(audio2.shape[0] * sr/fs2))
        stimulus1_name = rename(stimulus1, 'audio')
        stimulus2_name = rename(stimulus2, 'audio')
        file_path1 = os.path.join(path, f'{stimulus1_name}')
        file_path2 = os.path.join(path, f'{stimulus2_name}')
        if not os.path.exists(file_path1):
            #os.makedirs(stimulus1_name)
            # envelope1 = np.sum(envelope[0][0][:,:,0], axis=1, keepdims=True)
            audio1_split = split_feature(audio1, window, sr, overlap)
            np.save(file_path1, audio1_split)
        if not os.path.exists(file_path2):
            #os.makedirs(stimulus2_name)
            # envelope2 = np.sum(envelope[0][0][:,:,1], axis=1, keepdims=True)
            audio2_split = split_feature(audio2, window, sr, overlap)
            np.save(file_path2, audio2_split)


def frame_signal(signal, frame_length, hop_size):
    """
    对信号进行分帧处理

    :param signal: 输入信号（1D numpy array）
    :param frame_length: 窗长（采样点数）
    :param hop_size: 窗移（采样点数）
    :return: 分帧后的信号，形状为 (num_frames, frame_length)
    """
    num_frames = (len(signal) - frame_length) // hop_size + 1
    frames = np.zeros((num_frames, frame_length))

    for i in range(num_frames):
        start_idx = i * hop_size
        end_idx = start_idx + frame_length
        frames[i] = signal[start_idx:end_idx]

    return frames

def get_lpc(wav_path, save_path, target_fs=128, number_filters=16, window=1, overlap=1, feat_name = 'lpc', dataset="KUL"):
    if dataset == "KUL":
        file_list = glob.glob(os.path.join(wav_path, '*dry.wav'))
    else:
        file_list = glob.glob(os.path.join(wav_path, '*.wav'))
    for file in file_list:
        file_name = os.path.basename(file)
        # fs, audio = wavfile.read(file)
        # new_fs = 16000
        # resample_audio = librosa.resample(audio, orig_sr=fs, target_sr=new_fs)
        audio, fs = librosa.load(file, sr=16000)
        frame_data = frame_signal(audio, frame_length=512, hop_size=125)
        #mel_data = calculate_mel_spectrogram(audio_path=file, target_fs=target_fs, nb_filters=number_filters)
        lpc_data = librosa.lpc(frame_data, order=number_filters)
        data = split_feature(lpc_data[:,1:], window=window, sample_rate=target_fs, overlap=overlap)
        # data_mean = np.mean(data, axis=2, keepdims=True)
        # # data_std = np.std(data, axis=1, keepdims=True)
        # # data = (data-data_mean)/data_std
        # data_max = np.max(data, axis=2, keepdims=True)
        # data_min = np.min(data, axis=2, keepdims=True)
        # data = (data-data_min) / (data_max-data_min + 1e-6)
        newname = rename_new(file_name, name=feat_name, window=window, dataset=dataset)
        np.save(os.path.join(save_path, newname), data)
# get_audio()
# get_wav2vec_all(wav_path='D:/KUL/stimuli', save_path='D:/EEGViT/stimulus', window=1, overlap=1)
# get_envelope()

# fs, audio = wavfile.read('/media/nchen/CYB/KUL/stimuli/part1_track1_hrtf.wav')
# print(len(audio)/fs)
# data = calculate_mel_spectrogram(audio_path = '/media/nchen/CYB/KUL/stimuli/part1_track1_hrtf.wav', target_fs=128)
# print(data.shape)

# get_melspectrogram(wav_path='D:\KUL/stimuli', save_path='D:/EEGViT/stimulus', feat_name='melnormlogmin_28')
# get_log_melspectrogram(wav_path='D:\KUL/stimuli', save_path='D:/EEGViT/stimulus',window=3, overlap=1/3, number_filters=16)
# get_log_melspectrogram(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', window=3, overlap=1/3, number_filters=16)
# get_log_melspectrogram(wav_path='D:\KUL/stimuli', save_path='D:/EEGViT/stimulus',window=0.5, overlap=1, number_filters=16, dataset='KUL')
# get_log_melspectrogram(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', window=0.5, overlap=1, number_filters=16, dataset='DTU')
# get_log_melspectrogram(wav_path='D:\KUL/stimuli', save_path='D:/EEGViT/stimulus',window=1.5, overlap=1, number_filters=16, dataset='KUL')
# get_log_melspectrogram(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', window=1.5, overlap=1, number_filters=16, dataset='DTU')
# get_log_melspectrogram(wav_path='E:/ESAA/stim', save_path='D:/EEGViT/stimulus_ESAA',window=2, overlap=1/2, number_filters=16)
# get_log_melspectrogram(wav_path='D:/New_AAD_Dataset/audiobooks/fM', save_path='D:/EEGViT/stimulus_NewDataset/fM',number_filters=16)
# get_log_melspectrogram(wav_path='D:/New_AAD_Dataset/audiobooks/fW', save_path='D:/EEGViT/stimulus_NewDataset/fW',number_filters=16)
# get_log_melspectrogram(wav_path='E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/raw_stimulus', save_path='D:/EEGViT/stimulus_NJU', window=0.5, overlap=1, number_filters=16, dataset='NJU')
# get_log_melspectrogram(wav_path='E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/raw_stimulus', save_path='D:/EEGViT/stimulus_NJU', window=1.5, overlap=1, number_filters=16, dataset='NJU')
# get_wav2vec_all(wav_path='D:\KUL/stimuli', save_path='D:/EEGViT/stimulus', window=2, overlap=1/2)
# get_wav2vec_all(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', window=2, overlap=1/2, dataset='DTU')
# get_wav2vec_all(wav_path='E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/raw_stimulus', save_path='D:/EEGViT/stimulus_NJU', window=2, overlap=1/2, dataset='NJU')
# get_wavlm_all(wav_path='D:/KUL/stimuli', save_path='D:/EEGViT/stimulus', layer_num=24, pca_dim=64, window=2, overlap=1/2)
# get_wavlm_all(wav_path='D:/KUL/stimuli', save_path='D:/EEGViT/stimulus', layer_num=24, pca_dim=64, window=0.5, overlap=1, dataset='KUL')
# get_wavlm_all(wav_path='D:/KUL/stimuli', save_path='D:/EEGViT/stimulus', layer_num=24, pca_dim=64, window=1.5, overlap=2/3, dataset='KUL')
# get_wavlm_all(wav_path='D:/KUL/stimuli', save_path='D:/EEGViT/stimulus', layer_num=24, pca_dim=64, window=3, overlap=1/3)
# get_wavlm_all(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', layer_num=24, pca_dim=64, window=0.5, overlap=1, dataset='DTU')
# get_wavlm_all(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', layer_num=24, pca_dim=64, window=1.5, overlap=2/3, dataset='DTU')
# get_wavlm_all(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', layer_num=24, pca_dim=64, window=2, overlap=1/2, dataset='DTU')
# get_wavlm_all(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', layer_num=24, pca_dim=64, window=3, overlap=1/3, dataset='DTU')
# get_wavlm_all(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', layer_num=24, pca_dim=64, window=1, overlap=1)
# get_wavlm_all(wav_path='E:/ESAA/stim', save_path='D:/EEGViT/stimulus_ESAA', layer_num=24, pca_dim=16, window=1, overlap=1)
# get_wavlm_all(wav_path='E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/raw_stimulus', save_path='D:/EEGViT/stimulus_NJU', layer_num=24, pca_dim=64, window=0.5, overlap=1, dataset='NJU')
# get_wavlm_all(wav_path='E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/raw_stimulus', save_path='D:/EEGViT/stimulus_NJU', layer_num=24, pca_dim=64, window=1.5, overlap=2/3, dataset='NJU')
# get_wavlm_all(wav_path='E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/raw_stimulus', save_path='D:/EEGViT/stimulus_NJU', layer_num=24, pca_dim=64, window=2, overlap=1/2)
# get_wavlm_all(wav_path='E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/raw_stimulus', save_path='D:/EEGViT/stimulus_NJU', layer_num=24, pca_dim=64, window=3, overlap=1/3)
#get_wavlm_all(wav_path='D:/New_AAD_Dataset/audiobooks/fM', save_path='D:/EEGViT/stimulus_NewDataset/fM', layer_num=24, pca_dim=64, window=1, overlap=1)
#get_wavlm_all(wav_path='D:/New_AAD_Dataset/audiobooks/fW', save_path='D:/EEGViT/stimulus_NewDataset/fW', layer_num=24, pca_dim=64, window=1, overlap=1)
# get_hubert_all(wav_path='D:/KUL/stimuli', save_path='D:/EEGViT/stimulus', layer_num=24, pca_dim=64, window=1, overlap=1)
# get_wav2vec_all_24(wav_path='D:\KUL/stimuli', save_path='D:/EEGViT/stimulus')
# get_lpc(wav_path='D:\KUL/stimuli', save_path='D:/EEGViT/stimulus', window=0.5, overlap=1, feat_name='lpc', dataset="KUL")
# get_lpc(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', window=0.5, overlap=1, feat_name='lpc', dataset="DTU")
# get_lpc(wav_path='D:\KUL/stimuli', save_path='D:/EEGViT/stimulus', window=1.5, overlap=1, feat_name='lpc', dataset="KUL")
# get_lpc(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', window=1.5, overlap=1, feat_name='lpc', dataset="DTU")
# get_lpc(wav_path='E:/ESAA/stim', save_path='D:/EEGViT/stimulus_ESAA', feat_name='lpc')
# get_lpc(wav_path='D:/New_AAD_Dataset/audiobooks/fM', save_path='D:/EEGViT/stimulus_NewDataset/fM',feat_name='lpc')
# get_lpc(wav_path='D:/New_AAD_Dataset/audiobooks/fW', save_path='D:/EEGViT/stimulus_NewDataset/fW',feat_name='lpc')
# get_lpc(wav_path='E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/raw_stimulus', save_path='D:/EEGViT/stimulus_NJU', window=0.5, overlap=1, feat_name='lpc', dataset="NJU")
# get_lpc(wav_path='E:/NanJingUniversity/NJUNCA_preprocessed_arte_removed/raw_stimulus', save_path='D:/EEGViT/stimulus_NJU', window=1.5, overlap=1, feat_name='lpc', dataset="NJU")
#get_whisper_all(wav_path='E:/DTU/AUDIO', save_path='D:/EEGViT/stimulus_DTU', layer_num=24, pca_dim=64, window=1, overlap=1)
#get_whisper_all(wav_path='E:/ESAA/stim', save_path='D:/EEGViT/stimulus_ESAA', layer_num=24, pca_dim=64, window=1, overlap=1)
# get_whisper_all(wav_path='D:/KUL/stimuli', save_path='D:/EEGViT/stimulus', layer_num=24, pca_dim=64, window=1, overlap=1)
get_envelope_NJU()