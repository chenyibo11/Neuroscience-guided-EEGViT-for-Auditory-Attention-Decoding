import matplotlib.pyplot as plt

# from dalle2_pytorch import Decoder
from models.EEGViT_pretrained import EEGViT_pretrained
from models.EEGViT import EEGViT_raw
from models.ViTBase import ViTBase
from models.ViTBase_pretrained import ViTBase_pretrained
from models.EEGViT_KUL import EEGViT_KUL_CL, EEGViT_KUL_pretrained_Restruct, EEGViT_KUL_pretrained_wav2vec_frequency_sum_768
from dataset.EEGEyeNet import EEGEyeNetDataset
from dataset.DTU import data_read_CL_feature_path_DTU
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
from tqdm import tqdm
import numpy as np
import os
import math
import argparse

import yaml
from pathlib import Path
import hydra
from hydra import utils
# from omegaconf import DictConfig
from itertools import chain
# from torch.utils.tensorboard import SummaryWriter
# from VQVAE.ZeroSpeech.model_new import Encoder, Decoder
from torch.autograd import Variable
from scipy import signal, fft
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

'''
models: EEGViT_pretrained; EEGViT_raw; ViTBase; ViTBase_pretrained
'''

def pearson_pt(y_true, y_pred, axis=1):
    """
    Pearson correlation function implemented in PyTorch.

    Parameters
    ----------
    y_true: torch.Tensor
        Ground truth labels. Shape is (batch_size, time_steps, n_features)
    y_pred: torch.Tensor
        Predicted labels. Shape is (batch_size, time_steps, n_features)
    axis: int
        Axis along which to compute the Pearson correlation. Default is 1.

    Returns
    -------
    torch.Tensor
        Pearson correlation.
        Shape is (batch_size, 1, n_features) if axis is 1.
    """
    # Compute the mean of the true and predicted values
    y_true_mean = torch.mean(y_true, dim=axis, keepdim=True)
    y_pred_mean = torch.mean(y_pred, dim=axis, keepdim=True)

    # Compute the numerator and denominator of the Pearson correlation
    numerator = torch.sum((y_true - y_true_mean) * (y_pred - y_pred_mean), dim=axis, keepdim=True)
    std_true = torch.sum((y_true - y_true_mean) ** 2, dim=axis, keepdim=True)
    std_pred = torch.sum((y_pred - y_pred_mean) ** 2, dim=axis, keepdim=True)
    denominator = torch.sqrt(std_true * std_pred + 1e-6)

    # Compute the Pearson correlation
    return torch.mean(torch.div(numerator, denominator).nan_to_num(), dim=-1)

class CLUB(nn.Module):  # CLUB: Mutual Information Contrastive Learning Upper Bound
    '''
        https://arxiv.org/abs/2006.12013
        This class provides the CLUB estimation to I(X,Y) 
        Method:
            forward() :      provides the estimation with input samples  
            loglikeli() :   provides the log-likelihood of the approximation q(Y|X) with input samples
        Arguments:
            x_dim, y_dim :         the dimensions of samples from X, Y respectively
            hidden_size :          the dimension of the hidden layer of the approximation network q(Y|X)
            x_samples, y_samples : samples from X and Y, having shape [sample_size, x_dim/y_dim] 
    '''
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUB, self).__init__()
        # p_mu outputs mean of q(Y|X)
        #print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim))
        # p_logvar outputs log of variance of q(Y|X)
        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
    
    def forward(self, x_samples, y_samples): 
        mu, logvar = self.get_mu_logvar(x_samples)
        # print(mu.shape)
        # print(logvar.shape)
        # print(y_samples.shape)
        # exit()
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2./logvar.exp()  
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2./logvar.exp() 

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

class CLUBSample(nn.Module):  # Sampled version of the CLUB estimator
    # x_dim 512 y_dim 256
    def __init__(self, x_dim, y_dim, hidden_size):
        super(CLUBSample, self).__init__()
        self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size*2),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(hidden_size*2, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, y_dim))

        self.p_logvar = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, hidden_size*2),
                                       nn.ReLU(),
                                       nn.Dropout(0.2),
                                       nn.Linear(hidden_size*2, hidden_size),
                                       nn.ReLU(),
                                       nn.Linear(hidden_size, y_dim),
                                       nn.Tanh())

    def get_mu_logvar(self, x_samples):
        mu = self.p_mu(x_samples)
        logvar = self.p_logvar(x_samples)
        return mu, logvar
     
        
    def loglikeli(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2 /logvar.exp()-logvar).mean() #.sum(dim=1).mean(dim=0)
    

    def forward(self, x_samples, y_samples):
        mu, logvar = self.get_mu_logvar(x_samples)
        
        sample_size = x_samples.shape[0]
        #random_index = torch.randint(sample_size, (sample_size,)).long()
        random_index = torch.randperm(sample_size).long()
        
        positive = - (mu - y_samples)**2 / logvar.exp()
        negative = - (mu - y_samples[random_index])**2 / logvar.exp()
        upper_bound = (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()
        return upper_bound/2.

    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)
    
class CLUBMean(nn.Module):  # Set variance of q(y|x) to 1, logvar = 0. Update 11/26/2022
    def __init__(self, x_dim, y_dim, hidden_size=None):
        # p_mu outputs mean of q(Y|X)
        # print("create CLUB with dim {}, {}, hiddensize {}".format(x_dim, y_dim, hidden_size))
        
        super(CLUBMean, self).__init__()
   
        if hidden_size is None:
            self.p_mu = nn.Linear(x_dim, y_dim)
        else:
            # self.p_mu = nn.Sequential(nn.Linear(x_dim, int(hidden_size)),
            #                            nn.ReLU(),
            #                            nn.Linear(int(hidden_size), y_dim))
            self.p_mu = nn.Sequential(nn.Linear(x_dim, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, hidden_size*2),
                                  nn.ReLU(),
                                  nn.Dropout(0.2),
                                  nn.Linear(hidden_size*2, hidden_size),
                                  nn.ReLU(),
                                  nn.Linear(hidden_size, y_dim))


    def get_mu_logvar(self, x_samples):
        # variance is set to 1, which means logvar=0
        mu = self.p_mu(x_samples)
        return mu, 0
    
    def forward(self, x_samples, y_samples):

        mu, logvar = self.get_mu_logvar(x_samples)
        
        # log of conditional probability of positive sample pairs
        positive = - (mu - y_samples)**2 /2.
        
        prediction_1 = mu.unsqueeze(1)          # shape [nsample,1,dim]
        y_samples_1 = y_samples.unsqueeze(0)    # shape [1,nsample,dim]

        # log of conditional probability of negative sample pairs
        negative = - ((y_samples_1 - prediction_1)**2).mean(dim=1)/2.

        return (positive.sum(dim = -1) - negative.sum(dim = -1)).mean()

    def loglikeli(self, x_samples, y_samples): # unnormalized loglikelihood 
        mu, logvar = self.get_mu_logvar(x_samples)
        return (-(mu - y_samples)**2).sum(dim=1).mean(dim=0)
    
    def learning_loss(self, x_samples, y_samples):
        return - self.loglikeli(x_samples, y_samples)

def get_contrastive(eeg, stimulus_att, stimulus_unatt):
    # 扩展维度实现全样本对广播  
    eeg_expanded = eeg.unsqueeze(1)   # [8, 1, 64, 128]  
    stimulus_att = stimulus_att.unsqueeze(0)   # [1, 8, 64, 128]  
    stimulus_unatt = stimulus_unatt.unsqueeze(0)
    
    # 计算余弦相似度（沿dim=-1即128维计算）  
    cos_sim_att = F.cosine_similarity(eeg_expanded,  stimulus_att, dim=-1)  # [8, 8, 64]  

    cos_sim_unatt = F.cosine_similarity(eeg_expanded,  stimulus_unatt, dim=-1)
    
    # 沿64维取平均  
    cos_sim_att = cos_sim_att.mean(dim=-1)   # [8, 8]  
    cos_sim_unatt = cos_sim_unatt.mean(dim=-1)
    sim = torch.cat([torch.diag(cos_sim_att).unsqueeze(-1), torch.diag(cos_sim_unatt).unsqueeze(-1)], dim=-1)

    return sim
    #return cos_sim_att

import torch
import numpy as np
from sklearn.decomposition  import PCA
from sklearn.manifold  import TSNE
import matplotlib.pyplot  as plt


def plotPCA(data1, data2, data3, data4):
    # 合并数据
    combined = torch.cat([data1,  data2, data3, data4], dim=0).cpu().detach().numpy()

    # 降维，这里以PCA为例
    pca = PCA(n_components=2)
    projected = pca.fit_transform(combined) 
    # tsne = TSNE(n_components=2, perplexity=15, random_state=42)
    # projected = tsne.fit_transform(combined)

    plt.figure()
    # 绘制
    plt.scatter(projected[1:65,  0], projected[1:65, 1], label='Speech1 (64 samples)')
    plt.scatter(projected[0,  0], projected[0, 1], color='blue', marker='*', s=100, label='EEG1 (1 sample)')
    plt.scatter(projected[66:,  0], projected[66:, 1], label='Speech2 (64 samples)')
    plt.scatter(projected[65,  0], projected[65, 1], color='red', marker='*', s=100, label='EEG2 (1 sample)')
    plt.legend() 
    # plt.title('PCA  Visualization of Contrastive Space')
    plt.title('tSNE Visualization of Contrastive Space')
    # plt.show() 
    plt.savefig('/media/c1/CYB/EEGViT/picture/222.png')
    plt.close()

def min_max_norm(tensor):
    # min_val = (tensor.min(dim=-1).values).unsqueeze(-1)   # 按特征维度计算最小值 
    # max_val = (tensor.max(dim=-1).values).unsqueeze(-1) 
    # return (tensor - min_val) / (max_val - min_val + 1e-8)
    mean = tensor.mean(dim=-1).unsqueeze(-1)     # 按特征维度计算均值 
    std = tensor.std(dim=-1).unsqueeze(-1) 
    return (tensor - mean) / (std + 1e-8)
    #max_val = torch.abs(tensor).max(dim=-1).values.unsqueeze(-1)
    #j = torch.ceil(torch.log10(max_val))
    #normalized_data = tensor / (10 ** j)
    #return normalized_data

def pairwise_orthogonal_loss(a, b, normalized=True):
    dot_products = (a * b).sum(dim=1)  # Shape: (batch,)
    if normalized:
        a_norm = torch.norm(a,  p=2, dim=1)  # L2 norm per sample 
        b_norm = torch.norm(b,  p=2, dim=1)
        loss = (dot_products / (a_norm * b_norm + 1e-8)) ** 2 
    else:
        loss = dot_products ** 2 
    return loss.mean() 

def intra_matrix_orthogonal_loss(matrix, normalized=True):
    if normalized:
        matrix = matrix / (torch.norm(matrix,  p=2, dim=1, keepdim=True) + 1e-8)
    gram_matrix = matrix @ matrix.T  # Shape: (batch, batch)
    identity = torch.eye(gram_matrix.size(0),  device=matrix.device) 
    off_diagonal = gram_matrix - identity  # 仅保留非对角线元素（假设已归一化）
    loss = torch.norm(off_diagonal,  p='fro') ** 2  # Frobenius 范数平方 
    return loss 

def cross_matrix_orthogonal_loss(a, b, normalized=False):
    if normalized:
        a = a / (torch.norm(a,  p=2, dim=1, keepdim=True) + 1e-8)
        b = b / (torch.norm(b,  p=2, dim=1, keepdim=True) + 1e-8)
    gram_matrix = a @ b.T  # Shape: (batch, batch)
    loss = torch.mean(gram_matrix  ** 2)  # 所有元素平方的均值 
    return loss 

class LabelBasedSupConLoss(nn.Module):
    def __init__(self, temperature=0.7, eeg_band=5, band_index=[0, 1, 2, 3, 4], feature_dim=1):
        super(LabelBasedSupConLoss, self).__init__()
        self.temperature = temperature
        self.band = eeg_band
        self.band_index = band_index
        self.dim = feature_dim
        self.criterion = ProbCrossEntropyLoss()
        #self.MI = CLUB(x_dim=128, y_dim=128, hidden_size=64)

    def forward(self, feature_eeg, feature_stim, labels, Classify_att_high, Classify_unatt_high, Classify_att_low, Classify_unatt_low, Criterion, layer=1, training=True, optimer_MI=None):
        """
        Args:
            features: tensor of shape [batch_size, 3, feature_dim], where:
                      - features[:, 0, :] is the EEG embedding
                      - features[:, 1, :] is the stimulus1 embedding
                      - features[:, 2, :] is the stimulus2 embedding
            labels: tensor of shape [batch_size], with values 1 or 2, which indicates:
                    - 1: stimulus1 is the positive sample
                    - 2: stimulus2 is the positive sample
        Returns:
            A scalar loss value.
        """
        # 确保输入的形状符合 [batch_size, 3, feature_dim]
        # assert features.shape[1] == 1 + self.dim*2, "Each sample must have 3 components: [eeg, stimulus1, stimulus2]"
        # print(features.shape)
        # print(self.band + self.dim * 2)
        self.MI = Criterion
        self.band = 3
        #assert features.shape[1] == self.band + self.dim * 2, "Each sample must have 3 components: [eeg, stimulus1, stimulus2]"
        # 分离 eeg, stimulus1, stimulus2
        # stimulus1 = feature_stim[:, :self.dim]  # [batch_size, feature_dim]
        # stimulus2 = feature_stim[:, self.dim:]
        stimulus1 = min_max_norm(feature_stim[:, :self.dim])  # [batch_size, feature_dim]
        stimulus2 = min_max_norm(feature_stim[:, self.dim:]) 
        if layer < 3:
            eeg = feature_eeg[:, :2].float()
            eeg_att = Classify_att_low(eeg)
            eeg_unatt = Classify_unatt_low(eeg)
            #eeg_att = min_max_norm(Classify_att_low(eeg))
            #eeg_unatt = min_max_norm(Classify_unatt_low(eeg))
            self.band = 2
            self.band_index = [0, 1]
            self.layer = 2
        else:
            eeg = feature_eeg[:, 2].unsqueeze(1).float()# [batch_size, eeg_dim]
            eeg_att = Classify_att_high(eeg)
            eeg_unatt = Classify_unatt_high(eeg)
            #eeg_att = min_max_norm(Classify_att_high(eeg))
            #eeg_unatt = min_max_norm(Classify_unatt_high(eeg))
            self.band = 1
            self.band_index = [0]
            self.layer = 5
         # [batch_size, feature_dim]

        if training:
            # 创建 mask，根据标签来选择 stimulus1 还是 stimulus2 作为正样本
            #eeg = eeg.unsqueeze(2).repeat(1, 1, stimulus1.shape[1], 1)
            eeg_att = eeg_att.unsqueeze(2).repeat(1, 1, stimulus1.shape[1], 1)
            eeg_unatt = eeg_unatt.unsqueeze(2).repeat(1, 1, stimulus1.shape[1], 1)
            # pos_mask = (labels == 1).float().view(-1, 1)  # [batch_size, 1] -> 选择 stimulus1 的 mask
            # neg_mask = (labels == 2).float().view(-1, 1)  # [batch_size, 1] -> 选择 stimulus2 的 mask

            # # 根据标签选择正样本和负样本
            # pos_sim = (F.cosine_similarity(eeg, stimulus1, dim=2)).mean(dim=1) * pos_mask.squeeze() + (
            #     F.cosine_similarity(eeg, stimulus2, dim=2)).mean(dim=1) * neg_mask.squeeze()
            # neg_sim = (F.cosine_similarity(eeg, stimulus1, dim=2)).mean(dim=1) * neg_mask.squeeze() + (
            #     F.cosine_similarity(eeg, stimulus2, dim=2)).mean(dim=1) * pos_mask.squeeze()

            # mask = pos_mask.squeeze()[:, None, None]
            # att_stim = torch.where(mask == 1, stimulus1, stimulus2)

            # batch_sim = (F.cosine_similarity(eeg[:, 0, :, :].unsqueeze(1), stimulus1.unsqueeze(0), dim=-1).mean(dim=-1) +
            #            F.cosine_similarity(eeg[:, 1, :, :].unsqueeze(1), stimulus1.unsqueeze(0), dim=-1).mean(dim=-1) +
            #            F.cosine_similarity(eeg[:, 2, :, :].unsqueeze(1), stimulus1.unsqueeze(0), dim=-1).mean(dim=-1) +
            #            F.cosine_similarity(eeg[:, 3, :, :].unsqueeze(1), stimulus1.unsqueeze(0), dim=-1).mean(dim=-1) +
            #            F.cosine_similarity(eeg[:, 4, :, :].unsqueeze(1), stimulus1.unsqueeze(0), dim=-1).mean(dim=-1)) / 5
            # print('eeg_shape', eeg.shape)
            # batch_sim = F.cosine_similarity(eeg[:, 0, :, :].unsqueeze(1), stimulus1.unsqueeze(0), dim=-1).mean(dim=-1)
            # if self.band != 1:
            #     for b in range(1, self.band):
            #         batch_sim = batch_sim + F.cosine_similarity(eeg[:, self.band_index[b], :, :].unsqueeze(1), stimulus1.unsqueeze(0), dim=-1).mean(dim=-1)
            # batch_sim = batch_sim / self.band
            # optim_loss = torch.abs(F.cosine_similarity(eeg_att[:, 0, :, :].unsqueeze(1), eeg_unatt[:, 0, :, :].unsqueeze(1), dim=-1).mean(dim=-1)).to(eeg_att.device) #+ (eeg_att[:, 0, 0, :] - eeg_unatt[:, 0, 0, :]).mean(dim=-1)
            # if self.band != 1:
            #     for b in range(1, self.band):
            #         optim_loss = optim_loss + torch.abs(F.cosine_similarity(eeg_att[:, self.band_index[b], :, :].unsqueeze(1), eeg_unatt[:, self.band_index[b], :, :].unsqueeze(1), dim=-1).mean(dim=-1)).to(eeg_att.device) #+ (eeg_att[:, self.band_index[b], 0, :] - eeg_unatt[:, self.band_index[b], 0, :]).mean(dim=-1)
            # optim_loss = optim_loss / self.band
            
            self.MI.train()
            optim_loss = self.MI.learning_loss(eeg_att[:, 0, 0, :], eeg_unatt[:, 0, 0, :]).mean(dim=-1) #+ (eeg_att[:, 0, 0, :] - eeg_unatt[:, 0, 0, :]).mean(dim=-1)
            if self.band != 1:
                for b in range(1, self.band):
                    optim_loss = optim_loss + self.MI.learning_loss(eeg_att[:, self.band_index[b], 0, :], eeg_unatt[:, self.band_index[b], 0, :]).mean(dim=-1) #+ (eeg_att[:, self.band_index[b], 0, :] - eeg_unatt[:, self.band_index[b], 0, :]).mean(dim=-1)
            optim_loss = optim_loss / self.band

            orth_loss = pairwise_orthogonal_loss(eeg_att[:, 0, 0, :], eeg_unatt[:, 0, 0, :]).mean(dim=-1)
            if self.band != 1:
                for b in range(1, self.band):
                    orth_loss = orth_loss + pairwise_orthogonal_loss(eeg_att[:, self.band_index[b], 0, :], eeg_unatt[:, self.band_index[b], 0, :]).mean(dim=-1)
            orth_loss = orth_loss / self.band
            # optim_loss.backward(retain_graph=True)
            # optimizer_MI.step()
            # self.MI.eval()
            # plotPCA(eeg_att[0, 0, 0, :].unsqueeze(0), stimulus1[0, :], eeg_unatt[0, 0, 0, :].unsqueeze(0), stimulus2[0, :])
            optim_loss_test = self.MI(eeg_att[:, 0, 0, :], eeg_unatt[:, 0, 0, :]).mean(dim=-1) #+ (eeg_att[:, 0, 0, :] - eeg_unatt[:, 0, 0, :]).mean(dim=-1)
            if self.band != 1:
                for b in range(1, self.band):
                    optim_loss_test = optim_loss_test + self.MI(eeg_att[:, self.band_index[b], 0, :], eeg_unatt[:, self.band_index[b], 0, :]).mean(dim=-1) #+ (eeg_att[:, self.band_index[b], 0, :] - eeg_unatt[:, self.band_index[b], 0, :]).mean(dim=-1)
            optim_loss_test = optim_loss_test / self.band

            # optim_loss = torch.tensor(0.0).to(eeg_att.device)
            # for j in range(eeg_att.shape[0]):
            #     optim_loss = optim_loss + 1 - eeg_att[j, 0, 0, :].unsqueeze(1) @ eeg_unatt[j, 0, 0, :].unsqueeze(1).T #+ (eeg_att[:, 0, 0, :] - eeg_unatt[:, 0, 0, :]).mean(dim=-1)
            # if self.band != 1:
            #     for b in range(1, self.band):
            #         for k in range(eeg_att.shape[0]):
            #             optim_loss = optim_loss + 1 - eeg_att[k, self.band_index[b], 0, :].unsqueeze(1) @ eeg_unatt[k, self.band_index[b], 0, :].unsqueeze(1).T #+ (eeg_att[:, self.band_index[b], 0, :] - eeg_unatt[:, self.band_index[b], 0, :]).mean(dim=-1)
            # optim_loss = optim_loss / self.band
            # pos_sim = (Pearson_corr(eeg, stimulus1)).mean(dim=1) * pos_mask.squeeze() + (
            #     Pearson_corr(eeg, stimulus2)).mean(dim=1) * neg_mask.squeeze()
            # neg_sim = (Pearson_corr(eeg, stimulus1)).mean(dim=1) * neg_mask.squeeze() + (
            #     Pearson_corr(eeg, stimulus2)).mean(dim=1) * pos_mask.squeeze()
            # pos_sim = ((F.cosine_similarity(eeg[:,0,:,:], stimulus1, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:,1,:,:], stimulus1, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:,2,:,:], stimulus1, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:,3,:,:], stimulus1, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:,4,:,:], stimulus1, dim=2)).mean(dim=1)) /5
            #pos_sim = (F.cosine_similarity(eeg[:,0,:,:], stimulus1, dim=-1)).mean(dim=1)
            pos_sim_att = (F.cosine_similarity(eeg_att[:,0,:,:], stimulus1, dim=-1)).mean(dim=1)
            pos_sim_unatt = (F.cosine_similarity(eeg_unatt[:,0,:,:], stimulus2, dim=-1)).mean(dim=1)
            sim_att = get_contrastive(eeg_att[:, 0, :, :], stimulus_att=stimulus1, stimulus_unatt=stimulus2)
            sim_unatt = get_contrastive(eeg_unatt[:, 0, :, :], stimulus_att=stimulus2, stimulus_unatt=stimulus1)
            if self.band != 1:
                for b in range(1, self.band):
                    #pos_sim = pos_sim + (F.cosine_similarity(eeg[:,self.band_index[b],:,:], stimulus1, dim=-1)).mean(dim=1)
                    pos_sim_att = pos_sim_att + (F.cosine_similarity(eeg_att[:,self.band_index[b],:,:], stimulus1, dim=-1)).mean(dim=1)
                    pos_sim_unatt = pos_sim_unatt + (F.cosine_similarity(eeg_unatt[:,self.band_index[b],:,:], stimulus2, dim=-1)).mean(dim=1)
                    sim_att = sim_att + get_contrastive(eeg_att[:, self.band_index[b], :, :], stimulus_att=stimulus1, stimulus_unatt=stimulus2)
                    sim_unatt = sim_unatt + get_contrastive(eeg_unatt[:, self.band_index[b], :, :], stimulus_att=stimulus2, stimulus_unatt=stimulus1)
            #pos_sim = pos_sim / self.band
            pos_sim_att = pos_sim_att / self.band
            pos_sim_unatt = pos_sim_unatt / self.band
            # neg_sim = ((F.cosine_similarity(eeg[:,0,:,:], stimulus2, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 1, :, :], stimulus2, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 2, :, :], stimulus2, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 3, :, :], stimulus2, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 4, :, :], stimulus2, dim=2)).mean(dim=1))/5
            #neg_sim = (F.cosine_similarity(eeg[:,0,:,:], stimulus2, dim=-1)).mean(dim=1)
            neg_sim_att = (F.cosine_similarity(eeg_att[:,0,:,:], stimulus2, dim=-1)).mean(dim=1)
            neg_sim_unatt = (F.cosine_similarity(eeg_unatt[:,0,:,:], stimulus1, dim=-1)).mean(dim=1)
            if self.band != 1:
                for b in range(1, self.band):
                    #neg_sim = neg_sim + (F.cosine_similarity(eeg[:,self.band_index[b],:,:], stimulus2, dim=-1)).mean(dim=1)
                    neg_sim_att = neg_sim_att + (F.cosine_similarity(eeg_att[:,self.band_index[b],:,:], stimulus2, dim=-1)).mean(dim=1)
                    neg_sim_unatt = neg_sim_unatt + (F.cosine_similarity(eeg_unatt[:,self.band_index[b],:,:], stimulus1, dim=-1)).mean(dim=1)
            #neg_sim = neg_sim / self.band
            neg_sim_att = neg_sim_att / self.band
            neg_sim_unatt = neg_sim_unatt / self.band
            # pos_sim = pearson_pt(eeg, stimulus1, axis=-1).mean(dim=1)
            # neg_sim = pearson_pt(eeg, stimulus2, axis=-1).mean(dim=1)
            # 通过 temperature 来调整
            #pos_sim = (pos_sim / self.layer) / self.temperature
            pos_sim_att = (pos_sim_att / self.layer) / self.temperature
            pos_sim_unatt = (pos_sim_unatt / self.layer) / self.temperature
            #neg_sim = (neg_sim / self.layer) / self.temperature
            neg_sim_att = (neg_sim_att / self.layer) / self.temperature
            neg_sim_unatt = (neg_sim_unatt / self.layer) / self.temperature

            sim_att = sim_att / (self.band * self.layer * self.temperature)
            sim_unatt = sim_unatt / (self.band * self.layer * self.temperature)
            # print(sim_att.shape)
            # print(neg_sim_unatt.shape)
            # sim_att = torch.cat([sim_att, neg_sim_unatt.unsqueeze(dim=-1)], dim=-1)
            # sim_unatt = torch.cat([sim_unatt, pos_sim_unatt.unsqueeze(dim=-1)], dim=-1)
            #sim_att = torch.cat([sim_att, neg_sim_att.unsqueeze(-1)], dim=-1)
            #sim_unatt = torch.cat([sim_unatt, pos_sim_att.unsqueeze(-1)], dim=-1)
            #batch_sim = batch_sim / self.temperature
            # 使用 log-softmax 处理正负对的对比
            # pos = pos_sim
            # neg = neg_sim

            # pos = pos_sim_att #- neg_sim_unatt
            # #neg = pos_sim_unatt - neg_sim_att
            # neg = pos_sim_unatt

            pos = pos_sim_att #- neg_sim_unatt
            neg = neg_sim_att #- pos_sim_unatt

            # pos = pos_sim_att - neg_sim_att
            # neg = pos_sim_unatt - neg_sim_unatt

            # neg = neg_sim_unatt #- pos_sim_unatt
            #logit_1 = torch.stack([pos_sim_att, neg_sim_unatt], dim=1)
            #logit_2 = torch.stack([neg_sim_att, pos_sim_unatt], dim=1)
            #logit_2 = torch.stack([pos_sim_unatt, neg_sim_att], dim=1)
            #loss_att = F.cross_entropy(logit_1, torch.zeros(logit_1.size(0), dtype=torch.long).to(features.device)) + F.cross_entropy(logit_2, torch.zeros(logit_2.size(0), dtype=torch.long).to(features.device))
            # pos = pos_sim_att - neg_sim_att
            # neg = neg_sim_unatt - pos_sim_unatt
            # pos = pos_sim_att - neg_sim_unatt 
            # neg = neg_sim_att - pos_sim_unatt  
            logits = torch.stack([pos, neg], dim=1)  # [batch_size, 2]
            #logits_1 = torch.stack([pos_sim_att, neg_sim_att], dim=1)
            #print('loss_pos:', F.cross_entropy(sim_att, torch.arange(sim_att.size(0), dtype=torch.long).to(feature_eeg.device)))
            #print('loss_neg:', F.cross_entropy(sim_unatt, torch.arange(sim_unatt.size(0), dtype=torch.long).to(feature_eeg.device)))
            # logits_2 = torch.stack([pos_sim_unatt, neg_sim_att], dim=1)
            #loss_pn = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long).to(features.device))
            # loss_pn = self.criterion(nn.functional.softmax(logits_1, dim=-1), torch.zeros(logits.size(0), dtype=torch.long).to(features.device)) \
            #             + self.criterion(nn.functional.softmax(logits_2, dim=-1), torch.zeros(logits.size(0), dtype=torch.long).to(features.device))
            targets = torch.zeros_like(sim_att).long()
            #print(targets.shape)
            targets[:, 0] = 1
            #targets[:, :sim_att.size(0)] = 0.3
            #targets.fill_diagonal_(1)
            #sim_att = F.log_softmax(sim_att,  dim=1)
            #sim_unatt = F.log_softmax(sim_unatt,  dim=1)
            # loss = -(targets * log_prob).sum(dim=1).mean()
            #loss_pn = -(targets * sim_att).sum(dim=1).mean() - (targets * sim_unatt).sum(dim=1).mean() #+ F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long).to(feature_eeg.device))
            loss_pn = F.cross_entropy(sim_unatt, torch.zeros(sim_unatt.size(0), dtype=torch.long).to(feature_eeg.device)) \
                    + F.cross_entropy(sim_att, torch.zeros(sim_att.size(0), dtype=torch.long).to(feature_eeg.device)) #+ F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long).to(feature_eeg.device))
                    #F.cross_entropy(sim_att, torch.arange(sim_att.size(0), dtype=torch.long).to(feature_eeg.device))/2 \
                    #+ F.cross_entropy(sim_unatt, torch.arange(sim_unatt.size(0), dtype=torch.long).to(feature_eeg.device))/2 \
                        #+ F.cross_entropy(logits_1, torch.zeros(logits.size(0), dtype=torch.long).to(feature_eeg.device))
                        #+ self.criterion(nn.functional.softmax(logits_1, dim=-1), torch.zeros(logits.size(0), dtype=torch.long).to(feature_eeg.device))
                        #+ self.criterion(nn.functional.softmax(logits, dim=-1), torch.zeros(logits.size(0), dtype=torch.long).to(feature_eeg.device))
                        #+F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long).to(feature_eeg.device))
                        # self.criterion(nn.functional.softmax(logits_1, dim=-1), torch.zeros(logits.size(0), dtype=torch.long).to(features.device)) +\
                        # self.criterion(nn.functional.softmax(logits_2, dim=-1), torch.zeros(logits.size(0), dtype=torch.long).to(features.device))
            #loss_pn = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long).to(features.device))
            #loss_batch = F.cross_entropy(batch_sim, torch.arange(0, logits.size(0), dtype=torch.long).to(features.device))
            # print('loss_pn:', loss_pn)
            # print('optim_loss_test:', optim_loss_test.mean()/(optim_loss_test.mean()+loss_pn).detach())
            # print('optim_loss:', optim_loss.mean()/(optim_loss.mean()+loss_pn).detach())
            # index = torch.argmax(logits, dim=1)\
            # print('loss_pn:', loss_pn)
            # print('orth_loss:', orth_loss)
            # print('optim_loss_test', optim_loss_test)
            # print('optim_loss', optim_loss)
            return loss_pn + orth_loss.mean()  , optim_loss_test.mean(), logits, optim_loss.mean()
        else:
            # if len(eeg.shape) == len(stimulus1.shape):
            #     stim1_sim = F.cosine_similarity(eeg, stimulus1)
            #     stim2_sim = F.cosine_similarity(eeg, stimulus2)
            # else:
            #eeg = eeg.unsqueeze(2).repeat(1, 1, stimulus1.shape[1], 1)
            eeg_att = eeg_att.unsqueeze(2).repeat(1, 1, stimulus1.shape[1], 1)
            eeg_unatt = eeg_unatt.unsqueeze(2).repeat(1, 1, stimulus1.shape[1], 1)
            # stim1_sim = (F.cosine_similarity(eeg, stimulus1, dim=2)).mean(dim=1)
            # stim2_sim = (F.cosine_similarity(eeg, stimulus2, dim=2)).mean(dim=1)
            # stim1_sim = pearson_pt(eeg, stimulus1, axis=-1).mean(dim=1)
            # stim2_sim = pearson_pt(eeg, stimulus2, axis=-1).mean(dim=1)
            # stim1_sim = ((F.cosine_similarity(eeg[:, 0, :, :], stimulus1, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 1, :, :], stimulus1, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 2, :, :], stimulus1, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 3, :, :], stimulus1, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 4, :, :], stimulus1, dim=2)).mean(dim=1)) / 5
            #stim1_sim = (F.cosine_similarity(eeg[:,0,:,:], stimulus1, dim=-1)).mean(dim=1)
            stim1_sim_att = (F.cosine_similarity(eeg_att[:,0,:,:], stimulus1, dim=-1)).mean(dim=1)
            stim1_sim_unatt = (F.cosine_similarity(eeg_unatt[:,0,:,:], stimulus1, dim=-1)).mean(dim=1)
            if self.band != 1:
                for b in range(1, self.band):
                    #stim1_sim = stim1_sim + (F.cosine_similarity(eeg[:,self.band_index[b],:,:], stimulus1, dim=-1)).mean(dim=1)
                    stim1_sim_att = stim1_sim_att + (F.cosine_similarity(eeg_att[:,self.band_index[b],:,:], stimulus1, dim=-1)).mean(dim=1)
                    stim1_sim_unatt = stim1_sim_unatt + (F.cosine_similarity(eeg_unatt[:,self.band_index[b],:,:], stimulus1, dim=-1)).mean(dim=1)
            #stim1_sim = stim1_sim / self.band
            stim1_sim_att = stim1_sim_att / self.band
            stim1_sim_unatt = stim1_sim_unatt / self.band
            # stim2_sim = ((F.cosine_similarity(eeg[:, 0, :, :], stimulus2, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 1, :, :], stimulus2, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 2, :, :], stimulus2, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 3, :, :], stimulus2, dim=2)).mean(dim=1) +
            #            (F.cosine_similarity(eeg[:, 4, :, :], stimulus2, dim=2)).mean(dim=1)) / 5
            #stim2_sim = (F.cosine_similarity(eeg[:,0,:,:], stimulus2, dim=-1)).mean(dim=1)
            stim2_sim_att = (F.cosine_similarity(eeg_att[:,0,:,:], stimulus2, dim=-1)).mean(dim=1)
            stim2_sim_unatt = (F.cosine_similarity(eeg_unatt[:,0,:,:], stimulus2, dim=-1)).mean(dim=1)
            if self.band != 1:
                for b in range(1, self.band):
                    #stim2_sim = stim2_sim + (F.cosine_similarity(eeg[:,self.band_index[b],:,:], stimulus2, dim=-1)).mean(dim=1)
                    stim2_sim_att = stim2_sim_att + (F.cosine_similarity(eeg_att[:,self.band_index[b],:,:], stimulus2, dim=-1)).mean(dim=1)
                    stim2_sim_unatt = stim2_sim_unatt + (F.cosine_similarity(eeg_unatt[:,self.band_index[b],:,:], stimulus2, dim=-1)).mean(dim=1)
                    
            #stim2_sim = stim2_sim / self.band
            stim2_sim_att = stim2_sim_att / self.band
            stim2_sim_unatt = stim2_sim_unatt / self.band

            #stim1_sim = (stim1_sim / self.layer) / self.temperature
            stim1_sim_att = (stim1_sim_att / self.layer) / self.temperature
            stim1_sim_unatt = (stim1_sim_unatt / self.layer) / self.temperature
            #stim2_sim = (stim2_sim / self.layer) / self.temperature
            stim2_sim_att = (stim2_sim_att / self.layer) / self.temperature
            stim2_sim_unatt = (stim2_sim_unatt / self.layer) / self.temperature
            # stim1_final = stim1_sim
            # stim2_final = stim2_sim

            # stim1_final =  stim1_sim_att #- stim2_sim_unatt
            # #stim2_final = stim1_sim_unatt - stim2_sim_att
            # stim2_final = stim1_sim_unatt

            stim1_final = stim1_sim_att #- stim1_sim_unatt#/(stim1_sim_att + stim2_sim_unatt)
            stim2_final = stim2_sim_att #- stim2_sim_unatt#/(stim2_sim_att + stim1_sim_unatt)
            #stim2_final =  stim2_sim_att #- stim1_sim_unatt
            # stim1_final = stim1_sim_att - stim2_sim_att
            # stim2_final = stim2_sim_unatt - stim1_sim_unatt 
            # stim1_final = stim1_sim_att - stim2_sim_unatt
            # stim2_final = stim2_sim_att - stim1_sim_unatt 
            logits = torch.stack([stim1_final, stim2_final], dim=1)
            #labels = labels - 1
            loss = F.cross_entropy(logits, labels.long()-1)
            #loss_pn = self.criterion(nn.functional.softmax(logits, dim=-1), labels.long())
            # index = torch.argmax(logits, dim=1)

            #return loss_pn, logits
            return loss, logits

class RestructionSupConLoss(nn.Module):
    def __init__(self, temperature=0.07, feature_dim=1):
        super(RestructionSupConLoss, self).__init__()
        self.temperature = temperature
        self.dim = feature_dim

    def forward(self, restruction, att_stim, unatt_stim, labels, training=True):
        """
        Args:
            features: tensor of shape [batch_size, 3, feature_dim], where:
                      - features[:, 0, :] is the EEG embedding
                      - features[:, 1, :] is the stimulus1 embedding
                      - features[:, 2, :] is the stimulus2 embedding
            labels: tensor of shape [batch_size], with values 1 or 2, which indicates:
                    - 1: stimulus1 is the positive sample
                    - 2: stimulus2 is the positive sample
        Returns:
            A scalar loss value.
        """
        restruction = restruction.squeeze()
        att_stim = att_stim.squeeze()
        unatt_stim = unatt_stim.squeeze()
        if training:
            # 根据标签选择正样本和负样本
            # pos_sim = F.cosine_similarity(restruction, att_stim)
            # neg_sim = F.cosine_similarity(restruction, unatt_stim)
            pos_sim = Pearson_corr(restruction, att_stim)
            neg_sim = Pearson_corr(restruction, unatt_stim)
            # 通过 temperature 来调整
            pos_sim = pos_sim / self.temperature
            neg_sim = neg_sim / self.temperature

            # 使用 log-softmax 处理正负对的对比
            logits = torch.stack([pos_sim, neg_sim], dim=1).squeeze()  # [batch_size, 2]
            loss = F.cross_entropy(logits, torch.zeros(logits.size(0), dtype=torch.long).to(att_stim.device))

            # index = torch.argmax(logits, dim=1)
            return loss, logits
        else:

            # stim1_sim = F.cosine_similarity(restruction, att_stim)
            # stim2_sim = F.cosine_similarity(restruction, unatt_stim)
            stim1_sim = Pearson_corr(restruction, att_stim)
            stim2_sim = Pearson_corr(restruction, unatt_stim)

            stim1_sim = stim1_sim / self.temperature
            stim2_sim = stim2_sim / self.temperature

            logits = torch.stack([stim1_sim, stim2_sim], dim=1)
            labels = labels - 1
            loss = F.cross_entropy(logits, labels.long())
            # index = torch.argmax(logits, dim=1)

            return loss, logits


def Pearson_corr(x, y):
    # x = x.unsqueeze(dim=0)
    # # x = x.transpose(1, 0)
    # y = y.unsqueeze(dim=0)
    # mean_x = torch.mean(x, dim=-1, keepdims=True)
    # std_x = torch.std(x, dim=-1, keepdims=True)
    # mean_y = torch.mean(y, dim=-1, keepdims=True)
    # std_y = torch.std(y, dim=-1, keepdims=True)

    # covariance = torch.sum((x - mean_x) * (y - mean_y), dim = -1, keepdim=True)
    # correlation = covariance / (std_x * std_y)
    numerator = ((x - x.mean(dim=1, keepdim=True)) *
                 (y - y.mean(dim=1, keepdim=True))).sum(dim=1) / (x.shape[1] - 1)  # 33
    denominator = x.std(dim=1) * y.std(dim=1)  # 33
    r = numerator / (denominator + 1e-12)

    return r

def STFT(epochsData, sfreq, band_index):
    bandFreqs_sum = [
        {'name': 'Delta', 'fmin': 1, 'fmax': 3},
        {'name': 'Theta', 'fmin': 4, 'fmax': 7},
        {'name': 'Alpha', 'fmin': 8, 'fmax': 13},
        {'name': 'Beta', 'fmin': 14, 'fmax': 30},
        {'name': 'Gamma', 'fmin': 31, 'fmax': 50}
    ]
    bandFreqs = [bandFreqs_sum[i] for i in band_index]
    # 利用signal包进行STFT变换，f为频率，t为时间，Zxx为变换结果
    f, t, Zxx = signal.stft(epochsData, fs=sfreq)
    # 分频带保存STFT后的结果
    bandResult = []
    # 单独分析某一个频率范围
    for iter_freq in bandFreqs:
        # 定位有效频率的索引
        index = np.where((iter_freq['fmin'] < f) & (f < iter_freq['fmax']))
        # 生成新的参数矩阵，初始化为复数元素为0
        portion = np.zeros(Zxx.shape, dtype=np.complex_)
        # 将有效频率赋值给新的参数矩阵
        portion[:, :, index, :] = Zxx[:, :, index, :]
        # 进行逆STFT变换，保留目标频率范围的信息
        _, xrec = signal.istft(portion, fs=sfreq)
        # 保存滤波后的结果
        bandResult.append(xrec)
    return bandResult

class KULdataset(Dataset):
    def __init__(self, data, label, stimulus, args):
        eeg_layer, stimulus_layer = 1, args.feature_layer
        eeg_dim, feature_dim = 64, args.feature_dim
        final_data = np.empty((0, args.eeg_band, 128, eeg_dim))
        final_feature = np.empty((0, stimulus_layer*2, 128, feature_dim))
        # final_audio = np.empty((0, 16000, 2))
        final_label = np.empty((0, 1))
        for key in data.keys():
            eeg = np.squeeze(data[key][0])
            frequency = np.array(STFT(eeg.transpose((0, 2, 1)), sfreq=128, band_index=args.band_index)).transpose((1, 0, 3, 2))

            # x = np.array([_ for _ in range(128)])
            # plt.plot(x, eeg.transpose((0, 2, 1))[0][0], label='raw', color='blue')  # 第一条曲线
            # plt.plot(x, frequency[0][0][0], label='y = 1', color='red')  # 第二条曲线
            # plt.plot(x, frequency[0][1][0], label='y = 2', color='green')
            # plt.plot(x, frequency[0][2][0], label='y = 3', color='yellow')  # 第一条曲线
            # plt.plot(x, frequency[0][3][0], label='y = 4', color='pink')  # 第二条曲线
            # plt.plot(x, frequency[0][4][0], label='y = 5', color='black')
            # # 第三条曲线
            # # 添加标题和标签
            # plt.title('Multiple Lines Example')
            # plt.xlabel('x')
            # plt.ylabel('y')
            # # 显示图例
            # plt.legend()
            # # 显示图形
            # plt.show()

            att_stim_name = data[key][1]
            unatt_stim_name = data[key][2]
            # att_audio_name = data[key][3]
            # unatt_audio_name = data[key][4]
            att_stimulus = stimulus[att_stim_name][:frequency.shape[0]]
            # att_audio = stimulus[att_audio_name][:eeg.shape[0]]
            unatt_stimulus = stimulus[unatt_stim_name][:frequency.shape[0]]
            # unatt_audio = stimulus[unatt_audio_name][:eeg.shape[0]]
            att_stimulus = att_stimulus[:, args.feature_layer_index]
            unatt_stimulus = unatt_stimulus[:, args.feature_layer_index]
            stim_sigle = np.concatenate((att_stimulus, unatt_stimulus), axis=1)
            # audio_sigle = np.concatenate((att_audio, unatt_audio), axis=-1)
            final_data = np.concatenate((final_data, frequency), axis=0)
            final_feature =  np.concatenate((final_feature, stim_sigle), axis=0)
            # final_audio = np.concatenate((final_audio, audio_sigle), axis=0)
            final_label = np.concatenate((final_label, label[key + '_stimulus'][0].reshape(-1, 1)), axis=0)
        self.data = final_data.transpose((0, 1, 3, 2))
        self.stim = final_feature.transpose((0, 1, 3, 2))
        # self.audio = final_audio
        self.label = final_label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index], self.stim[index], self.label[index]


class EarlyStopping:
    def __init__(self, patience=5, delta=0, verbose=False, path='checkpoint.pt'):
        """
        Args:
            patience (int): 容忍验证集性能不提升的轮数 (default: 5)
            delta (float): 性能提升的阈值，只有当提升大于该值时，才认为有提升 (default: 0)
            verbose (bool): 是否打印提示信息 (default: False)
            path (str): 模型保存的路径 (default: 'checkpoint.pt')
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.path = path + 'checkpoint.pt'
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score >= self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''保存模型，当验证集损失减小时调用'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def save_checkpoint(encoder, decoder, optimizer, amp, scheduler, step, checkpoint_dir):
    checkpoint_state = {
        "encoder": encoder.state_dict(),
        "decoder": decoder.state_dict(),
        "optimizer": optimizer.state_dict(),
        "amp": amp.state_dict(),
        "scheduler": scheduler.state_dict(),
        "step": step}
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    checkpoint_path = checkpoint_dir / "model.ckpt-{}.pt".format(step)
    torch.save(checkpoint_state, checkpoint_path)
    print("Saved checkpoint: {}".format(checkpoint_path.stem))


# @hydra.main(config_path="/media/nchen/CYB/EEGViT/VQVAE/ZeroSpeech/config/train.yaml")
# def main(cfg: DictConfig):
#     encoder, decoder = model_load(**cfg)
#     return encoder, decoder

def load_config(file_path):
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def model_load(device):
    config_path = "D:/EEGViT/VQVAE/ZeroSpeech/config/model/default.yaml"
    cfg = load_config(config_path)
    from VQVAE.ZeroSpeech.model_new import Encoder, Decoder
    # tensorboard_path = Path(utils.to_absolute_path("tensorboard")) / cfg['checkpoint_dir']
    # checkpoint_dir = Path(utils.to_absolute_path(cfg['checkpoint_dir']))
    # writer = SummaryWriter(tensorboard_path)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    encoder = Encoder(**cfg['model']['encoder'])
    decoder = Decoder(**cfg['model']['decoder'])
    encoder.to(device)
    decoder.to(device)

    print("Resume checkpoint from: {}:".format(True))
    # resume_path = utils.to_absolute_path(cfg['checkpoint_dir'])
    # checkpoint = torch.load(resume_path, map_location=lambda storage, loc: storage)
    checkpoint = torch.load('D:/EEGViT/VQVAE/ZeroSpeech/checkpoints/2019english/model.ckpt-500000.pt')
    encoder.load_state_dict(checkpoint["encoder"], strict=False)
    decoder.load_state_dict(checkpoint["decoder"], strict=False)
    # optimizer.load_state_dict(checkpoint["optimizer"])
    # amp.load_state_dict(checkpoint["amp"])
    # scheduler.load_state_dict(checkpoint["scheduler"])
    # global_step = checkpoint["step"]

    return encoder, decoder


def get_att_stim(stims, args, eeg_outputs=None, targets=None):

    if targets is not None and eeg_outputs is not None:
        # 将 label 转换为一维张量
        label = targets.view(-1)  # 形状变为 (32,)
        index = label.detach().cpu().int()
        # 创建布尔掩码
        mask_0 = (label == 1)  # shape: (32,)
        mask_1 = (label == 2)  # shape: (32,)
        # 提取对应的数据
        data_0 = stims[mask_0][:, :args.feature_layer, :, :]  # 提取 (i, 1:1+dim, :)
        data_2 = stims[mask_0][:, args.feature_layer:, :, :]
        data_1 = stims[mask_1][:, args.feature_layer:, :, :]  # 提取 (i, 1+dim:1+2*dim, :)
        data_3 = stims[mask_1][:, :args.feature_layer, :, :]
        eeg_out = torch.tensor([]).to(eeg_outputs.device)
        for i in range(len(index)):
            eeg_trial = torch.cat([eeg_outputs[i, :, index[i]-1].unsqueeze(-1), eeg_outputs[i, :, 2-index[i]].unsqueeze(-1)], dim=1)
            eeg_out = torch.cat([eeg_out, eeg_trial.unsqueeze(0)], dim=0)
        # 合并提取的数据
        att_stim = torch.cat([data_0, data_1], dim=0)  # shape: (提取的数量, dim, 128)
        unatt_stim = torch.cat([data_2, data_3], dim=0)
        return att_stim, unatt_stim, eeg_out
    else:
        return stims[:, :args.feature_layer, :, :], stims[:, args.feature_layer:, :, :]

def fixed_data(eeg_features, eeg_outputs, att_stim, unatt_stim, args):
    # eeg_features.unsqueeze_(2)
    eeg_outputs.unsqueeze_(1)
    feature_eeg = torch.tensor([]).to(eeg_outputs.device)
    feature_stim = torch.tensor([]).to(eeg_outputs.device)
    #att_stim_final = att_stim[:, -1, :, :].unsqueeze(1)
    #unatt_stim_final = unatt_stim[:, -1, :, :].unsqueeze(1)
    # output = torch.concatenate([eeg_outputs, att_stim[:, -1, :, :].unsqueeze_(1), unatt_stim[:, -1, :, :].unsqueeze_(1)], dim=2)
    #output = torch.concatenate([eeg_outputs, att_stim_final, unatt_stim_final], dim=2)
    for i in range(args.feature_layer):
        # hidden = torch.concatenate([eeg_features[:, int(args.feature_layer_index[i]/2), :, :].unsqueeze_(1), att_stim[:, i, :, :].unsqueeze_(1), unatt_stim[:, i, :, :].unsqueeze_(1)], dim=2)
        # feature = torch.concatenate([feature, hidden], dim=1)
        hidden_eeg = eeg_features[:, int(args.feature_layer_index[i]/2), :, :].unsqueeze_(1)
        feature_eeg = torch.concatenate([feature_eeg, hidden_eeg], dim=1)
        hidden_stim = torch.concatenate([att_stim[:, i, :, :].unsqueeze_(1), unatt_stim[:, i, :, :].unsqueeze_(1)], dim=2)
        feature_stim = torch.concatenate([feature_stim, hidden_stim], dim=1)

    #feature = torch.concatenate([feature, output], dim=1)

    return feature_eeg, feature_stim

class ProbCrossEntropyLoss(nn.Module):
    def __init__(self, eps=1e-12):
        super().__init__()
        self.eps  = eps 
        self.nll  = nn.NLLLoss()
 
    def forward(self, probs, targets):
        log_probs = torch.log(probs  + self.eps)   # 防止log(0)
        return self.nll(log_probs,  targets)
    
def deal(eeg_outputs, targets):
    label = targets.view(-1)  # 形状变为 (32,)
    index = label.detach().cpu().int()
    eeg_out = torch.tensor([]).to(eeg_outputs.device)
    for i in range(len(index)):

        eeg_trial = torch.tensor([eeg_outputs[i, 1-index[i]], eeg_outputs[i, 2-index[i]]]).to(eeg_outputs.device)
        # print(eeg_trial.shape)
        eeg_out = torch.cat([eeg_out, eeg_trial.unsqueeze(0)], dim=0)
    
    return eeg_out

import random
def train(data, label, stimulus, model, Classify_att_high, Classify_unatt_high, Classify_att_low, Classify_unatt_low, MI, args, optimizer, scheduler=None, optimizer_MI=None, scheduler_MI=None, fold=5):
    '''
        model: model to train
        optimizer: optimizer to update weights
        scheduler: scheduling learning rate, used when finetuning pretrained models
    '''
    cuda_id = args.cuda_id
    result_path = args.result_path
    batch_size = args.batch_size
    n_epoch = args.n_epoch
    condition = args.condition
    fold = fold
    print(f'nSub:{nSub}')
    torch.cuda.empty_cache()
    # if condition == 'hrtf' or 'dry':
    #     sample = 2
    # elif condition == 'all':
    sample = 30
    print(sample)
    trial_index = list(data.keys())

    trial_label_name = list(map(lambda x: x + '_stimulus', trial_index))
    print(trial_label_name)
    trial_label = [label[name][0][0] for name in trial_label_name]
    trial_label1 = []
    trial_label2 = []
    for i, value in enumerate(trial_label):
        if value == 1:
            trial_label1.append(i)
        elif value == 2:
            trial_label2.append(i)


    # test_trial_index.extend(list(data.keys())[trial_label1[fold]])
    # test_trial_index.extend(list(data.keys())[trial_label2[fold]])
    # test_trial_index = [list(data.keys())[trial_label1[-fold]]] + [list(data.keys())[trial_label2[fold]]]
    # test_trial_index = [list(data.keys())[trial_label1[fold]]] + [list(data.keys())[trial_label2[(sample-1) - fold]]]
    # test_trial_index = [list(data.keys())[trial_label1[fold*6 : (fold+1)*6]]] + [list(data.keys())[trial_label2[fold*6 : (fold+1)*6]]]
    sigle = int(len(trial_label) / (args.fold * 2))
    # random.shuffle(trial_label1)
    # random.shuffle(trial_label2)
    # test_trial_index = trial_index[48:]
    test_trial_index = []
    test_length = min(len(trial_label1), len(trial_label2))
    length_max = max(len(trial_label1), len(trial_label2))
    fold_max = int(test_length/sigle)
    a = int(len(trial_label1)/2)
    b = int(len(trial_label2)/2)
    if fold < fold_max:
        for n in range(fold*sigle, (fold+1)*sigle):
            test_trial_index.extend([list(data.keys())[trial_label1[n]]])
            test_trial_index.extend([list(data.keys())[trial_label2[n]]])
    else:
        for n in range(fold*sigle, test_length):
            test_trial_index.extend([list(data.keys())[trial_label1[n]]])
            test_trial_index.extend([list(data.keys())[trial_label2[n]]])
        # if len(trial_label1) > len(trial_label2):
        #     for n in range(test_length, length_max):
        #         test_trial_index.extend([list(data.keys())[trial_label1[n]]])
        if len(trial_label1) > len(trial_label2):
            for n in range(test_length, length_max):
                test_trial_index.extend([list(data.keys())[trial_label1[n]]])
        else:
            for n in range(test_length, length_max):
                test_trial_index.extend([list(data.keys())[trial_label2[n]]])

    test_label_index = list(map(lambda x: x + '_stimulus', test_trial_index))
    train_trial_index = list(set(trial_index) - set(test_trial_index))
    train_label_index = list(map(lambda x: x + '_stimulus', train_trial_index))

    train_data = {key: data[key] for key in train_trial_index}
    test_data = {key: data[key] for key in test_trial_index}
    train_label = {key: label[key] for key in train_label_index}
    test_label = {key: label[key] for key in test_label_index}
    # train_speaker = {key: label[key] for key in train_speaker_index}
    # test_speaker = {key: label[key] for key in test_speaker_index}
    print(test_label)

    train = KULdataset(train_data, train_label, stimulus, args) #(sample, layer_sum, feature_dim = 64, time = 128)
    # val = KULdataset(data[4:6], label[4:6])
    test = KULdataset(test_data, test_label, stimulus, args)
    print('create dataloader...')

    criterion_CL = LabelBasedSupConLoss(eeg_band=args.eeg_band, band_index=args.band_index, feature_dim=args.feature_dim)
    # criterion_MES = nn.MSELoss()
    criterion_CE = nn.CrossEntropyLoss()
    criterion_MES = nn.L1Loss()
    criterion_Re_CL = RestructionSupConLoss(feature_dim=args.feature_dim)
    Criterion_MI = MI

    save_path = f'./model_path/S{nSub}_fold{fold}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Folder '{save_path}' created successfully.")
    else:
        print(f"Folder '{save_path}' already exists.")

    if not os.path.exists(result_path):
        file = open(result_path, 'w')
        file.close()

    early_stopping = EarlyStopping(patience=args.patience, verbose=True, path=save_path)

    train_loader = DataLoader(train, batch_size=batch_size, drop_last=True, shuffle=True)
    print(len(train_loader))
    test_loader = DataLoader(test, batch_size=batch_size, drop_last=True, shuffle=True)
    print(len(test_loader))

    if torch.cuda.is_available():
        gpu_id = cuda_id  # Change this to the desired GPU ID if you have multiple GPUs
        torch.cuda.set_device(gpu_id)
        device = torch.device(f"cuda:{gpu_id}")
    else:
        device = torch.device("cpu")
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)  # Wrap the model with DataParallel
    print("HI")

    model = model.to(device)
    Classify_att_high = Classify_att_high.to(device)
    Classify_unatt_high = Classify_unatt_high.to(device)
    Classify_att_low = Classify_att_low.to(device)
    Classify_unatt_low = Classify_unatt_low.to(device)
    # Encoder = Encoder.to(device)
    # Decoder = Decoder.to(device)
    criterion_CL = criterion_CL.to(device)
    criterion_CE = criterion_CE.to(device)
    criterion_MES = criterion_MES.to(device)
    criterion_Re_CL = criterion_Re_CL.to(device)
    criterion_ProCE = ProbCrossEntropyLoss()
    Criterion_MI = Criterion_MI.to(device)
    stop_epoch = 0

    # Initialize lists to store losses
    train_losses = []
    # val_losses = []
    test_losses = []
    accuracy_best = 0.0
    print('training...')
    # Train the model
    for epoch in range(n_epoch):
        model.train()
    #    model.eval()
        # for name, param in model.named_parameters():
        #     if "classify" in name:
        #         param.requires_grad = True
        Classify_att_high.train()
        Classify_unatt_high.train()
        Classify_att_low.train()
        Classify_unatt_low.train()
        Criterion_MI.train()
        # Criterion_MI.eval()
        # Encoder.train()
        # Decoder.train()
        epoch_train_loss = 0.0
        accuracy_train = 0.0

        for i, (inputs, stims, targets) in tqdm(enumerate(train_loader)):
            # Move the inputs and targets to the GPU (if available)
            inputs = inputs.unsqueeze(dim=1).to(device).float()
            stims = stims.to(device)
            targets = targets.to(device)
            #print(inputs.shape)
            # Compute the outputs and loss for the current batch
            optimizer.zero_grad()
            eeg_features, eeg_outputs = model(inputs) #eeg_feature:(bs=8, layer, 128)  outputs:(bs=8, 128)
            # loss, preds = criterion(outputs.squeeze(), targets.squeeze())
            att_stim, unatt_stim, eeg_outputs_1 = get_att_stim(stims, args, eeg_outputs, targets)  # shape: (提取的数量, layer, dim, 128)
            
            feature_eeg, feature_stim = fixed_data(eeg_features, eeg_outputs_1, att_stim, unatt_stim, args)
            loss_layer = 0.0
            pred_layer = torch.zeros([batch_size, 2]).to(feature_eeg.device)
            for l in range(feature_eeg.shape[1]):
                # optimizer_MI.zero_grad()
                feature_layer_eeg = feature_eeg[:, l, :, :]
                feature_layer_stim = feature_stim[:, l, :, :]
                #print('featurelayer:', feature_layer.shape)
                loss_CL, loss_optim, pred_CL, MI_loss = criterion_CL(feature_layer_eeg, feature_layer_stim, targets.squeeze(), Classify_att_high, Classify_unatt_high, Classify_att_low, Classify_unatt_low, Criterion_MI, l, Classify_att_high.training, optimizer_MI)
                # print('loss_CL:', loss_CL)
                # print('loss_optim:', loss_optim)
                # print('loss_optim_test:', loss_optim_test)
                #pred_CL = torch.abs(pred_CL)
                # loss_layer = loss_layer + loss_CL * (l/feature_fixed.shape[1]) + loss_batch
                loss_layer = loss_layer + loss_CL + loss_optim + MI_loss #+ loss_optim_test #/10000
                pred_layer = pred_layer + pred_CL #*(l/feature_eeg.shape[1])
                # MI_loss.backward(retain_graph=True)
                # optimizer_MI.step()

            #pred = nn.functional.softmax(pred_layer, dim=-1)
            #pred = nn.functional.softmax(pred_layer, dim=-1) #+ nn.functional.softmax(eeg_outputs.squeeze(1).mean(dim=1), dim=-1)
            pred_audio = deal(nn.functional.softmax(pred_layer, dim=-1), targets)
            # print(pred_audio.shape)
            pred_eeg = nn.functional.softmax(eeg_outputs.squeeze(1), dim=-1).mean(dim=1)
            #pred_eeg =  nn.functional.softmax(eeg_outputs_1.squeeze(1), dim=-1).mean(dim=1)
            pred = nn.functional.softmax(pred_audio + pred_eeg, dim=-1)
            #targets_right = torch.zeros([batch_size, 1]).to(targets.device)
            preds= torch.argmax(pred, dim=-1, keepdim=True)
            accuracy_train += (preds == targets-1).sum()
            # targ = torch.tensor(targets.squeeze_(), dtype=int).to(device)
            # targ = targets.squeeze_() - 1
            # loss_CE = criterion_CE(pred, targ)
            # loss = loss_layer + loss_CE
            loss_EEGCE = criterion_CE(eeg_outputs.squeeze(1).mean(dim=1), targets.squeeze().long()-1)
            loss_AudioCE = criterion_ProCE(pred_audio, targets.squeeze().long()-1)
            loss_final = criterion_ProCE(pred, targets.squeeze().long()-1)
            # print('loss_layer:',loss_layer)
            # print('loss_EEGCE:', loss_EEGCE)
            # print('loss_AudioCE:', loss_AudioCE)
            # print('loss_final:', loss_final)
            #loss = loss_layer + loss_EEGCE + loss_final + loss_AudioCE
            loss = loss_layer + loss_EEGCE #+ loss_AudioCE/10 #+ loss_final 

            # Compute the gradients and update the parameters
            loss.backward(retain_graph=True)
            #if (i + 1) % 50 == 0 or i + 1 == len(train_loader):
            optimizer.step()
            epoch_train_loss += loss.item()

        epoch_train_loss /= len(train_loader) * batch_size
        accuracy_train /= len(train_loader) * batch_size
        print(f"Epoch {epoch}, Train Loss:{epoch_train_loss}, Accuracy: {accuracy_train}")
        train_losses.append(epoch_train_loss)

        # Evaluate the model on the validation set
        model.eval()
        Classify_att_high.eval()
        Classify_unatt_high.eval()
        Classify_att_low.eval()
        Classify_unatt_low.eval()
        Criterion_MI.eval()
        # Encoder.eval()
        # Decoder.eval()
        with torch.no_grad():
            test_loss = 0.0
            accuracy_test = 0.0
            for i, (inputs, stims, targets) in tqdm(enumerate(test_loader)):
                # Move the inputs and targets to the GPU (if available)
                inputs = inputs.unsqueeze(dim=1).to(device).float()
                stims = stims.to(device)
                targets = targets.to(device)

                # Compute the outputs and loss for the current batch
                eeg_features, eeg_outputs = model(inputs)
                stim1, stim2= get_att_stim(stims, args)

                feature_eeg, feature_stim = fixed_data(eeg_features, eeg_outputs, stim1, stim2, args)
                loss_layer = 0.0
                pred_layer = torch.zeros([batch_size, 2]).to(feature_eeg.device)
                for l in range(feature_eeg.shape[1]):
                    feature_layer_eeg = feature_eeg[:, l, :, :]
                    feature_layer_stim =feature_stim[:, l, :, :]
                    loss_CL, pred_CL = criterion_CL(feature_layer_eeg, feature_layer_stim, targets.squeeze(), Classify_att_high, Classify_unatt_high, Classify_att_low, Classify_unatt_low, Criterion_MI, l, model.training)
                    #pred_CL = torch.abs(pred_CL)
                    loss_layer = loss_layer + loss_CL #* (l/feature_fixed.shape[1])
                    pred_layer = pred_layer + pred_CL #* (l/feature_eeg.shape[1])

                #pred = nn.functional.softmax(pred_layer, dim=-1)
                # pred = nn.functional.softmax(pred_layer, dim=-1) #+ nn.functional.softmax(eeg_outputs.squeeze(1).mean(dim=1), dim=-1)
                # preds = torch.argmax(pred, dim=-1, keepdim=True)
                # accuracy_test += (preds == targets-1).sum()

                # targ = torch.tensor(targets.squeeze_(), dtype=int).to(device) - 1
                # loss_CE = criterion_CE(pred, targ)
                # loss = loss_layer + loss_CE
                # test_loss += loss.item()
                pred_audio = nn.functional.softmax(pred_layer, dim=-1)
                # pred_eeg = nn.functional.softmax(eeg_outputs.squeeze(1).mean(dim=1), dim=-1)
                pred_eeg = nn.functional.softmax(eeg_outputs.squeeze(1), dim=-1).mean(dim=1)
                # pred = nn.functional.softmax(pred_audio + pred_eeg, dim=-1)
                pred = nn.functional.softmax(pred_audio + pred_eeg, dim=-1)
                preds = torch.argmax(pred, dim=-1, keepdim=True)
                accuracy_test += (preds == targets-1).sum()

                #targ = torch.tensor(targets.squeeze_(), dtype=int).to(device) - 1
                #loss_CE = criterion_CE(pred, targ)
                loss_EEGCE = criterion_CE(eeg_outputs.squeeze(1).mean(dim=1), targets.squeeze().long()-1)
                loss_AudioCE = criterion_ProCE(pred_audio, targets.squeeze().long()-1)
                loss_final = criterion_ProCE(pred, targets.squeeze().long()-1)
                loss = loss_layer + loss_EEGCE #+ loss_AudioCE
                test_loss += loss.item()


            test_loss /= len(test_loader) * batch_size
            accuracy_test /= len(test_loader) * batch_size
            test_losses.append(test_loss)

            print(f"Epoch {epoch}, test Loss: {test_loss}, accuracy: {accuracy_test}")
            if epoch > args.warm_up:

                if accuracy_test > accuracy_best:
                    accuracy_best = accuracy_test
                    epoch_best = epoch
                    loss_best = test_loss

                early_stopping(accuracy_test, model)

                if early_stopping.early_stop:
                    print("Early stopping triggered.")
                    stop_epoch = epoch
                    stop_loss = test_loss
                    stop_accuracy = accuracy_test
                    file = open(result_path, 'a')
                    file.write(f'fold: {fold} \n')
                    file.write(
                        f'nSub: {nSub}   best_epoch: {epoch_best}    best_Loss: {loss_best}     best_accuracy: {accuracy_best} \n')
                    file.write(
                        f'nSub: {nSub}   stop_epoch: {stop_epoch}    stop_Loss: {stop_loss}     stop_accuracy: {stop_accuracy} \n')
                    file.close()
                    break

            # model.load_state_dict(torch.load('checkpoint.pt'))

        if scheduler is not None:
            scheduler.step()
            scheduler_MI.step()

        # print(f"Epoch {epoch}, test Loss: {test_loss}, accuracy: {accuracy_test}")
        if epoch == n_epoch - 1:
            file = open(result_path, 'a')
            file.write(f'fold: {fold} \n')
            file.write(
                f'nSub: {nSub}   best_epoch: {epoch_best}    best_Loss: {loss_best}     best_accuracy: {accuracy_best} \n')
            file.write(
                f'nSub: {nSub}   final_epoch: {stop_epoch}    final_Loss: {test_loss}     final_accuracy: {accuracy_test} \n')
            file.close()

# class Projection(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.project = nn.Linear(64, 64)

#     def forward(self, x):
#         x = x.float()
#         x = x.permute(0, 1, 3, 2)
#         x = self.project(x)
#         x = x.permute(0, 1, 3, 2)
#         # x = x.float()
#         return x
    
class Projection(nn.Module):
    def __init__(self):
        super().__init__()

        self.classfy = torch.nn.Sequential(
                torch.nn.Dropout(p=0.1),
                nn.Linear(128, 256, bias=True),
                nn.ReLU(),
                nn.Linear(256, 128, bias=True),
                # nn.Linear(32, 2, bias=True),
            )
    def forward(self, x):
        x = x + self.classfy(x)
        # x = x.float()
        return x

import torch.nn.init  as init
# 初始化方法1：Xavier正态分布（适合Sigmoid/Tanh）
def init_xavier_normal(module):
    if isinstance(module, nn.Linear):
        init.xavier_normal_(module.weight)
        if module.bias is not None:
            init.zeros_(module.bias)

            # 初始化方法2：Kaiming均匀分布（适合ReLU）

def init_kaiming_uniform(module):
    if isinstance(module, nn.Linear):
        init.kaiming_uniform_(module.weight, nonlinearity='relu')
        if module.bias is not None:
            init.normal_(module.bias, mean=0, std=0.01)

class Projection_768_low(nn.Module):
    def __init__(self, in_size=768, out_size=128):
        super().__init__()

        self.classfy = torch.nn.Sequential(
                #torch.nn.Dropout(p=0.2),
                nn.Linear(in_size, out_size*2, bias=True),
                nn.ReLU(),
                torch.nn.Dropout(p=0.2),
                nn.Linear(out_size*2, out_size, bias=True),
                #nn.BatchNormd(out_size, False)
                # nn.Linear(32, 2, bias=True),
            )
        self.BN = nn.BatchNorm2d(out_size, False)
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.LN = nn.LayerNorm(out_size, eps=1e-8)
    def forward(self, x):
        # x = x + self.classfy(x)
        # a = self.classfy(x)#.unsqueeze(dim=-1)
        # print(a.shape)
        # x =self.BN((self.classfy(x)).permute(0, 2, 1).unsqueeze(dim=-1)).squeeze(dim=-1).permute(0, 2, 1) 
        x = self.LN(self.classfy(x))
        #x =self.BN((self.avgpooling(x) + self.classfy(x)).permute(0, 2, 1).unsqueeze(dim=-1)).squeeze(dim=-1).permute(0, 2, 1)
        # x = x.float()
        return x

class Projection_768_high(nn.Module):
    def __init__(self, in_size=768, out_size=128):
        super().__init__()

        self.classfy = torch.nn.Sequential(
                #torch.nn.Dropout(p=0.2),
                nn.Linear(in_size, in_size*2, bias=True),
                nn.ReLU(),
                nn.Linear(in_size*2, 512, bias=True),
                nn.ReLU(),
                torch.nn.Dropout(p=0.2),
                nn.Linear(512, out_size*2, bias=True),
                nn.ReLU(),
                nn.Linear(out_size*2, out_size, bias=True),
                #nn.BatchNormd(out_size, False)
                # nn.Linear(32, 2, bias=True),
            )
        self.BN = nn.BatchNorm2d(out_size, False)
        self.avgpooling = torch.nn.AvgPool1d(6)
        self.LN = nn.LayerNorm(out_size, eps=1e-8)
    def forward(self, x):
        # x = x + self.classfy(x)
        # a = self.classfy(x)#.unsqueeze(dim=-1)
        # print(a.shape)
        # x =self.BN((self.classfy(x)).permute(0, 2, 1).unsqueeze(dim=-1)).squeeze(dim=-1).permute(0, 2, 1) 
        x = self.LN(self.classfy(x))
        #x =self.BN((self.avgpooling(x) + self.classfy(x)).permute(0, 2, 1).unsqueeze(dim=-1)).squeeze(dim=-1).permute(0, 2, 1)
        # x = x.float()
        return x

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Contrastive learning on KUL')
    parser.add_argument('--PT_model', type=str, default='/media/c1/CYB/EEGViT/vit-base-patch16-224', help='name of features')
    parser.add_argument('--eeg_data', type=str, default='/media/c1/CYB/EEGViT/eeg_data/DTU_right/', help='name of features')
    parser.add_argument('--stimulus_data', type=str, default='/media/c1/CYB/EEGViT/stimulus_DTU', help='name of features')
    parser.add_argument('--cuda_id', type=int, default=0, help='cuda device id')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--n_epoch', type=int, default=200, help='number of epoch')
    parser.add_argument('--patience', type=int, default=10, help='patience of earlystop')
    parser.add_argument('--warm_up', type=int, default=1, help='patience of earlystop')
    parser.add_argument('--fold', type=int, default=4, help='number of k-fold training')
    parser.add_argument('--window_length', type=int, default=1, help='window length of EEG')
    parser.add_argument('--overlap', type=int, default=1, help='overlap of EEG')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate') #1e-3  5e-4
    parser.add_argument('--feature_name', type=str, default='wavlm_24layer_1s', help='name of features')
    parser.add_argument('--eeg_band', type=int, default=3, help='bands of eeg')
    parser.add_argument('--band_index', type=list, default=[0, 1, 4], help='index of eeg bands')
    parser.add_argument('--feature_dim', type=int, default=64, help='dim of features')
    parser.add_argument('--feature_layer', type=int, default=7, help='layers of features')
    parser.add_argument('--feature_layer_index', type=list, default=[6, 7, 17, 21, 22, 23, 24], help='layers of features')
    parser.add_argument('--condition', type=str, default='all', help='condition')
    parser.add_argument('--result_path', type=str, default='right_sum_optim_mlp_DTU_result_wavlm_frequency_24layer_1s_chosen.txt', help='result path')
    args = parser.parse_args()

    random.seed(2002)

    file = open(args.result_path, 'a')
    file.write(f'eeg_band: {args.band_index} \n')
    print(args.band_index)
    for i in range(1, 19):
        nSub = i
        for k in range(args.fold):
            # model = EEGViT_KUL_CL(args)
            model = EEGViT_KUL_pretrained_wav2vec_frequency_sum_768(args)
            Classify_att_high = Projection_768_high()
            Classify_att_high.apply(init_xavier_normal)
            Classify_att_low = Projection_768_low()
            Classify_att_low.apply(init_xavier_normal)
            Classify_unatt_high = Projection_768_high()
            Classify_unatt_high.apply(init_kaiming_uniform)
            Classify_unatt_low = Projection_768_low()
            Classify_unatt_low.apply(init_kaiming_uniform)
            #MI = CLUB(128, 128, 512)
            #MI = CLUBSample(128, 128, 256)
            MI = CLUBMean(128, 128, 256)
            # Encoder, Decoder = model_load()
            # Encoder, Decoder = model_load(device=f'cuda:{args.cuda_id}')
            # Encoder = Projection()
            # Decoder = Projection()
            # decode = Decoder(out_dim=args.feature_dim, vocab_size=10000, d_model=128, num_layers=1, num_heads=1, d_ff=512, dropout=0.1)
            #data, label, stimulus = data_read_CL_feature_DTU(nSub, args.feature_name, args.condition, args.eeg_data, args.stimulus_data)
            data, label, stimulus = data_read_CL_feature_path_DTU(nSub, args.feature_name, args.window_length, args.overlap, args.condition, args.eeg_data, args.stimulus_data)


            # optimizer = torch.optim.Adam(chain(model.parameters(), Classify_att.parameters(), Classify_unatt.parameters()),
            #                              lr=args.learning_rate)
            params = [{"params": model.parameters(), "lr": args.learning_rate},
                    {"params": chain(Classify_att_high.parameters(), Classify_unatt_high.parameters()), "lr": args.learning_rate , "weight_decay": args.learning_rate},
                    {"params": chain(Classify_att_low.parameters(), Classify_unatt_low.parameters()), "lr": args.learning_rate , "weight_decay": args.learning_rate},
                    {"params": MI.parameters(), "lr": args.learning_rate}
                    ]
            params_MI = [{"params": MI.parameters(), "lr": args.learning_rate * 10}]
            optimizer = torch.optim.AdamW(params, weight_decay=0.)
            optimizer_MI = torch.optim.AdamW(params_MI, weight_decay=0.)
        #     lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        # optimizer, mode="min", patience=2, factor=0.8)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
            scheduler_MI = torch.optim.lr_scheduler.StepLR(optimizer_MI, step_size=2, gamma=0.5)

            train(data, label, stimulus, model, Classify_att_high, Classify_unatt_high, Classify_att_low, Classify_unatt_low, MI, args, optimizer=optimizer, scheduler=scheduler, optimizer_MI = optimizer_MI, scheduler_MI=scheduler_MI, fold=k)