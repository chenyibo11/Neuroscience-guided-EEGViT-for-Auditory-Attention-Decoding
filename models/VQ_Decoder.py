import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from tqdm import tqdm
import numpy as np

class VQEmbedding(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25):
        """
        n_embeddings: 码本大小，即码字总数
        embedding_dim: 每个码字的维度
        commitment_cost: 在总体损失函数中，commitment loss前的系数，即commitment cost
        """
        super(VQEmbedding, self).__init__()
        self.commitment_cost = commitment_cost

        ### 初始化码本 ###
        init_bound = 1 / n_embeddings
        self.embedding = torch.Tensor(n_embeddings, embedding_dim).to('cuda:1')
        # self.embedding = nn.Embedding(n_embeddings, embedding_dim)
        # 从均匀分布U[-1/512,1/512]中抽样数值对tensor进行填充
        self.embedding.uniform_(-init_bound, init_bound)
        # self.embedding.weight.data.uniform_(-init_bound, init_bound)
        # self.register_buffer("embedding", self.embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def forward(self, x):
        """
        输入：
        x: [B, T, D]，为连续向量序列
        ----------------------------------
        输出：
        quantized: 离散化后的序列，即\hat{z}
        loss: VQ Loss
        """
        K, D = self.embedding.size()  # K表示码字总数/码本大小，D表示码字维度
        # K, D = self.n_embeddings, self.embedding_dim
        x_flat = x.detach().reshape(-1, D).float()  # x:[B,T,D]->x_flat:[BxT,D]

        # torch.addmm(M,M1,M2,a,b) = bM+a(M1@M2), 其中M1@M2表示矩阵乘法
        # 计算序列x和码本中各码字之间的距离
        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        # # 选择距离最近的码字，获得的indices为相应码字的索引序列
        # indices = torch.argmin(distances.float(), dim=-1)

        # ### 获得相应的码字 ###
        # quantized = self.embedding(indices)  # quantized为检索到的相应的码字
        # quantized = quantized.view_as(x)  # [BxT,D]->[B,T,D]

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, K).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)


        # if not self.training:
        #     return quantized

        ### 计算VQ Loss ###
        # VQ损失，固定x，使quantized向x更靠近
        q_latent_loss = F.mse_loss(quantized, x.detach())
        # VQ损失，固定quantized，使x向quantized更靠近，即commitment loss
        e_latent_loss = F.mse_loss(x, quantized.detach())
        # 整体VQ损失
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        ### 使用stop-gradient operator, 便于反向传播计算梯度###
        # .detach()即使用了stop-gradient operator，在反向传播的时候只计算对x的梯度
        quantized = x + (quantized - x).detach()

        return quantized, loss


class VQEmbeddingEMA(nn.Module):
    def __init__(self, n_embeddings, embedding_dim, commitment_cost=0.25, decay=0.999, epsilon=1e-5):
        super(VQEmbeddingEMA, self).__init__()
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        init_bound = 1 / n_embeddings
        embedding = torch.Tensor(n_embeddings, embedding_dim)
        embedding.uniform_(-init_bound, init_bound)
        self.register_buffer("embedding", embedding)
        self.register_buffer("ema_count", torch.zeros(n_embeddings))
        self.register_buffer("ema_weight", self.embedding.clone())

    def encode(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D).float()

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)
        return quantized, indices

    def forward(self, x):
        M, D = self.embedding.size()
        x_flat = x.detach().reshape(-1, D).float()

        distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
                                torch.sum(x_flat ** 2, dim=1, keepdim=True),
                                x_flat, self.embedding.t(),
                                alpha=-2.0, beta=1.0)

        indices = torch.argmin(distances.float(), dim=-1)
        encodings = F.one_hot(indices, M).float()
        quantized = F.embedding(indices, self.embedding)
        quantized = quantized.view_as(x)

        if self.training:
            self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

            n = torch.sum(self.ema_count)
            self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

            dw = torch.matmul(encodings.t(), x_flat)
            self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

            self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

        e_latent_loss = F.mse_loss(x, quantized.detach())
        loss = self.commitment_cost * e_latent_loss

        quantized = x + (quantized - x).detach()

        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

        return quantized, loss, perplexity


class BiLSTMNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(BiLSTMNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=False,
                            batch_first=True,
                            dropout=self.dropout)

        self.fc1 = nn.Linear(self.hidden_size, self.hidden_size*2)
        self.fc2 = nn.Linear(self.hidden_size*2, self.output_size)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        output, (h_n, c_n) = self.lstm(x)
        # out = torch.concat([h_n[-1, :, :], h_n[-2, :, :]], dim=-1)
        out = F.relu(output)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        out = out.permute(1, 0, 2)
        return out

class LinearNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, output_size):
        super(LinearNet, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.output_size = output_size

        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size*2),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size*2, self.hidden_size),
            nn.Dropout(self.dropout),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        out = self.model(x)
        return out
