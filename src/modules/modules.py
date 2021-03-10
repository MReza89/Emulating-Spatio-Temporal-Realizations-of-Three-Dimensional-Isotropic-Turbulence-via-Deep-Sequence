import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantization(nn.Module):
    def __init__(self, embedding_size, num_embedding, vq_commit, decay=0.99, eps=1e-5):
        super().__init__()
        self.embedding_size = embedding_size
        self.num_embedding = num_embedding
        self.decay = decay
        self.eps = eps
        embedding = torch.randn(self.embedding_size, self.num_embedding)
        self.register_buffer('embedding', embedding)
        self.register_buffer('cluster_size', torch.zeros(self.num_embedding))
        self.register_buffer('embedding_mean', embedding.clone())
        self.vq_commit = vq_commit

    def forward(self, input):
        input = input.transpose(1, -1).contiguous()
        flatten = input.view(-1, self.embedding_size)
        dist = (
                flatten.pow(2).sum(1, keepdim=True)
                - 2 * flatten @ self.embedding
                + self.embedding.pow(2).sum(0, keepdim=True)
        )
        _, embedding_ind = dist.min(1)
        embedding_onehot = F.one_hot(embedding_ind, self.num_embedding).type(flatten.dtype)
        embedding_ind = embedding_ind.view(*input.shape[:-1])
        quantize = F.embedding(embedding_ind, self.embedding.transpose(0, 1))
        if self.training:
            self.cluster_size.data.mul_(self.decay).add_(embedding_onehot.sum(0), alpha=1 - self.decay)
            embedding_sum = flatten.transpose(0, 1) @ embedding_onehot
            self.embedding_mean.data.mul_(self.decay).add_(embedding_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.num_embedding * self.eps) * n
            )
            embedding_normalized = self.embedding_mean / cluster_size.unsqueeze(0)
            self.embedding.data.copy_(embedding_normalized)
        diff = self.vq_commit * F.mse_loss(quantize.detach(), input)
        quantize = input + (quantize - input).detach()
        quantize = quantize.transpose(1, -1).contiguous()
        return quantize, diff, embedding_ind

    def embedding_code(self, embedding_ind):
        return F.embedding(embedding_ind, self.embedding.transpose(0, 1)).transpose(1, -1).contiguous()
