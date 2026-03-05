import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch as th


class Model(nn.Module):
    def __init__(self, cluster_size, feature_size, ghost_clusters):
        super(Model, self).__init__()

        self.feature_size = feature_size
        self.cluster_size = cluster_size
        self.ghost_clusters = ghost_clusters

        init_sc = (1 / math.sqrt(feature_size))
        clusters = cluster_size + ghost_clusters

        # The `clusters` weights are the `(w,b)` in the paper
        self.clusters = nn.Parameter(init_sc * th.randn(feature_size, clusters))
        self.batch_norm = nn.BatchNorm1d(clusters)
        # The `clusters2` weights are the visual words `c_k` in the paper
        self.clusters2 = nn.Parameter(init_sc * th.randn(1, feature_size, cluster_size))
        self.out_dim = self.cluster_size * feature_size
        
        # Initialize with fixed values for reproducibility
        torch.manual_seed(42)
        self.clusters.data = init_sc * th.randn(feature_size, clusters)
        self.clusters2.data = init_sc * th.randn(1, feature_size, cluster_size)
        self.batch_norm.running_mean.zero_()
        self.batch_norm.running_var.fill_(1.0)
        self.batch_norm.weight.data.fill_(1.0)
        self.batch_norm.bias.data.fill_(0.0)

    def forward(self, x, mask=None):
        """Fixed forward pass for reproducibility"""
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)

        assignment = th.matmul(x, self.clusters)
        
        # Use batch norm in evaluation mode
        assignment = (assignment - self.batch_norm.running_mean) * \
                   torch.rsqrt(self.batch_norm.running_var + self.batch_norm.eps) * \
                   self.batch_norm.weight + self.batch_norm.bias

        assignment = assignment.softmax(dim=1)
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)

        a_sum = th.sum(assignment, dim=1, keepdim=True)
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)
        x = x.view(-1, max_sample, self.feature_size)

        vlad = th.matmul(assignment, x)
        vlad = vlad.transpose(1, 2)
        vlad = vlad - a

        # L2 intra norm
        vlad = vlad / vlad.norm(dim=-1, keepdim=True)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)
        vlad = vlad / vlad.norm(dim=-1, keepdim=True)

        return vlad

batch_size = 2048
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 0

def get_inputs():
  # Fixed input
  torch.manual_seed(42)
  return [torch.rand(batch_size, num_features, feature_size)]

def get_init_inputs():
  return [num_clusters, feature_size, ghost_clusters]
