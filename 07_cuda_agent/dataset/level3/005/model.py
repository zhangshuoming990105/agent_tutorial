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
        
    def forward(self, x, mask=None):
        """Aggregates feature maps into a fixed size representation.  In the following
        notation, B = batch_size, N = num_features, K = num_clusters, D = feature_size.

        Args:
            x (th.Tensor): B x N x D

        Returns:
            (th.Tensor): B x DK
        """
        
        assignment = self._compute_assignment(x)
        vlad = self._compute_vlad(x, assignment)
        
        return vlad
    
    def _compute_assignment(self, x):
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)  # B x N x D -> BN x D

        assignment = th.matmul(x, self.clusters)  # (BN x D) x (D x (K+G)) -> BN x (K+G)
        assignment = self.batch_norm(assignment)

        assignment = F.softmax(assignment, dim=1)  # BN x (K+G) -> BN x (K+G)
        # remove ghost assigments
        assignment = assignment[:, :self.cluster_size]
        assignment = assignment.view(-1, max_sample, self.cluster_size)  # -> B x N x K
        
        return assignment
    
    def _compute_vlad(self, x, assignment):
        max_sample = x.size()[1]
        x = x.view(-1, self.feature_size)  # B x N x D -> BN x D
        
        a_sum = th.sum(assignment, dim=1, keepdim=True)  # B x N x K -> B x 1 x K
        a = a_sum * self.clusters2

        assignment = assignment.transpose(1, 2)  # B x N x K -> B x K x N

        x = x.view(-1, max_sample, self.feature_size)  # BN x D -> B x N x D
        vlad = th.matmul(assignment, x)  # (B x K x N) x (B x N x D) -> B x K x D
        vlad = vlad.transpose(1, 2)  # -> B x D x K
        vlad = vlad - a

        # L2 intra norm
        vlad = F.normalize(vlad)

        # flattening + L2 norm
        vlad = vlad.reshape(-1, self.cluster_size * self.feature_size)  # -> B x DK
        vlad = F.normalize(vlad)
        return vlad  # B x DK


batch_size = 2048
num_features = 100
num_clusters = 32
feature_size = 512
ghost_clusters = 16


def get_inputs():
    return [torch.rand(batch_size, num_features, feature_size)]


def get_init_inputs():
    return [num_clusters, feature_size, ghost_clusters]


def initialize_model():
    model = Model(*get_init_inputs())
    model.eval()
    return model
