import torch
from torch import nn
# (N, k, d)
region_feats = torch.rand(5, 3, 2)
convKK = nn.Conv1d(3, 3 * 3, 2, groups=3)
convd = convKK(region_feats)
# torch.Size([5, 9, 1])
# print(convd)
convd = convd.view(5, 3, 3)
# print(convd)
# print(convd.shape)
# dim=-1也就是从最里面的一个维度计算
activation = nn.Softmax(dim=-1)
multiplier = activation(convd)
# torch.Size([5, 3, 3])
print(convd)
print(convd.squeeze(1))
