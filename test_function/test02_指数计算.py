import math
from torch import nn
import torch

a = 0.9984
b = 0.0393
c = 0.5064
constant = (math.exp(a - a) + math.exp(b - a) + math.exp(c - a))
d = math.exp(a - a)/constant
e = math.exp(b - a)/constant
f = math.exp(c - a)/constant
print(d, e, f)
print(d+e+f)

softmax = nn.Softmax(dim=0)
result = softmax(torch.Tensor([a, b, c]))
print(result)
