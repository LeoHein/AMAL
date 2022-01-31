import torch
from tp1 import mse, linear
from tp1 import *

# Test du gradient de MSE
yhat = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
y = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
print(torch.autograd.gradcheck(mse, (yhat, y)))


x = torch.randn(10, 8, requires_grad=True, dtype=torch.float64)
w = torch.randn(8, 5, requires_grad=True, dtype=torch.float64)
b = torch.randn(10, 5, requires_grad=True, dtype=torch.float64)
print(torch.autograd.gradcheck(linear, (x, w, b)))

