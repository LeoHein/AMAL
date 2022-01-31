# Ne pas oublier d'executer dans le shell avant de lancer python :
# source /users/Enseignants/baskiotis/venv/amal/3.7/bin/activate

import numpy as np
import torch
from torch.autograd import Function
from torch.autograd import gradcheck



class Context:
    def __init__(self):
        self._saved_tensors = ()
    def save_for_backward(self, *args):
        self._saved_tensors = args
    @property
    def saved_tensors(self):
        return self._saved_tensors


class MSE(Function):

    @staticmethod
    def forward(ctx, yhat, y):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(yhat, y)
        q = y.size()[0]
        MSE = 1/q * torch.linalg.norm(y-yhat)**2
        return MSE

    @staticmethod
    def backward(ctx, grad_output):
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        yhat, y = ctx.saved_tensors
        q = y.size()[0]
        diff = yhat - y
        gyhat = 2/q * diff
        gy = -2/q * diff
        return grad_output*gyhat, grad_output*gy


class Linear(Function):

    @staticmethod
    def forward(ctx, x, w, b):
        ## Garde les valeurs nécessaires pour le backwards
        ctx.save_for_backward(x, w, b)
        lin = x @ w + b
        return lin

    @staticmethod
    def backward(ctx, grad_output):
        #print(grad_output)
        ## Calcul du gradient du module par rapport a chaque groupe d'entrées
        x, w, b = ctx.saved_tensors
        gx = grad_output @ w.t()
        gw = x.t() @ grad_output
        gb = grad_output
        return gx, gw, gb

## Utile dans ce TP que pour le script tp1_gradcheck
mse = MSE.apply
linear = Linear.apply

