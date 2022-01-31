import torch
from torch.utils.tensorboard import SummaryWriter
from tp1 import *


# Les données supervisées
x = torch.randn(50, 13, requires_grad=False, dtype=torch.float64)
y = torch.randn(50, 3, requires_grad=False, dtype=torch.float64)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3, requires_grad=True, dtype=torch.float64)
b = torch.randn(50, 3, requires_grad=True, dtype=torch.float64)

epsilon = 0.05
niter = 2000

writer = SummaryWriter()

for n_iter in range(niter):
    # Calcul de la loss
    yhat = linear(x, w, b)
    loss = mse(yhat, y)

    # Affichages
    writer.add_scalar('Loss/train', loss, n_iter)
    print(f"Itérations {n_iter}: loss {loss}")

    # Back propagation
    cont = Context()
    cont.save_for_backward(yhat, y)
    gyhat, gy = MSE.backward(cont, 1)
    cont.save_for_backward(x, w, b)
    gx, gw, gb = Linear.backward(cont, gyhat)

    w = w - epsilon * gw
    b = b - epsilon * gb
    print(w)
    print(b)


"""

# Les données supervisées
x = torch.randn(50, 13, requires_grad=False, dtype=torch.float64)
y = torch.randn(50, 3, requires_grad=False, dtype=torch.float64)


def param_init(x, y, nsamples):
    sx = list(x.size())
    sy = list(y.size())
    if nsamples > 0:
        w = torch.randn(sx[1], sy[1], requires_grad=True, dtype=torch.float64)
        b = torch.randn(nsamples, sy[1], requires_grad=True, dtype=torch.float64)
    else:
        w = torch.randn(sx[1], sy[1], requires_grad=True, dtype=torch.float64)
        b = torch.randn(sy[0], sy[1], requires_grad=True, dtype=torch.float64)
    return w, b


def Batch(y, x, epsilon, niter):
    writer = SummaryWriter()
    w, b = param_init(x, y, 0)
    for n_iter in range(niter):
        yhat = linear(x, w, b)
        loss = mse(yhat, y)
        writer.add_scalar('Loss/train', loss, n_iter)
        loss.backward()
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad
        w.grad.data.zero_()
        b.grad.data.zero_()


def SGD(y, x, epsilon, niter):
    writer = SummaryWriter()
    w, b = param_init(x, y, 1)
    for n_iter in range(niter):
        # creation du mini-batch
        indices = torch.randperm(len(x))[:1]
        xmb = x[indices]
        ymb = y[indices]
        yhat = linear(xmb, w, b)
        loss = mse(yhat, ymb)
        writer.add_scalar('Loss/train', loss, n_iter)
        loss.backward()
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad
        w.grad.data.zero_()
        b.grad.data.zero_()


def MiniBatch(y, x, epsilon, niter, nsamples):
    writer = SummaryWriter()
    w, b = param_init(x, y, nsamples)
    for n_iter in range(niter):
        # creation du mini-batch
        indices = torch.randperm(len(x))[:nsamples]
        xmb = x[indices]
        ymb = y[indices]
        yhat = linear(xmb, w, b)
        loss = mse(yhat, ymb)
        writer.add_scalar('Loss/train', loss, n_iter)
        loss.backward()
        with torch.no_grad():
            w -= epsilon * w.grad
            b -= epsilon * b.grad
        w.grad.data.zero_()
        b.grad.data.zero_()
"""
"""
# Les données supervisées
x = torch.randn(50, 13, requires_grad=False, dtype=torch.float64)
y = torch.randn(50, 3, requires_grad=False, dtype=torch.float64)

# Les paramètres du modèle à optimiser
w = torch.randn(13, 3, requires_grad=True, dtype=torch.float64)
b = torch.randn(50, 3, requires_grad=True, dtype=torch.float64)

epsilon = 1e-3
nb_epoch = 1000000

optim = torch.optim.SGD(params=[w, b], lr=epsilon) ## on optimise selon w et b, lr : pas de gradient
optim.zero_grad()

for i in range(nb_epoch):
    loss = mse(linear(x, w, b), y)
    loss.backward() # Retropropagation
    if i % 100 == 0:
        print(loss)
        optim.step() # Mise-à-jour des paramètres w et b
        optim.zero_grad() # Reinitialisation du gradient
        """