import torch
import tensorboard
import datetime
from torch.utils.tensorboard import SummaryWriter
## Installer datamaestro et datamaestro-ml pip install datamaestro datamaestro-ml
import datamaestro
from tqdm import tqdm
#from tp1_descente import Batch, SGD, MiniBatch
from torch import nn
import datamaestro
from datetime import datetime
from tp1 import MSE, Linear
from tqdm import tqdm
from sklearn.model_selection import train_test_split



class Network(torch.nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = torch.nn.Linear(n, 1)
        self.tanh = torch.nn.Tanh()
        self.linear2 = torch.nn.Linear(1, 1)
        self.mse = torch.nn.MSELoss()

    def forward(self, x):
        z1 = self.linear1(x)
        a1 = self.tanh(z1)
        y = self.linear2(a1)
        return y



writer = SummaryWriter('TP2_AMAL')

data = datamaestro.prepare_dataset("edu.uci.boston")
colnames, datax, datay = data.data()
datax = torch.tensor(datax, dtype=torch.float64)
datay = torch.tensor(datay, dtype=torch.float64).reshape(-1, 1)

#Batch(datay, datax, 1e-6, 1500)
#SGD(datay, datax, 1e-6, 1500)
#MiniBatch(datay, datax, 1e-6, 1500, 20)


x_train, x_test, y_train, y_test = train_test_split(datax, datay, test_size=0.1)

q_training = x_train.shape[0]
n = x_train.shape[1]
p = y_train.shape[1]


writer = SummaryWriter()

epsilon = 1e-4
N = 10000 # Nbr of iterations

seed = 1
torch.manual_seed(seed)

model = Network()
mse = torch.nn.MSELoss()
optim = torch.optim.SGD(model.parameters(), lr=epsilon)

for n_iter in tqdm(range(N)):
    # Calcul du forward
    y = model(x_train.float())
    loss = mse(y, y_train.float())
    writer.add_scalar("Loss_2layers/Train", loss, n_iter)
    optim.zero_grad()
    loss.backward(retain_graph=True)
    optim.step()
