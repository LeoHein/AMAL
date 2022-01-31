from pathlib import Path
import os
import torch
from torchvision.utils import make_grid
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import datetime
import math
import matplotlib.pyplot as plt
from datamaestro import prepare_dataset
from tqdm import tqdm


# Writer
writer = SummaryWriter()


# Téléchargement des données
ds = prepare_dataset("com.lecun.mnist")
train_images, train_labels = ds.train.images.data(), ds.train.labels.data()
test_images, test_labels = ds.test.images.data(), ds.test.labels.data()

images = torch.tensor(train_images[0:5]).reshape(-1, 1, 28, 28)
images = make_grid(images)

writer.add_image(f'samples', images, 0)


# Sauvegarde
savepath = Path("model.pch")

#if savepath.is_file():
    #with savepath.open("rb") as fp:
        #state = torch.load(fp)

class state:
    def __init__(self, model, optim):
        self.model = model
        self.optim = optim
        self.epoch = 0
        self.iteration = 0


# Creation des DataSets
class Train(Dataset):
    def __init__(self):
        self.train_images = train_images
        self.train_labels = train_labels
        self.n_samples = self.train_images.shape[0]

    def __getitem__(self, index):
        return self.train_images[index], self.train_labels[index]

    def __len__(self):
        return self.n_samples


class Test(Dataset):
    def __init__(self):
        self.test_images = test_images
        self.test_labels = test_labels
        self.n_samples = self.test_images.shape[0]

    def __getitem__(self, index):
        return self.test_images[index], self.test_labels[index]

    def __len__(self):
        return self.n_samples


# Creation du DataLoader
train_dataset = Train()
train_dataloader = DataLoader(dataset=train_dataset, batch_size=10, shuffle=True)



# Création de l'autoencodeur
class Autoencoder(torch.nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(nn.Flatten(),
                                     nn.Linear(28 * 28, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 32),
                                     nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(32, 64),
                                     nn.ReLU(),
                                     nn.Linear(64, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 28*28),
                                     nn.Sigmoid())

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Definition des paramètres
learning_rate = 0.01
momentum = 0.5
epochs = 10


# Initialisation du network
network = Autoencoder()
distance = torch.nn.MSELoss()
optimizer = torch.optim.Adam(network.parameters(), lr=1e-5, weight_decay=1e-5)


# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
network = network.to(device)


# Train
i = 0
#for epoch in range(state.epoch, iterations):
for epoch in range(epochs):
    for (i, batch) in enumerate(train_dataloader):
        print(batch)
        #if i >= 1000:
            #break
        print(i+1)
        #print(state.iteration)
        #state.optim.zero_grad()
        # ===================forward=====================
        batch = batch.type(torch.FloatTensor)
        reconstructed = network(batch)
        reconstructed = reconstructed.reshape(-1, 1, 28, 28)
        loss = distance(reconstructed, batch)
        print(loss)
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i += 1
        #state.iteration += 1

    #with savepath.open("wb") as fp:
        #state.epoch = epoch + 1
        #torch.save(state, fp)


writer.add_image(f'samples', images, 0)

