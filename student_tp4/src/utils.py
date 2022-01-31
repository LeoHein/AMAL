import torch
import torch.nn as nn
from torch.utils.data import Dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RNN(nn.Module):
    def __init__(self, dim, latent, output):
        super(RNN, self).__init__()
        self.i2h = nn.Linear(dim, latent)
        self.h2h = nn.Linear(latent, latent)
        self.h2y = nn.Linear(latent, output)
        self.tanh = nn.Tanh()

        self.latent = latent
        self.output = output
        self.dim = dim

    def one_step(self, x, h):
        Wx = self.i2h(x)
        Wh = self.h2h(h)
        Sum = Wx + Wh
        h = self.tanh(Sum)
        return h

    def forward(self, x, h):
        xu = x.unbind(0)
        for idx, x_t in enumerate(xu):
            h_t = self.one_step(x_t, h)
            h_t = torch.unsqueeze(h_t, dim=0)
            y_t = self.decode(h_t)
            y_t = torch.unsqueeze(y_t, dim=0)
            if idx == 0:
                Hf = h_t
                Yf = y_t
            if idx > 0:
                Hf = torch.cat([Hf, h_t], dim=0)
                Yf = torch.cat([Yf, y_t], dim=0)
        return Hf, Yf

    def decode(self, h):
        y = self.h2y(h)
        y = torch.squeeze(y)
        return y


class SampleMetroDataset(Dataset):
    def __init__(self, data, length, stations_max=None):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length = data, length
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1, self.data.size(2), self.data.size(3)), 0)[0]

        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1), self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.classes*self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self, i):
        ## transformation de l'index 1d vers une indexation 3d
        ## renvoie une séquence de longueur length et l'id de la station.
        station = i // ((self.nb_timeslots-self.length) * self.nb_days)
        i = i % ((self.nb_timeslots-self.length) * self.nb_days)
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day,timeslot:(timeslot+self.length),station],station


class ForecastMetroDataset(Dataset):
    def __init__(self, data, length=20, stations_max=None, horizon=1):
        """
            * data : tenseur des données au format  Nb_days x Nb_slots x Nb_Stations x {In,Out}
            * length : longueur des séquences d'exemple
            * stations_max : normalisation à appliquer
        """
        self.data, self.length, self.horizon = data, length, horizon
        ## Si pas de normalisation passée en entrée, calcul du max du flux entrant/sortant
        self.stations_max = stations_max if stations_max is not None else torch.max(self.data.view(-1,self.data.size(2),self.data.size(3)),0)[0]
        ## Normalisation des données
        self.data = self.data / self.stations_max
        self.nb_days, self.nb_timeslots, self.classes = self.data.size(0), self.data.size(1)-self.horizon+1, self.data.size(2)

    def __len__(self):
        ## longueur en fonction de la longueur considérée des séquences
        return self.nb_days*(self.nb_timeslots - self.length)

    def __getitem__(self, i):
        ## Transformation de l'indexation 1d vers indexation 2d
        ## renvoie x[d,t:t+length-1,:,:], x[d,t+1:t+length,:,:]
        timeslot = i // self.nb_days
        day = i % self.nb_days
        return self.data[day, timeslot:(timeslot+self.length-1)], self.data[day, (timeslot+self.horizon+1):(timeslot+self.length+self.horizon)]