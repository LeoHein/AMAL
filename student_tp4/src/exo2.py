from utils import RNN, SampleMetroDataset
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import numpy.random as rd


PATH = "data/"

LENGTH = 20
MIN_LENGTH = 10
MAX_LENGTH = 40
LENGTHS = [i for i in range(MIN_LENGTH, MAX_LENGTH)]
Variable = True

CLASSES = 10
DIM_INPUT = 2
BATCH_SIZE = 32
LATENT = 15
epsilon = 0.01
epochs = 5

matrix_train, matrix_test = torch.load(open(PATH+"hzdataset.pch", "rb"))

if Variable:
    Ldata_train = []
    Ldata_test = []
    for l in LENGTHS:
        ds_train_l = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=l)
        ds_test_l = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=l, stations_max=ds_train_l.stations_max)
        data_train_l = DataLoader(ds_train_l, batch_size=BATCH_SIZE, shuffle=True)
        data_test_l = DataLoader(ds_test_l, batch_size=BATCH_SIZE, shuffle=False)
        Ldata_train.append(data_train_l)
        Ldata_test.append(data_test_l)
    data_train, data_test = [], []
    for k in range(len(Ldata_train)):
        for idx, (batch, targets) in enumerate(Ldata_train[k]):
            if list(batch.size())[1] != 0:
                data_train.append((batch, targets))
    for k in range(len(Ldata_test)):
        for idx, (batch, targets) in enumerate(Ldata_test[k]):
            if list(batch.size())[1] != 0:
                data_test.append((batch, targets))
    rd.shuffle(data_test)
    rd.shuffle(data_train)

else:
    ds_train = SampleMetroDataset(matrix_train[:, :, :CLASSES, :DIM_INPUT], length=LENGTH)
    ds_test = SampleMetroDataset(matrix_test[:, :, :CLASSES, :DIM_INPUT], length=LENGTH, stations_max=ds_train.stations_max)
    data_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    data_test = DataLoader(ds_test, batch_size=BATCH_SIZE, shuffle=False)


model = RNN(DIM_INPUT, LATENT, CLASSES)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=epsilon)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

train_writer = SummaryWriter(log_dir='runs/train')
test_writer = SummaryWriter(log_dir='runs/test')


tqdm.write("Training")
for epoch in range(epochs):
    print('epoch ' + str(epoch+1) + ' out of ' + str(epochs))

    # Testing
    running_loss = 0
    correct = 0
    for idx, (batch, targets) in enumerate(data_test):
        batch, targets = batch.to(device), targets.to(device)
        batch_s = list(batch[:, 0, 0].size())[0]
        h = torch.zeros(batch_s, LATENT)
        batch = torch.transpose(batch, 0, 1)
        _, outputs = model(batch, h)
        outputs = outputs[-1, :, :]
        loss = criterion(outputs, targets)
        running_loss += loss
        correct += (outputs.argmax(1) == targets).float().mean()
    error = 1 - correct / len(data_test)
    test_writer.add_scalar(f'ErrRNN/', error, epoch)
    running_loss /= len(data_test)
    test_writer.add_scalar(f'LossRNN/', running_loss, epoch)

    # Training
    running_loss = 0
    correct = 0
    for idx, (batch, targets) in enumerate(tqdm(data_train)):
        batch, targets = batch.to(device), targets.to(device)
        batch_s = list(batch[:, 0, 0].size())[0]
        h = torch.zeros(batch_s, LATENT)
        batch = torch.transpose(batch, 0, 1)
        optimizer.zero_grad()
        _, outputs = model(batch, h)
        outputs = outputs[-1, :, :]
        loss = criterion(outputs, targets)
        running_loss += loss
        loss.backward()
        optimizer.step()
        correct += (outputs.argmax(1) == targets).float().mean()
    error = 1 - correct / len(data_train)
    train_writer.add_scalar(f'ErrRNN/', error, epoch)
    running_loss /= len(data_train)
    train_writer.add_scalar(f'LossRNN/', running_loss, epoch)








