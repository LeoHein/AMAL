import string
import unicodedata
import torch
import sys
import tqdm
import torch.nn.functional as F
from tqdm import tqdm, trange
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader

from utils import RNN, device

## Liste des symboles autorisés
LETTRES = string.ascii_letters + string.punctuation+string.digits+' '
## Dictionnaire index -> lettre
id2lettre = dict(zip(range(1,len(LETTRES)+1),LETTRES))
id2lettre[0]='' ##NULL CHARACTER
## Dictionnaire lettre -> index
lettre2id = dict(zip(id2lettre.values(),id2lettre.keys()))

def normalize(s):
    """ Nettoyage d'une chaîne de caractères. """
    return ''.join(c for c in unicodedata.normalize('NFD', s) if c in LETTRES)

def string2code(s):
    """ Transformation d'une chaîne de caractère en tenseur d'indexes """
    return torch.tensor([lettre2id[c] for c in normalize(s)])

def code2string(t):
    """ Transformation d'une liste d'indexes en chaîne de caractères """
    if type(t) !=list:
        t = t.tolist()
    return ''.join(id2lettre[i] for i in t)

class TrumpDataset(Dataset):
    def __init__(self,text,maxsent=None,maxlen=None):
        """  Dataset pour les tweets de Trump
            * text : texte brut
            * maxsent : nombre maximum de phrases.
            * maxlen : longueur maximale des phrases.
        """
        maxlen = maxlen or sys.maxsize
        full_text = normalize(text)
        self.phrases = [p[:maxlen].strip()+"." for p in full_text.split(".") if len(p)>0]
        if maxsent is not None:
            self.phrases=self.phrases[:maxsent]
        self.MAX_LEN = max([len(p) for p in self.phrases])

    def __len__(self):
        return len(self.phrases)

    def __getitem__(self,i):
        t = string2code(self.phrases[i])
        t = torch.cat([torch.zeros(self.MAX_LEN-t.size(0), dtype=torch.float),t])
        return t[:-1], t[1:]


LENGTH = 20
CLASSES = 10
DIM_INPUT = 1
BATCH_SIZE = 8
LATENT = 15
OUTPUTS = 96
epsilon = 1e-3
epochs = 10
horizon = 1

PATH = "data/"
speech = str(open(PATH+"trump_full_speech.txt", "rb").read())
dataset = TrumpDataset(speech)
data_train = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
data_test = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

model = RNN(DIM_INPUT, LATENT, OUTPUTS)
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
    for idx, (batch, targets) in enumerate(tqdm(data_test)):
        batch, targets = batch.to(device), targets.to(device)
        print(code2string(batch[0, :]))
        batch_s = list(batch[:, 0].size())[0]
        h = torch.zeros(batch_s, LATENT)
        batch = torch.transpose(batch, 0, 1)
        batch = torch.unsqueeze(batch, dim=2)
        targets = torch.transpose(targets, 0, 1)
        _, outputs = model(batch, h)
        outputs = torch.transpose(outputs, 1, 2)
        targets = targets.long()
        loss = criterion(outputs, targets)
        running_loss += loss
    running_loss /= len(data_test)
    test_writer.add_scalar(f'LossRNN/', running_loss, epoch)

    # Training
    running_loss = 0
    for idx, (batch, targets) in enumerate(tqdm(data_train)):
        batch, targets = batch.to(device), targets.to(device)
        batch_s = list(batch[:, 0].size())[0]
        h = torch.zeros(batch_s, LATENT)
        batch = torch.transpose(batch, 0, 1)
        batch = torch.unsqueeze(batch, dim=2)
        targets = torch.transpose(targets, 0, 1)
        _, outputs = model(batch, h)
        outputs = torch.transpose(outputs, 1, 2)
        targets = targets.long()
        loss = criterion(outputs, targets)
        running_loss += loss
        loss.backward()
        optimizer.step()
    running_loss /= len(data_train)
    train_writer.add_scalar(f'LossRNN/', running_loss, epoch)


SEQ_LENGTH = 10
init_seq_b = "I am"

for i in range(SEQ_LENGTH):
    init_seq = torch.cat((torch.zeros(803-len(init_seq_b)), string2code(init_seq_b)), dim=-1)
    init_seq = torch.unsqueeze(init_seq, dim=1)
    init_seq = torch.unsqueeze(init_seq, dim=2)
    init_seq = init_seq.float()
    h = torch.zeros(1, LATENT)
    _, outputs = model(init_seq, h)
    outputs = torch.argmax(outputs, dim=1)
    print(outputs)
    outputs = code2string(outputs)
    outputs = init_seq_b + outputs[-1]
    init_seq_b = outputs
    print(init_seq_b)


