from data_trf import load_imdb
import torch
from torch import nn
import torch.nn.functional as F

(x_train, y_train), (x_val, y_val), (i2w, w2i), numcls = load_imdb(final=False)

print(w2i['.pad']) # pad token is index 0

# padding function
def padding(lists):
    # maximum length of data
    max_len = max(len(seq) for seq in lists)

    padded_lists = []
    for seq in lists:
        while len(seq) < max_len:
            seq = seq + [0]
        padded_lists.append(seq)

    batch = torch.tensor(padded_lists, dtype=torch.long)

    return batch

print(len(i2w)) #there are 99430 tokens

class NeuralNetwork(nn.Module):
    def __init__(self, vocabulary_size, num_cls, embedding_size=300, pool='mean'):
        super().__init__()
        self.pool = pool
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.out = nn.Linear(embedding_size, num_cls)

    def forward(self, x):
        x = self.embedding(x)

        if self.pool == 'mean':
            x = x.mean(dim=1)
        elif self.pool =='select':
            x = x[:,0,:]
        elif self.pool == 'max':
            x = x.max(dim=1)[0]

        logits = self.out(x)

        return logits

model = NeuralNetwork(99430, numcls, embedding_size=300, pool='mean')
