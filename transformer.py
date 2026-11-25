from data_trf import load_imdb, load_xor, load_imdb_synth
import torch
from torch import nn
import torch.nn.functional as F
import random

# load all data
(x_train_imdb, y_train_imdb), (x_val_imdb, y_val_imdb), (i2w_imdb, w2i_imdb), numcls_imdb = load_imdb(final=False)
(x_train_synth, y_train_synth), (x_val_synth, y_val_synth), (i2w_synth, w2i_synth), numcls_synth = load_imdb_synth()
(x_train_xor, y_train_xor), (x_val_xor, y_val_xor), (i2w_xor, w2i_xor), numcls_xor = load_xor()

#print(w2i_imdb['.pad'])

# padding function
def padding(lists, pad_index):
    # maximum length of data
    max_len = max(len(seq) for seq in lists)

    padded_lists = []
    for seq in lists:
        while len(seq) < max_len:
            seq = seq + [pad_index]
        padded_lists.append(seq)

    batch = torch.tensor(padded_lists, dtype=torch.long)

    return batch

# creating batches
def create_batches(x, y, batch_size, pad_index):
    batches = []
    indices = list(range(len(x)))
    random.shuffle(indices)

    for i in range(0,len(indices),batch_size):
        batch_indices = indices[i:i+batch_size]
        x_batch = padding([x[j] for j in batch_indices], pad_index)
        y_batch = torch.tensor([y[j] for j in batch_indices], dtype=torch.long)
        batches.append((x_batch, y_batch))
    
    return batches
    
#print(len(i2w_imdb)) 

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

# training the model
def train(model, x_train, y_train, x_val, y_val, pad_index, epochs, learning_rate, batch_size):
    optimizer = torch.optim.Adam(model.parameters(), learning_rate)
    history = []

    for epoch in range(1, epochs + 1):
        # training
        model.train()
        total_loss = 0.0

        batches = create_batches(x_train, y_train, batch_size, pad_index)
        for x_batch, y_batch in batches:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x_batch), y_batch)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        train_loss = total_loss / len(batches)

        # validation
        model.eval()
        correct = 0

        with torch.no_grad():
            for x_batch, y_batch in create_batches(x_val, y_val, batch_size, pad_index):
                predictions = model(x_batch).argmax(dim=1)
                correct += (predictions == y_batch).sum().item()

        val_accuracy = correct / len(x_val)
        history.append(val_accuracy)
        print("epoch: ", epoch, ", train loss: ", train_loss, ", validation accuracy: ", val_accuracy)

    return history

# run the experiments

# parameters
pool = 'select'
epochs = 5
learning_rate = 1e-3
batch_size = 64

# imdb data
model_imdb = NeuralNetwork(len(i2w_imdb), numcls_imdb, embedding_size=300, pool=pool)
history_imdb = train(model_imdb, x_train_imdb, y_train_imdb, x_val_imdb, y_val_imdb, w2i_imdb['.pad'], epochs, learning_rate, batch_size)
print("history_imdb: ", history_imdb)

# imdb synth data
model_synth = NeuralNetwork(len(i2w_synth), numcls_synth, embedding_size=300, pool=pool)
history_synth = train(model_synth, x_train_synth, y_train_synth, x_val_synth, y_val_synth, w2i_synth['.pad'], epochs, learning_rate, batch_size)
print("history imdb synth: ", history_synth)

# xor data
model_xor = NeuralNetwork(len(i2w_xor), numcls_xor, embedding_size=300, pool=pool)
history_xor = train(model_xor, x_train_xor, y_train_xor, x_val_xor, y_val_xor, w2i_xor['.pad'], epochs, learning_rate, batch_size)
print("history xor: ", history_xor)