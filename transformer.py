from data_trf import load_imdb, load_xor, load_imdb_synth
import torch
from torch import nn
import torch.nn.functional as F
import random
import optuna

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

class SelfAttention(nn.Module):
    def __init__(self, vocabulary_size, num_cls, embedding_size=300, pool='max'):
        super().__init__()
        self.pool = pool
        self.embedding = nn.Embedding(vocabulary_size, embedding_size)
        self.out = nn.Linear(embedding_size, num_cls)

    def forward(self, x):
        # maximum length
        x = x[:, :256]
        x = self.embedding(x)                      

        # dot product
        weights = torch.bmm(x, x.transpose(1, 2))   
        # print(weights.shape) 
        # softmax
        weights = F.softmax(weights, dim=2)         
        # weihgted sum
        x = torch.bmm(weights, x)                   

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
        #print("epoch: ", epoch, ", train loss: ", train_loss, ", validation accuracy: ", val_accuracy)

    return history

# run the experiments

# parameters
pool = 'select'
epochs = 30
learning_rate = 1e-3
batch_size = 64

# # imdb question 1-3
# model_imdb = NeuralNetwork(len(i2w_imdb), numcls_imdb, embedding_size=300, pool=pool)
# history_imdb = train(model_imdb, x_train_imdb, y_train_imdb, x_val_imdb, y_val_imdb, w2i_imdb['.pad'], epochs, learning_rate, batch_size)
# print("history_imdb: ", history_imdb)

# # imdb synth question 1-3
# model_synth = NeuralNetwork(len(i2w_synth), numcls_synth, embedding_size=300, pool=pool)
# history_synth = train(model_synth, x_train_synth, y_train_synth, x_val_synth, y_val_synth, w2i_synth['.pad'], epochs, learning_rate, batch_size)
# print("history imdb synth: ", history_synth)

# # xor question 1-3
# model_xor = NeuralNetwork(len(i2w_xor), numcls_xor, embedding_size=300, pool=pool)
# history_xor = train(model_xor, x_train_xor, y_train_xor, x_val_xor, y_val_xor, w2i_xor['.pad'], epochs, learning_rate, batch_size)
# print("history xor: ", history_xor)

# # imdb question 5 --> imdb data with max pooling, don't spend a lot time tuning it
# model_imdb = SelfAttention(len(i2w_imdb), numcls_imdb, embedding_size=300, pool='max')
# history_imdb = train(model_imdb, x_train_imdb, y_train_imdb, x_val_imdb, y_val_imdb, w2i_imdb['.pad'], epochs, learning_rate, batch_size)
# print("history imdb: ", history_imdb)

# imd synth question 5 --> synth data with select pooling, using optuna to optimize hyperparameters
def objectiveSynth(trial):
    # hyperparameters to tune
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

    model = SelfAttention(len(i2w_synth), numcls_synth, embedding_size=300, pool='select')

    history = train(model, x_train_synth, y_train_synth, x_val_synth, y_val_synth, w2i_synth['.pad'], epochs=30, learning_rate=lr, batch_size=batch_size)
    print("trial: ", trial.number, ", lr: ", lr, ", batch size: ", batch_size, ", best val acc: ", max(history))
    return max(history)

# study = optuna.create_study(direction='maximize')
# study.optimize(objectiveSynth, n_trials=20, timeout=3600)

# print("best val acc:", study.best_value)
# print("best params:", study.best_params)

# question 5 XOR --> select pooling, using optuna to optimize hyperparameters
def objectiveXor(trial):
    lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [8, 16, 32, 64])

    model = SelfAttention(len(i2w_xor), numcls_xor, embedding_size=300, pool='select')

    history = train(model, x_train_xor, y_train_xor, x_val_xor, y_val_xor, w2i_xor['.pad'], epochs=30, learning_rate=lr, batch_size=batch_size)
    print("trial: ", trial.number, ", lr: ", lr, ", batch size: ", batch_size, ", score: ", max(history))
    return max(history)

# study = optuna.create_study(direction='maximize')
# study.optimize(objectiveXor, n_trials=20, timeout=3600)

# print("best val acc:", study.best_value)
# print("best params:", study.best_params)

# question 5
# best hyperparameter configurations from the optuna search
# (learning rate, batch size)
# configs = [
#     (0.008614, 16),
#     (0.007430, 8),
#     (0.009969, 16),
#     (0.003224, 16),
# ]
# epochs_final = 100

# for learning_rate, batch_size in configs:
#     print("lr: ", learning_rate, ", batch size: ", batch_size)
#     model = SelfAttention(len(i2w_synth), numcls_synth, embedding_size=300, pool='select')

#     train(model, x_train_synth, y_train_synth, x_val_synth, y_val_synth, w2i_synth['.pad'], epochs_final, learning_rate, batch_size)

# question 5, XOR
# (learning rate, batch size)
configs_xor = [
    (0.007079, 8), 
    (0.002973, 64),   
    (0.001702, 16),    
    (0.005446, 32),   
    (0.006270, 32),   
    (0.001639, 8),   
    (0.007029, 64), 
    (0.002053, 32),    
    (0.003471, 64),  
    (0.006184, 16),   
]
epochs_final = 100

for learning_rate, batch_size in configs_xor:
    print("lr: ", learning_rate, ", batch size: ", batch_size)
    model = SelfAttention(len(i2w_xor), numcls_xor, embedding_size=300, pool='select')

    history = train(model, x_train_xor, y_train_xor, x_val_xor, y_val_xor, w2i_xor['.pad'], epochs_final, learning_rate, batch_size)

    best_acc = max(history)
    best_epoch = history.index(best_acc) + 1   # first epoch with the highest accuracy

    print("lr: ", learning_rate, ", batch size: ", batch_size, ", best epoch: ", best_epoch, ", best val acc: ", best_acc, ", final epoch val acc: ", history[-1])
    
