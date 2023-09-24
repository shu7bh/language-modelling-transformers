# %%
import numpy as np
import preprocess as pp

# %%
dir = 'RES/LSTM/TEST/200'

import os
if not os.path.exists(dir):
    os.makedirs(dir)

# %% [markdown]
# ## Dataset
# 
# Creating a custom dataset which has one variable: data
# 
# data contains a list of sentences where each sentence is a list of words' indexes.

# %%
import torch
from torch.utils.data import Dataset

class SentencesDataset(Dataset):
    def __init__(self, sentences: list, Emb):
        super().__init__()

        self.data = []
        for sentence in sentences:
            self.data.append(pp.get_sentence_index(sentence, Emb))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# %%
training_data = SentencesDataset(pp.train_sents, pp.Emb)
validation_data = SentencesDataset(pp.validation_sents, pp.Emb)
test_data = SentencesDataset(pp.test_sents, pp.Emb)

# %%
training_data[0].shape

# %% [markdown]
# A custom collate function which pads the sentences to the same length with the max length of the batch.
# 
# This helps in parallelizing calling the LSTM by stacking the sentences of the same length together.

# %%
def padding_collate(X, Emb):
    # get max length in X
    max_len = max(map(lambda x: len(x), X))
    # set the pred tensor to be of the same size as the X
    Y = []
    for i in range(len(X)):
        # get the device of the tensor
        X[i] = torch.cat((X[i], torch.empty(max_len - len(X[i]), dtype=torch.long).fill_(Emb.key_to_index['pad'])))
        Y.append(X[i][1:])
        X[i] = X[i][:-1]
    return torch.stack(X), torch.stack(Y)

# %% [markdown]
# Creating dataloaders for the dataset

# %%
from torch.utils.data import DataLoader

def wrapper_collate(batch):
    return padding_collate(batch, pp.Emb)

training_dataloader = DataLoader(training_data, batch_size=pp.batch_size, shuffle=True, collate_fn=wrapper_collate)
validation_dataloader = DataLoader(validation_data, batch_size=pp.batch_size, shuffle=True, collate_fn=wrapper_collate)
test_dataloader = DataLoader(test_data, batch_size=pp.batch_size, shuffle=False, collate_fn=wrapper_collate)

# %% [markdown]
# Initializing the model, optimizer, and the loss_fn

# %%
from lstm import LSTM

lstm = LSTM(pp.Emb, pp.hidden_dim, pp.dropout, pp.device).to(pp.device)
optimizer = torch.optim.Adam(lstm.parameters(), lr=pp.learning_rate)
loss_fn = nn.CrossEntropyLoss()

# %%
len(pp.Emb)

# %% [markdown]
# One run of the dataloader is defined here

# %%
import tqdm

def run(lstm, dataloader, train, es):
    if train:
        lstm.train()
    else:
        lstm.eval()

    epoch_loss = []

    pbar = tqdm.tqdm(dataloader)

    for X, Y in pbar:
        lstm.init_hidden()
        Y_pred = []

        for i in range(X.shape[1]):
            Y_pred.append(lstm(X[:, i]))

        Y_pred = torch.stack(Y_pred, dim=1)
        Y_pred = Y_pred.view(-1, Y_pred.shape[2])
        Y = Y.view(-1).to(pp.device)

        loss = loss_fn(Y_pred, Y)
        epoch_loss.append(loss.item())

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_description(f'{"T" if train else "V"} Loss: {loss.item():7.4f}, Avg Loss: {np.mean(epoch_loss):7.4f}, Best Loss: {es.best_loss:7.4f}, Counter: {es.counter}')

    return np.mean(epoch_loss)

# %% [markdown]
# Training the Model here
# The best weights are saved as best_model.pth

# %%
import EarlyStopping as ES

es = ES.EarlyStopping()

for epoch in range(pp.epochs):
    print(f'Epoch {epoch+1}' + '\n')

    epoch_loss = run(lstm, training_dataloader, True, es)

    with torch.no_grad():
        epoch_loss = run(lstm, validation_dataloader, False, es)
        if es(epoch_loss, epoch):
            break

    torch.save(lstm.state_dict(), os.path.join(dir, f'lstm_{epoch + 1}.pth'))

os.rename(os.path.join(dir, f'lstm_{es.best_model_pth + 1}.pth'), os.path.join(dir, 'best_model.pth'))

# %%
best_model = LSTM(pp.Emb, pp.hidden_dim, pp.dropout, pp.device).to(pp.device)
best_pth = os.path.join(dir, 'best_model.pth')

# %% [markdown]
# Getting the perplexity scores for each sentence and outputting to the file

# %%
import sys
# test
def run_perplexity(dataloader, f):
    # f = sys.stdout
    best_model.load_state_dict(torch.load(best_pth))
    best_model.eval()

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        perplexity = []

        for X, Y in tqdm.tqdm(dataloader):
            best_model.init_hidden()
            Y_pred = []

            for i in range(X.shape[1]):
                Y_pred.append(best_model(X[:, i]))

            Y_pred = torch.stack(Y_pred, dim=1).to(pp.device)

            for i in range(Y_pred.shape[0]):
                sentence = ''
                for j in range(Y.shape[1]):
                    if Y[i][j] == pp.Emb.key_to_index['eos']:
                        Y_pred_ = Y_pred[i][:j]
                        Y_ = Y[i][:j].to(pp.device)
                        loss = loss_fn(Y_pred_, Y_)
                        perplexity.append(torch.exp(loss).item())
                        sentence = sentence.strip()
                        print(f'{sentence}: {perplexity[-1]}', file=f)
                        break
                    else:
                        sentence += pp.Emb.index_to_key[Y[i][j].item()] + ' '

        print(f'Average Perplexity: {np.mean(perplexity)}', file=f)

# %%
with open(os.path.join(dir, 'train.txt'), 'w') as f:
    run_perplexity(train_dataloader, f)

with open(os.path.join(dir, 'validation.txt'), 'w') as f:
    run_perplexity(val_dataloader, f)

with open(os.path.join(dir, 'test.txt'), 'w') as f:
    run_perplexity(test_dataloader, f)

# %% [markdown]
# General trial of how the model works. Problem in this is that I am taking the multinomial distribution for the next word and not the argmax. This is because the argmax will always give the same word and the model will not be able to generate new sentences.

# %%
# predict a sentence

best_model.load_state_dict(torch.load(best_pth))
best_model.eval()

current_word = 'this'
best_model.init_hidden()

while current_word != 'eos':
    X = pp.get_vocab_index(current_word, pp.Emb)

    Y_pred = best_model(X)
    # multinomial distribution on y_pred to get the next word
    Y_pred = torch.multinomial(torch.softmax(Y_pred, dim=0), 1).item()
    # Y_pred = torch.argmax(Y_pred, dim=0).item()

    current_word = pp.Emb.index_to_key[Y_pred]
    print(current_word, end=' ')


