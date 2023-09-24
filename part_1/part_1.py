# %%
import torch
import os
import numpy as np
import preprocess as pp

# %%
dir = 'RES/NNLM/TEST'

# %%
import os
if not os.path.exists(dir):
    os.makedirs(dir)

# %%
from torch.utils.data import Dataset

class SentencesDataset(Dataset):
    def __init__(self, sentences: list, Emb):
        super().__init__()

        self.X = []
        self.Y = []

        for sentence in sentences:
            s = pp.get_sentence_index_pad(sentence, Emb)

            for i in range(5, len(s)):
                self.X.append(s[i - 5:i])
                self.Y.append(s[i])

        self.X = torch.stack(self.X)
        self.Y = torch.stack(self.Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

# %%
from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8)

# %%
def load_data(Emb, batch_size):
    train_dataset = SentencesDataset(pp.train_sents, Emb)
    validation_dataset = SentencesDataset(pp.validation_sents, Emb)
    test_dataset = SentencesDataset(pp.test_sents, Emb)

    train_dataloader = get_dataloader(train_dataset, batch_size, shuffle=True)
    validation_dataloader = get_dataloader(validation_dataset, batch_size, shuffle=True)
    test_dataloader = get_dataloader(test_dataset, batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader

# %%
import tqdm
import numpy as np

def run(model, dataloader, train, es, device, loss_fn, optimizer):
    if train:
        model.train()
    else:
        model.eval()

    epoch_loss = []

    pbar = tqdm.tqdm(dataloader)

    for X, Y in pbar:
        Y_pred = model(X)

        Y = Y.to(device)
        loss = loss_fn(Y_pred, Y)
        epoch_loss.append(loss.item())

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_description(f'{"T" if train else "V"} Loss: {loss.item():7.4f}, Avg Loss: {sum(epoch_loss) / len(epoch_loss):7.4f}, Best Loss: {es.best_loss:7.4f}, Counter: {es.counter}')

    return np.mean(epoch_loss)

# %%
def train_epoch(model, train_dataloader, validation_dataloader, es, device, loss_fn, optimizer):
    train_loss = run(model, train_dataloader, True, es, device, loss_fn, optimizer)
    with torch.no_grad():
        validation_loss = run(model, validation_dataloader, False, es, device, loss_fn, optimizer)
    return train_loss, validation_loss

# %%
from nnlm import NNLM
from EarlyStopping import EarlyStopping
import torch.nn as nn

def train(train_dataloader, validation_dataloader):

    model = NNLM(pp.Emb, pp.hidden_dim, pp.dropout, pp.device).to(pp.device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=pp.learning_rate)

    es = EarlyStopping()

    for epoch in range(pp.epochs):
        _, validation_loss = train_epoch(model, train_dataloader, validation_dataloader, es, pp.device, loss_fn, optimizer)

        # Save model
        torch.save(model.state_dict(), os.path.join(dir, f'nnlm_{epoch}.pth'))

        if es(validation_loss, epoch):
            os.rename(os.path.join(dir, f'nnlm_{es.best_model_pth}.pth'), os.path.join(dir, f'best_model.pth'))
            break       

# %%
train_dataloader, validation_dataloader, test_dataloader = load_data(pp.Emb, pp.batch_size)
train(train_dataloader, validation_dataloader)

# %%
from nnlm import NNLM

best_model = NNLM(pp.Emb, pp.hidden_dim, pp.dropout, pp.device).to(pp.device)
best_pth = os.path.join(dir, 'best_model.pth')

# %%
import tqdm
# test
def run_perplexity(dataloader, f):
    best_model.load_state_dict(torch.load(best_pth))
    best_model.eval()

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        # epoch_loss = []
        perplexity = []

        current_sentence = ''
        current_pred = []
        current_truth = []

        for X, Y in tqdm.tqdm(dataloader):
            Y_pred = best_model(X)

            for i in range(Y.shape[0]):
                if Y[i].item() == pp.Emb.key_to_index['eos']:
                    if len(current_pred) == 0:
                        continue

                    current_pred = torch.stack(current_pred).to(pp.device)
                    current_truth = torch.tensor(current_truth).to(pp.device)
                    loss = loss_fn(current_pred, current_truth)

                    if torch.exp(loss).item() < 10000:
                        perplexity.append(torch.exp(loss).item())
                        print(f'{current_sentence.strip()}: {perplexity[-1]}', file=f)

                    current_sentence = ''
                    current_pred = []
                    current_truth = []

                elif Y[i].item() == pp.Emb.key_to_index['pad'] or Y[i].item() == pp.Emb.key_to_index['sos']:
                    continue
                else:
                    current_sentence += pp.Emb.index_to_key[Y[i].item()] + ' '
                    current_pred.append(Y_pred[i])
                    current_truth.append(Y[i])

            # epoch_loss.append(loss.item())

        # print(f'Average Loss: {np.mean(epoch_loss)}', file=f)
        print(f'Average Perplexity: {np.mean(perplexity)}', file=f)

# %%
with open(os.path.join(dir, 'test.txt'), 'w') as f:
    run_perplexity(test_dataloader, f)

with open(os.path.join(dir, 'train.txt'), 'w') as f:
    run_perplexity(train_dataloader, f)

with open(os.path.join(dir, 'val.txt'), 'w') as f:
    run_perplexity(validation_dataloader, f)

# %%
# best_model.load_state_dict(torch.load(best_pth))
# with torch.no_grad():
#     best_model.eval()
#     query = ['money', 'is', 'the', 'root', 'of']
#     print(*query, sep=' ', end=' ')

#     X = []
#     for word in query:
#         X.append(get_vocab_index(word))

#     while query[-1] != 'eos':
#         Y_pred = best_model(X)

#         # multinomial sampling
#         Y_pred = torch.multinomial(torch.softmax(Y_pred, dim=1), num_samples=1)

#         # Y_pred = torch.argmax(Y_pred, dim=1)
#         query = query[1:] + [Emb.index_to_key[Y_pred[-1].item()]]
#         X = X[1:] + [Y_pred[-1].item()]
#         print(Emb.index_to_key[Y_pred[-1].item()], end=' ')


