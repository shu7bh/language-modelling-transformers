# %%
import torch
import os
import numpy as np
import preprocess as pp
from config import *

# %%
train_len = config['parameters']['train_len']['value']
validation_len = config['parameters']['validation_len']['value']
test_len = config['parameters']['test_len']['value']

# %%
train_sents, validation_sents, test_sents = pp.get_sents('Auguste_Maquet.txt', train_len, validation_len, test_len)

# %%
import os

dir = 'RES/NNLM/TEST'

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
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

# %%
def load_data(Emb, batch_size):
    train_dataset = SentencesDataset(train_sents, Emb)
    validation_dataset = SentencesDataset(validation_sents, Emb)
    test_dataset = SentencesDataset(test_sents, Emb)

    train_dataloader = get_dataloader(train_dataset, batch_size, shuffle=True)
    validation_dataloader = get_dataloader(validation_dataset, batch_size, shuffle=True)
    test_dataloader = get_dataloader(test_dataset, batch_size, shuffle=False)

    return train_dataloader, validation_dataloader, test_dataloader

# %%
import tqdm
import numpy as np

def run(model, dataloader, train, es, device, loss_fn, optimizer, epoch):
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

        pbar.set_description(f'{epoch} {"T" if train else "V"} Loss: {loss.item():7.4f}, Avg Loss: {sum(epoch_loss) / len(epoch_loss):7.4f}, Best Loss: {es.best_loss:7.4f}, Counter: {es.counter}')

    return np.mean(epoch_loss)

# %%
def train_epoch(model, train_dataloader, validation_dataloader, es, device, loss_fn, optimizer, epoch):
    train_loss = run(model, train_dataloader, True, es, device, loss_fn, optimizer, epoch)
    with torch.no_grad():
        validation_loss = run(model, validation_dataloader, False, es, device, loss_fn, optimizer, epoch)
    return train_loss, validation_loss

# %%
from nnlm import NNLM
from EarlyStopping import EarlyStopping
import torch.nn as nn

def train(train_dataloader, validation_dataloader, cfg, Emb):

    model = NNLM(Emb, cfg.hidden_dim, cfg.dropout, pp.device).to(pp.device)
    print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, cfg.optimizer)(model.parameters(), lr=cfg.learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    es = EarlyStopping(patience=3)

    for epoch in range(cfg.epochs):
        _, validation_loss = train_epoch(model, train_dataloader, validation_dataloader, es, pp.device, loss_fn, optimizer, epoch)
        # Save model
        torch.save(model.state_dict(), os.path.join(dir, f'nnlm_{epoch}.pth'))

        if es(validation_loss, epoch):
            break

    os.rename(os.path.join(dir, f'nnlm_{es.best_model_pth}.pth'), os.path.join(dir, f'best_model.pth'))

    return es.best_loss

# %%
from nnlm import NNLM
import tqdm

def run_perplexity(dataloader, best_model, best_pth, Emb):
    best_model.load_state_dict(torch.load(best_pth))
    best_model.eval()

    loss_fn = nn.CrossEntropyLoss()

    with torch.no_grad():
        perplexity = []

        current_sentence = ''
        current_pred = []
        current_truth = []

        for X, Y in tqdm.tqdm(dataloader):
            Y_pred = best_model(X)

            for i in range(Y.shape[0]):
                if Y[i].item() == Emb.key_to_index['eos']:
                    if len(current_pred) == 0:
                        continue

                    current_pred = torch.stack(current_pred).to(pp.device)
                    current_truth = torch.tensor(current_truth).to(pp.device)
                    loss = loss_fn(current_pred, current_truth)

                    if torch.exp(loss).item() < 10000:
                        perplexity.append(torch.exp(loss).item())

                    current_sentence = ''
                    current_pred = []
                    current_truth = []

                elif Y[i].item() == Emb.key_to_index['pad'] or Y[i].item() == Emb.key_to_index['sos']:
                    continue
                else:
                    current_sentence += Emb.index_to_key[Y[i].item()] + ' '
                    current_pred.append(Y_pred[i])
                    current_truth.append(Y[i])

        print(f'Perplexity: {np.mean(perplexity)}')
        return np.mean(perplexity)

def get_all_perplexity_vals(test_dataloader, cfg, Emb):
    best_model = NNLM(Emb, cfg.hidden_dim, cfg.dropout, pp.device).to(pp.device)
    best_pth = os.path.join(dir, 'best_model.pth')

    return run_perplexity(test_dataloader, best_model, best_pth, Emb)

# %%
# WANDB init
import wandb

def run_everything(config=None):
    with wandb.init(config=config):
        cfg = wandb.config
        Emb = pp.create_vocab(train_sents, cfg.embedding_dim)
        print(len(Emb.key_to_index))

        train_dataloader, validation_dataloader, test_dataloader = load_data(Emb, cfg.batch_size)

        val_loss = train(train_dataloader, validation_dataloader, cfg, Emb)
        wandb.log({'val_loss': val_loss})

        train_perplexity = get_all_perplexity_vals(train_dataloader, cfg, Emb)
        test_perplexity = get_all_perplexity_vals(test_dataloader, cfg, Emb)

        wandb.log({'train_perplexity': train_perplexity})
        wandb.log({'test_perplexity': test_perplexity})

        return val_loss, test_perplexity
    
sweep_id = wandb.sweep(config, project="Assignment_1")
wandb.agent(sweep_id, run_everything, count=20)

# %%
# from nnlm import NNLM

# best_model = NNLM(Emb, cfg.hidden_dim, cfg.dropout, pp.device).to(pp.device)
# best_pth = os.path.join(dir, 'best_model.pth')


# %%
# import tqdm
# # test
# def run_perplexity(dataloader, f):
#     best_model.load_state_dict(torch.load(best_pth))
#     best_model.eval()

#     loss_fn = nn.CrossEntropyLoss()

#     with torch.no_grad():
#         # epoch_loss = []
#         perplexity = []

#         current_sentence = ''
#         current_pred = []
#         current_truth = []

#         for X, Y in tqdm.tqdm(dataloader):
#             Y_pred = best_model(X)

#             for i in range(Y.shape[0]):
#                 if Y[i].item() == Emb.key_to_index['eos']:
#                     if len(current_pred) == 0:
#                         continue

#                     current_pred = torch.stack(current_pred).to(pp.device)
#                     current_truth = torch.tensor(current_truth).to(pp.device)
#                     loss = loss_fn(current_pred, current_truth)

#                     if torch.exp(loss).item() < 10000:
#                         perplexity.append(torch.exp(loss).item())
#                         print(f'{current_sentence.strip()}: {perplexity[-1]}', file=f)

#                     current_sentence = ''
#                     current_pred = []
#                     current_truth = []

#                 elif Y[i].item() == Emb.key_to_index['pad'] or Y[i].item() == Emb.key_to_index['sos']:
#                     continue
#                 else:
#                     current_sentence += Emb.index_to_key[Y[i].item()] + ' '
#                     current_pred.append(Y_pred[i])
#                     current_truth.append(Y[i])

#         print(f'Average Perplexity: {np.mean(perplexity)}', file=f)
#         print(f'Average Perplexity: {np.mean(perplexity)}', file=f)

# %%
# with open(os.path.join(dir, 'train.txt'), 'w') as f:
#     run_perplexity(train_dataloader, f)

# with open(os.path.join(dir, 'val.txt'), 'w') as f:
#     run_perplexity(validation_dataloader, f)

# with open(os.path.join(dir, 'test.txt'), 'w') as f:
#     run_perplexity(test_dataloader, f)

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


