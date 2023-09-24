# %%
from part_3_config import config as cfg
import preprocess as pp
import numpy as np
import wandb
import torch
import os

# %%
TRAIN_LEN = cfg['parameters']['train_len']['value']
VALIDATION_LEN = cfg['parameters']['validation_len']['value']
TEST_LEN = cfg['parameters']['test_len']['value']

# TRAIN_LEN = 1000
# VALIDATION_LEN = 100
# TEST_LEN = 100

MAX_LEN = cfg['parameters']['max_len']['value']

dir = 'RES/TF/TEST'

# %%
train_sents, validation_sents, test_sents = pp.get_sents('Auguste_Maquet.txt', TRAIN_LEN, VALIDATION_LEN, TEST_LEN)

# %%
import os

if not os.path.exists(dir):
    os.makedirs(dir)

# %%
from torch.utils.data import Dataset
from gensim.models import KeyedVectors
import torch

class SentencesDataset(Dataset):
    def __init__(self, sentences: list, Emb: KeyedVectors, max_len: int = None):
        super().__init__()

        if max_len is not None:
            SentencesDataset.max_len = max_len + 1

        self.X = []
        self.Y = []

        for sentence in sentences:
            s = pp.get_sentence_index(sentence, Emb)
            max_sentence_len = min(SentencesDataset.max_len, len(s))

            self.X.append(torch.cat((s[:max_sentence_len], torch.empty(SentencesDataset.max_len - max_sentence_len, dtype=torch.long).fill_(Emb.key_to_index['pad']))))

            # self.Y.append(s[max_sentence_len])
            # for i in range(max_sentence_len):
            #     self.X.append(torch.cat((s[:i], torch.empty(SentencesDataset.max_len - i, dtype=torch.long).fill_(Emb.key_to_index['pad']))))
            #     self.Y.append(s[i])

        self.X = torch.stack(self.X)
        # self.Y = torch.stack(self.Y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        # return self.X[idx], self.Y[idx]
        return self.X[idx]

# %%
from torch.utils.data import DataLoader

def get_dataloader(dataset, batch_size, shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

# %%
def load_data(Emb, batch_size, device, max_len):
    train_dataset = SentencesDataset(train_sents, Emb, max_len)
    validation_dataset = SentencesDataset(validation_sents, Emb)
    test_dataset = SentencesDataset(test_sents, Emb)

    train_dataloader = get_dataloader(train_dataset, batch_size, True)
    validation_dataloader = get_dataloader(validation_dataset, batch_size, True)
    test_dataloader = get_dataloader(test_dataset, batch_size, False)

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

    for X in pbar:
        # print(X[0])
        Y = X[:, 1:]
        X = X[:, :-1]

        # print(X[0])
        # print(Y[0])

        # break

        Y_pred = model(X)
        Y = Y.to(device)

        Y_pred = Y_pred.view(-1, Y_pred.shape[-1])
        Y = Y.view(-1)

        loss = loss_fn(Y_pred, Y)
        epoch_loss.append(loss.item())

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        pbar.set_description(f'{epoch} {"T" if train else "V"} Loss: {loss.item():7.4f}, Avg Loss: {sum(epoch_loss) / len(epoch_loss):7.4f}, Best Loss: {es.best_loss:7.4f}, Counter: {es.counter}')

    return np.mean(epoch_loss)

# %%
import wandb

def train_epoch(model, train_dataloader, validation_dataloader, es, device, loss_fn, optimizer, epoch):
    train_loss = run(model, train_dataloader, True, es, device, loss_fn, optimizer, epoch)
    wandb.log({'train_loss': train_loss})
    with torch.no_grad():
        validation_loss = run(model, validation_dataloader, False, es, device, loss_fn, optimizer, epoch)
        wandb.log({'validation_loss': validation_loss})
    print(f'Epoch {epoch} Train Loss: {train_loss:7.4f}, Validation Loss: {validation_loss:7.4f}')
    return train_loss, validation_loss

# %%
from transformer import Decoder
from EarlyStopping import EarlyStopping
import torch.nn as nn

def train(train_dataloader, validation_dataloader, cfg, Emb):

    # nhead = cfg['parameters']['nhead']['value']
    # dim_feedforward = cfg['parameters']['dim_feedforward']['value']
    # num_layers = cfg['parameters']['num_layers']['value']
    # dropout = cfg['parameters']['dropout']['value']
    # max_len = cfg['parameters']['max_len']['value']
    # epochs = cfg['parameters']['epochs']['value']
    # learning_rate = cfg['parameters']['learning_rate']['value']
    # optimizer = cfg['parameters']['optimizer']['value']

    nhead = cfg.nhead
    dim_feedforward = cfg.dim_feedforward
    num_layers = cfg.num_layers
    dropout = cfg.dropout
    max_len = cfg.max_len
    epochs = cfg.epochs
    learning_rate = cfg.learning_rate
    optimizer = cfg.optimizer

    model = Decoder(Emb, nhead, dim_feedforward, num_layers, dropout, max_len, pp.device).to(pp.device)
    # print(model)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = getattr(torch.optim, optimizer)(model.parameters(), lr=learning_rate)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    es = EarlyStopping(patience=3)

    for epoch in range(epochs):
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

        pbar = tqdm.tqdm(dataloader)
        for X in pbar:
            Y = X[:, 1:]
            X = X[:, :-1]

            # tgt_key_padding_mask = (X == Emb.key_to_index['pad']).transpose(0, 1)

            Y_pred = best_model(X)
            Y = Y.to(pp.device)

            for i in range(Y_pred.shape[0]):
                pval = 0
                pix = 0
                sentence = ''
                
                # print(X.shape)
                # print(Y_pred.shape)
                # print(Y.shape)

                for j in range(1, X.shape[1]):
                    if X[i][j].item() == Emb.key_to_index['eos']:
                        break
                    sentence += Emb.index_to_key[X[i][j].item()] + ' '
                    pix = j + 1

                # print(sentence.strip())
                # print(Y_pred[i][:pix].shape)
                # print(Y[i][:pix].shape)
                # print(Y_pred[i][:pix])
                # print(Y[i][:pix])

                # print(loss_fn(Y_pred[i][:pix], Y[i][:pix]).item())

                pval = np.exp(loss_fn(Y_pred[i][:pix], Y[i][:pix]).item())
                # print(f'{sentence.strip()}: {pval}', file=f)
                perplexity.append(pval)

            #     break
            # break

        # print(f'Perplexity: {np.mean(perplexity)}')
        return np.mean(perplexity)

def get_all_perplexity_vals(test_dataloader, cfg, Emb):

    # nhead = cfg['parameters']['nhead']['value']
    # dim_feedforward = cfg['parameters']['dim_feedforward']['value']
    # num_layers = cfg['parameters']['num_layers']['value']
    # dropout = cfg['parameters']['dropout']['value']
    # max_len = cfg['parameters']['max_len']['value']

    nhead = cfg.nhead
    dim_feedforward = cfg.dim_feedforward
    num_layers = cfg.num_layers
    dropout = cfg.dropout
    max_len = cfg.max_len

    best_model = Decoder(Emb, nhead, dim_feedforward, num_layers, dropout, max_len, pp.device).to(pp.device)
    # print(best_model)
    best_pth = os.path.join(dir, 'best_model.pth')

    # with open(os.path.join(dir, 'test_perplexity.txt'), 'w') as f:
    return run_perplexity(test_dataloader, best_model, best_pth, Emb)
    
    # return run_perplexity(test_dataloader, best_model, best_pth, Emb)

# %%
# WANDB init
import wandb

def run_everything(cfg=None):
    with wandb.init(config=cfg):
        cfg = wandb.config

        # embedding_dim = cfg['parameters']['embedding_dim']['value']
        # batch_size = cfg['parameters']['batch_size']['value']
        # max_len = cfg['parameters']['max_len']['value']

        embedding_dim = cfg.embedding_dim
        batch_size = cfg.batch_size
        max_len = cfg.max_len

        Emb = pp.create_vocab(train_sents, embedding_dim)
        print(len(Emb.key_to_index))

        train_dataloader, validation_dataloader, test_dataloader = load_data(Emb, batch_size, pp.device, max_len)

        val_loss = train(train_dataloader, validation_dataloader, cfg, Emb)
        wandb.log({'best_loss': val_loss})

        train_perplexity = get_all_perplexity_vals(train_dataloader, cfg, Emb)
        test_perplexity = get_all_perplexity_vals(test_dataloader, cfg, Emb)

        wandb.log({'train_perplexity': train_perplexity})
        wandb.log({'test_perplexity': test_perplexity})

sweep_id = wandb.sweep(cfg, project="Transformer")
wandb.agent(sweep_id, run_everything, count=50)

# %%
# run_everything(cfg)

# %%
# from part_3_config import config as cfg
# import preprocess as pp

# embedding_dim = cfg['parameters']['embedding_dim']['value']
# batch_size = cfg['parameters']['batch_size']['value']
# max_len = cfg['parameters']['max_len']['value']

# Emb = pp.create_vocab(train_sents, embedding_dim)
# train_dataloader, validation_dataloader, test_dataloader = load_data(Emb, batch_size, pp.device, max_len)

# %%
# get_all_perplexity_vals(test_dataloader, cfg, Emb)

# %%
# nhead = cfg['parameters']['nhead']['value']
# dim_feedforward = cfg['parameters']['dim_feedforward']['value']
# num_layers = cfg['parameters']['num_layers']['value']
# dropout = cfg['parameters']['dropout']['value']
# max_len = cfg['parameters']['max_len']['value']

# best_model = Decoder(Emb, nhead, dim_feedforward, num_layers, dropout, max_len, pp.device).to(pp.device)
# print(best_model)
# best_pth = os.path.join(dir, 'best_model.pth')

# # generate a sentence from the model

# q = 'my name is '
# print(q, end=' ')

# best_model.load_state_dict(torch.load(best_pth))
# best_model.eval()

# with torch.no_grad():
#     for i in range(10):
#         e = pp.get_sentence_index(q, Emb)
#         e = e[:-1]
#         # print(e)
#         X = torch.cat((e, torch.empty(max_len - len(e), dtype=torch.long).fill_(Emb.key_to_index['pad'])))
#         # print (X)
#         Y_pred = best_model(X)
#         # print(Y_pred.shape)
#         Y_pred = Y_pred[0][len(e) - 1]
#         Y_pred = torch.softmax(Y_pred, dim=-1)
#         # Y_pred = torch.multinomial(Y_pred, num_samples=1)
#         Y_pred = torch.argmax(Y_pred)


#         # print(Y_pred.shape)

#         # for j in range(len(e)):
#         #     a = torch.softmax(Y_pred[0, j], dim=0)
#         #     # b = torch.multinomial(a, num_samples=1)
#         #     b = torch.argmax(a)
#         #     print(Emb.index_to_key[e[j].item()], Emb.index_to_key[b.item()])
#         # print (Y_pred)
#         # Y_pred = Y_pred[0, len(e) - 1]
#         # # Y_pred[len(e)]
#         # Y_pred = torch.softmax(Y_pred, dim=-1)
#         # print(Y_pred)
#         # Y_pred = torch.multinomial(Y_pred, num_samples=1)

#         q += ' ' + Emb.index_to_key[Y_pred.item()]
#         # print the word
#         print(Emb.index_to_key[Y_pred.item()], end=' ')

# %%



