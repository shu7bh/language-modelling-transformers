from nltk import sent_tokenize, word_tokenize
from gensim.models import KeyedVectors
import gensim.downloader as api
import unicodedata
import random
import torch
import re

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def normalize_unicode(text):
    return unicodedata.normalize('NFD', text)

def tokenize_corpus(s):
    s = normalize_unicode(s)
    s = s.lower()
    s = re.sub(r"""[^a-zA-Z0-9?.,'"]+""", " ", s)
    s = re.sub(r'(.)\1{3,}',r'\1', s)
    s = s.rstrip().strip()
    return s

def preprocess_corpus(file: str):
    # Read corpus
    corpus = open(file, 'r').read()

    print(len(corpus))

    # Preprocess corpus
    prep = tokenize_corpus(corpus)

    # Tokenize corpus
    doc = sent_tokenize(prep)

    doc = [s for s in doc if not s.startswith('chapter') and not s[0].isdigit()]

    return doc

def get_sents(file: str, train_len: int, validation_len: int, test_len: int):
    doc = preprocess_corpus(file)

    random.seed(25)
    random.shuffle(doc)

    train_sents = list(doc)[:train_len]
    validation_sents = list(doc)[train_len:train_len + validation_len]
    test_sents = list(doc)[train_len + validation_len:train_len + validation_len + test_len]

    print(len(train_sents))
    print(len(validation_sents))
    print(len(test_sents))

    print (train_sents[0])
    return train_sents, validation_sents, test_sents

glove_dict = {
    '100': 'glove-wiki-gigaword-100',
    '200': 'glove-wiki-gigaword-200'
}

glove_dict['100'] = api.load(glove_dict['100'])
glove_dict['200'] = api.load(glove_dict['200'])

def create_vocab(sentences: list, embedding_dim: int):
    glove = glove_dict[str(embedding_dim)]

    Emb = KeyedVectors(vector_size=glove.vector_size)
    vocab = set()

    for sentence in sentences:
        vocab.update(word_tokenize(sentence))

    vocab.add('unk')
    vocab.add('sos')
    vocab.add('eos')
    vocab.add('pad')

    vectors, keys = [], []
    for token in vocab:
        if token in glove:
            vectors.append(glove[token])
            keys.append(token)

    Emb.add_vectors(keys, vectors)

    return Emb

def get_sentence_vector(sentence: list, Emb: KeyedVectors):
    word_vec = []
    for word in sentence:
        word_vec.append(get_word_vector(word, Emb))
    return torch.stack(word_vec)

def get_word_vector(word: str, Emb: KeyedVectors):
    if word in Emb:
        return torch.from_numpy(Emb[word])
    return torch.from_numpy(Emb['unk'])

def get_sentence_index(sentence: str, Emb: KeyedVectors):
    word_vec = []

    word_vec.append(Emb.key_to_index['sos'])
    for word in word_tokenize(sentence):
        word_vec.append(get_vocab_index(word, Emb))
    word_vec.append(Emb.key_to_index['eos'])

    return torch.tensor(word_vec)

def get_sentence_index_pad(sentence: str, Emb: KeyedVectors):
    word_vec = [Emb.key_to_index['pad']] * 4
    word_vec.append(Emb.key_to_index['sos'])
    for word in word_tokenize(sentence):
        word_vec.append(get_vocab_index(word, Emb))
    word_vec.append(Emb.key_to_index['eos'])

    return torch.tensor(word_vec)

def get_vocab_index(word: str, Emb: KeyedVectors):
    if word in Emb:
        return Emb.key_to_index[word]
    return Emb.key_to_index['unk']