import datasets

from conlleval import evaluate

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence

import itertools
from collections import Counter

import copy
import gzip
import numpy as np
import math

class BiLSTM1(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, linear_dim, out_dim, dropout_prob):
        super(BiLSTM1, self).__init__()

        self.emb = nn.Embedding(num_embeddings=input_dim, embedding_dim=embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(2 * hidden_dim, linear_dim)
        self.elu = nn.ELU()
        self.out = nn.Linear(linear_dim, out_dim)

    def forward(self, x):
        out = self.emb(x)
        out, _ = self.lstm(out)
        out = self.drop(out)
        out = self.linear(out)
        out = self.elu(out)
        out = self.out(out)

        return out

class BiLSTM2(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, linear_dim, out_dim, dropout_prob, embs_npa):
        super(BiLSTM2, self).__init__()
        self.emb = nn.Embedding.from_pretrained(torch.from_numpy(embs_npa).float())
        self.lstm = nn.LSTM(embedding_dim + 1, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.drop = nn.Dropout(p=dropout_prob)
        self.linear = nn.Linear(2 * hidden_dim, linear_dim)
        self.elu = nn.ELU()
        self.clf = nn.Linear(linear_dim, out_dim)

    def forward(self, x, caps):
        out = self.emb(x)

        caps = caps.unsqueeze(2)
        out = torch.cat([out, caps], dim=2)

        out, _ = self.lstm(out)
        out = self.drop(out)
        out = self.linear(out)
        out = self.elu(out)
        out = self.clf(out)

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        den = torch.exp(-torch.arange(0, d_model, 2) * math.log(10000) / d_model)
        pos = torch.arange(0, max_len).reshape(max_len, 1)
        pos_embedding = torch.zeros((max_len, d_model))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)

        pos_embedding = pos_embedding.unsqueeze(-2)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, x):
        return x + self.pos_embedding[:x.size(0), :]


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embedding_dim):
        super(TokenEmbedding, self).__init__()

        self.emb = nn.Embedding(input_dim, embedding_dim)
        self.emb_size = embedding_dim

    def forward(self, tokens):
        return self.emb(tokens.long()) * math.sqrt(self.emb_size)


class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, num_attention_heads, ff_dim, max_seq_len, out_dim):
        super(TransformerEncoderModel, self).__init__()

        self.src_token_emb = TokenEmbedding(input_dim, embedding_dim)

        self.pos_enc = PositionalEncoding(embedding_dim, max_seq_len)

        self.enc_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=num_attention_heads,
                                                    dim_feedforward=ff_dim)
        self.encoder = nn.TransformerEncoder(self.enc_layer, num_layers=1)

        self.clf = nn.Linear(embedding_dim, out_dim)

    def forward(self, x, mask=None, src_key_padding_mask=None):
        out = self.src_token_emb(x)
        out = self.pos_enc(out)
        out = self.encoder(out, mask=mask, src_key_padding_mask=src_key_padding_mask)
        out = self.clf(out)

        return out