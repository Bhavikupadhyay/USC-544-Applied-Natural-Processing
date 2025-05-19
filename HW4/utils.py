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

def generate_true_and_pred(outputs, labels):
    trues = labels.reshape(-1).cpu().numpy()
    _, predicted = torch.max(outputs, 2)
    preds = predicted.reshape(-1).cpu().numpy()

    combined = list(zip(trues, preds))
    filtered = [(t, p) for (t, p) in combined if t != 9]

    trues, preds = zip(*filtered)

    return trues, preds

def create_padded_sequences(seq_list, pad_list):
    padded_seq = []
    for sequence, pad_val in zip(seq_list, pad_list):
        padded_seq.append(
            pad_sequence([torch.tensor(seq) for seq in sequence], batch_first=True, padding_value=pad_val))

    return padded_seq