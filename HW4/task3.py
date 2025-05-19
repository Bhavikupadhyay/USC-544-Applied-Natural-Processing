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

from task1 import task1_load_dataset
from utils import generate_true_and_pred, create_padded_sequences
from models import *

def task3_test(model, data_loader, device='cpu', verbose=False):
    label2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                 'I-MISC': 8, 'PAD-': 9}
    model = model.to(device)
    model.eval()

    all_true, all_pred = [], []
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.transpose(0, 1).to(device), labels.transpose(0, 1).to(device)

            outputs = model(data)
            trues, preds = generate_true_and_pred(outputs, labels)

            all_true.extend(trues)
            all_pred.extend(preds)

    label_map = {label: sym for sym, label in label2idx.items()}
    all_true = [label_map[true] for true in all_true]
    all_pred = [label_map[pred] for pred in all_pred]

    res = evaluate(all_true, all_pred, verbose=verbose)

    return res


def task3_train(model, train_loader, val_loader, criterion, optimizer, num_epochs=5, device='cpu', verbose=False):
    model = model.to(device)

    for epoch in range(num_epochs):
        model.train()
        for data, labels in train_loader:
            data, labels = data.transpose(0, 1).to(device), labels.transpose(0, 1).to(device)
            optimizer.zero_grad()

            src_mask = torch.zeros((data.size(0), data.size(0)), device=device)
            src_pad_mask = (data == word2idx['[PAD]']).transpose(0, 1)

            outputs = model(data, src_key_padding_mask=src_pad_mask)
            outputs = outputs.reshape(-1, out_dim)
            labels = labels.reshape(-1)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()

        all_true, all_pred = [], []
        for data, labels in val_loader:
            data, labels = data.transpose(0, 1).to(device), labels.transpose(0, 1).to(device)

            outputs = model(data)
            trues, preds = generate_true_and_pred(outputs, labels)
            all_true.extend(trues)
            all_pred.extend(preds)

        label_map = {label: sym for sym, label in label2idx.items()}
        all_true = [label_map[true] for true in all_true]
        all_pred = [label_map[pred] for pred in all_pred]

        prec, rec, f1 = evaluate(all_true, all_pred, verbose=verbose)
        print(f'Epoch: {epoch + 1} / {num_epochs}, val_f1: {f1}, val_precision: {prec}, val_recall: {rec}')


if __name__ == '__main__':
    input_dim = 23589
    emb_dim = 128
    num_attention_heads = 8
    max_seq_len = 128
    ff_dim = 128
    out_dim = 9

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = task1_load_dataset()
    model = TransformerEncoderModel(input_dim, emb_dim, num_attention_heads, ff_dim, max_seq_len, out_dim)
    model.load_state_dict(torch.load('./task3-optimus-prime.pth', map_location=device))

    prec, rec, f1 = task3_test(model, test_loader, device=device, verbose=True)
    print('F1: ', f1, 'Recall:', rec, 'Precision:', prec)