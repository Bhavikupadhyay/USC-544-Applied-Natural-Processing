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

from utils import generate_true_and_pred, create_padded_sequences
from models import *

def task1_load_dataset():
    dataset = datasets.load_dataset('conll2003')
    new_dataset = dataset.map(lambda sample: {'labels': sample['ner_tags']}, remove_columns=['id', 'ner_tags', 'pos_tags', 'chunk_tags'])

    word_freq = Counter(itertools.chain(*new_dataset['train']['tokens']))

    word2idx = {word: idx for idx, (word, freq) in enumerate(word_freq.items(), start=2) if freq >= 3}
    word2idx['[PAD]'] = 0
    word2idx['[UNK]'] = 1

    label2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                 'I-MISC': 8, 'PAD-': 9}

    new_dataset1 = new_dataset.map(
        lambda x: {
            'input_ids': [word2idx.get(word, word2idx['[UNK]']) for word in x['tokens']],
        },
        remove_columns='tokens'
    )

    pad_list = (word2idx['[PAD]'], label2idx['PAD-'])

    train_seq_list = (new_dataset1['train']['input_ids'], new_dataset1['train']['labels'])
    train_data, train_labels = create_padded_sequences(train_seq_list, pad_list)

    val_seq_list = (new_dataset1['validation']['input_ids'], new_dataset1['validation']['labels'])
    val_data, val_labels = create_padded_sequences(val_seq_list, pad_list)

    test_seq_list = (new_dataset1['test']['input_ids'], new_dataset1['test']['labels'])
    test_data, test_labels = create_padded_sequences(test_seq_list, pad_list)

    batch_size = 32
    train_loader = DataLoader(
        TensorDataset(train_data, train_labels),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(val_data, val_labels),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(test_data, test_labels),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader


def task1_test(model, data_loader, device='cpu', verbose=False):
    label2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                 'I-MISC': 8, 'PAD-': 9}

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        all_true, all_pred = [], []

        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            trues, preds = generate_true_and_pred(outputs, labels)
            all_true.extend(trues)
            all_pred.extend(preds)

        label_map = {label: sym for sym, label in label2idx.items()}
        all_true = [label_map[true] for true in all_true]
        all_pred = [label_map[pred] for pred in all_pred]

        res = evaluate(all_true, all_pred, verbose=verbose)
        return res


def task1_train(model, train_loader, val_loader, optimizer, criterion, out_dim=9, num_epochs=10, patience=5,
                device='cpu', path='./task1-bilstm.pth'):
    curr_patience = 0
    best_model = copy.deepcopy(model)
    best_f1 = 0

    model = model.to(device)
    for epoch in range(num_epochs):
        model.train()

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, out_dim), labels.view(-1))
            loss.backward()
            optimizer.step()

        prec, rec, f1 = task1_test(model, val_loader, device=device)
        print(f'Epoch: {epoch + 1} / {num_epochs}, val_f1: {f1}, val_precision: {prec}, val_recall: {rec}')

        if f1 > best_f1:
            best_f1 = f1
            curr_patience = 0
            best_model = copy.deepcopy(model)
        else:
            curr_patience += 1

        if curr_patience >= patience:
            print(f'Stopping after {epoch + 1} epochs')
            torch.save(best_model.state_dict(), path)
            return best_model

    torch.save(best_model.state_dict(), path)
    return best_model

if __name__ == '__main__':
    input_dim = 23589
    embedding_dim = 100
    hidden_dim = 256
    dropout_prob = 0.33
    linear_dim = 128
    out_dim = 9

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader = task1_load_dataset()
    model = BiLSTM1(input_dim, embedding_dim, hidden_dim, linear_dim, out_dim, dropout_prob)
    model.load_state_dict(torch.load('./task1-bilstm.pth', map_location=device))

    prec, rec, f1 = task1_test(model, test_loader, device=device, verbose=True)
    print('F1: ', f1, 'Recall:', rec, 'Precision:', prec)