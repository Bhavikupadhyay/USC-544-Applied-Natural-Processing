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


def task2_test(model, data_loader, device='cpu', verbose=False):
    label2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                 'I-MISC': 8, 'PAD-': 9}

    model = model.to(device)
    model.eval()

    with torch.no_grad():
        all_true, all_pred = [], []

        for inputs, caps, labels in data_loader:
            inputs, caps, labels = inputs.to(device), caps.to(device), labels.to(device)

            outputs = model(inputs, caps)

            trues = labels.view(-1).cpu().numpy()
            _, predicted = torch.max(outputs, 2)
            preds = predicted.view(-1).cpu().numpy()

            for true, pred in zip(trues, preds):
                if true != label2idx['PAD-']:
                    all_true.append(true)
                    all_pred.append(pred)

        label_map = {label: sym for sym, label in label2idx.items()}
        all_true = [label_map[true] for true in all_true]
        all_pred = [label_map[pred] for pred in all_pred]

        res = evaluate(all_true, all_pred, verbose=verbose)
        return res


def task2_train(model, train_loader, val_loader, optimizer, criterion, out_dim=9, num_epochs=10, patience=5,
                device='cpu', path='./task2-bilstm.pth'):
    model = model.to(device)
    curr_patience = 0
    best_model = copy.deepcopy(model)
    best_f1 = 0

    for epoch in range(num_epochs):
        model.train()

        for inputs, caps, labels in train_loader:
            optimizer.zero_grad()
            inputs, caps, labels = inputs.to(device), caps.to(device), labels.to(device)
            outputs = model(inputs, caps)

            loss = criterion(outputs.view(-1, out_dim), labels.view(-1))
            loss.backward()
            optimizer.step()

        prec, rec, f1 = task2_test(model, val_loader, device=device)
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

    return best_model


def task2_load_dataset_and_npa():
    dataset = datasets.load_dataset('conll2003')
    new_dataset = dataset.map(lambda sample: {'labels': sample['ner_tags']},
                              remove_columns=['id', 'ner_tags', 'pos_tags', 'chunk_tags'])
    print('Loaded the dataset')
    vocab_npa = np.load('./vocab_npa.npy')
    embs_npa = np.load('./embs_npa.npy')

    vocab2idx = {word: idx for idx, word in enumerate(vocab_npa)}
    label2idx = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7,
                 'I-MISC': 8, 'PAD-': 9}
    print('Processing the dataset')
    new_dataset2 = new_dataset.map(
        lambda x: {
            'isCap': [int(token[0].isupper()) for token in x['tokens']]
        }
    )
    print('Created isCap column in dataset')

    new_dataset2 = new_dataset2.map(
        lambda x: {
            'input_ids': [vocab2idx.get(token.lower(), vocab2idx['<unk>']) for token in x['tokens']]
        },
        remove_columns=['tokens']
    )
    print('Created input_ids using lower case words and vocab2idx')

    print('Creating dataloaders')
    batch_size = 32
    pad_list = (vocab2idx['<pad>'], label2idx['PAD-'], 0)

    train_seq_list = (new_dataset2['train']['input_ids'], new_dataset2['train']['labels'], new_dataset2['train']['isCap'])
    train_data, train_labels, train_caps = create_padded_sequences(train_seq_list, pad_list)

    val_seq_list = (new_dataset2['validation']['input_ids'], new_dataset2['validation']['labels'], new_dataset2['validation']['isCap'])
    val_data, val_labels, val_caps = create_padded_sequences(val_seq_list, pad_list)

    test_seq_list = (new_dataset2['test']['input_ids'], new_dataset2['test']['labels'], new_dataset2['test']['isCap'])
    test_data, test_labels, test_caps = create_padded_sequences(test_seq_list, pad_list)

    train_loader = DataLoader(
        TensorDataset(train_data, train_caps, train_labels),
        batch_size=batch_size,
        shuffle=True
    )

    val_loader = DataLoader(
        TensorDataset(val_data, val_caps, val_labels),
        batch_size=batch_size,
        shuffle=False
    )

    test_loader = DataLoader(
        TensorDataset(test_data, test_caps, test_labels),
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, val_loader, test_loader, embs_npa


if __name__ == '__main__':
    embedding_dim = 100
    hidden_dim = 256
    dropout_prob = 0.33
    linear_dim = 128
    out_dim = 9

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, test_loader, embs_npa = task2_load_dataset_and_npa()
    model = BiLSTM2(embedding_dim, hidden_dim, linear_dim, out_dim, dropout_prob, embs_npa)
    model.load_state_dict(torch.load('./task2-bilstm.pth', map_location=device))

    prec, rec, f1 = task2_test(model, test_loader, device=device, verbose=True)
    print('F1: ', f1, 'Recall:', rec, 'Precision:', prec)