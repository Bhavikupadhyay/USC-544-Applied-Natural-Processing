import argparse
import torch
import torch.cuda

from utils import *
from task1 import *
from task2 import *
from task3 import *

from models import *

def main():
    parser = argparse.ArgumentParser(description="""
This program runs named entity recognition (NER) models for different tasks on the CoNLL-2003 dataset.\n
It allows the user to choose between 3 tasks:\n
- Bidirectional LSTM model (1)\n
- Bidirectional LSTM model with GloVe embeddings (2)\n
- Transformer encoder model (3)\n
The user must specify the task number using the --task argument.\n \n

The program can evaluate the model on either the validation set or the test set, specified using the --dataset argument (1 for validation, 2 for test).\n
It prints the F1 scores, Recall and Precision scores for the given choice of dataset and model\n
The --verbose argument allows printing detailed output from the conlleval.evaluate() function You can pass (1) or not (0).\n
""")

    parser.add_argument('--task', type=int, choices=[1, 2, 3], help='Choose task from 1 to 3. (required)', required=True)
    parser.add_argument('--dataset', type=int, choices=[1, 2], help='Press 1 for evaluation on validation set and 2 for evaluation on test set. (required)', required=True)
    parser.add_argument('--verbose', type=int, choices=[0, 1], help='Choose verbosity, 0 for no verbosity, 1 for verbose output. (required)', required=True)

    args = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    task = args.task
    ds = args.dataset
    verbose = args.verbose

    if task == 1:
        # Task 1 configurations
        input_dim = 23589
        embedding_dim = 100
        hidden_dim = 256
        dropout_prob = 0.33
        linear_dim = 128
        out_dim = 9

        train_loader, val_loader, test_loader = task1_load_dataset()
        model = BiLSTM1(input_dim, embedding_dim, hidden_dim, linear_dim, out_dim, dropout_prob)
        model.load_state_dict(torch.load('./task1-bilstm.pth', map_location=device))

        if ds == 1:
            prec, rec, f1 = task1_test(model, val_loader, device=device, verbose=verbose)
            print('F1: ', f1, 'Recall:', rec, 'Precision:', prec)
        if ds == 2:
            prec, rec, f1 = task1_test(model, test_loader, device=device, verbose=verbose)
            print('F1: ', f1, 'Recall:', rec, 'Precision:', prec)

    elif task == 2:
        # Task 2 configurations
        embedding_dim = 100
        hidden_dim = 256
        dropout_prob = 0.33
        linear_dim = 128
        out_dim = 9

        train_loader, val_loader, test_loader, embs_npa = task2_load_dataset_and_npa()
        model = BiLSTM2(embedding_dim, hidden_dim, linear_dim, out_dim, dropout_prob, embs_npa)
        model.load_state_dict(torch.load('./task2-bilstm.pth', map_location=device))

        if ds == 1:
            prec, rec, f1 = task2_test(model, val_loader, device=device, verbose=verbose)
            print('F1: ', f1, 'Recall:', rec, 'Precision:', prec)
        if ds == 2:
            prec, rec, f1 = task2_test(model, test_loader, device=device, verbose=verbose)
            print('F1: ', f1, 'Recall:', rec, 'Precision:', prec)

    elif task == 3:
        # Task 3 configurations
        input_dim = 23589
        emb_dim = 128
        num_attention_heads = 8
        max_seq_len = 128
        ff_dim = 128
        out_dim = 9

        train_loader, val_loader, test_loader = task1_load_dataset()
        model = TransformerEncoderModel(input_dim, emb_dim, num_attention_heads, ff_dim, max_seq_len, out_dim)
        model.load_state_dict(torch.load('./task3-optimus-prime.pth', map_location=device))

        if ds == 1:
            prec, rec, f1 = task3_test(model, val_loader, device=device, verbose=verbose)
            print('F1: ', f1, 'Recall:', rec, 'Precision:', prec)
        if ds == 2:
            prec, rec, f1 = task3_test(model, test_loader, device=device, verbose=verbose)
            print('F1: ', f1, 'Recall:', rec, 'Precision:', prec)


if __name__ == '__main__':
    main()
