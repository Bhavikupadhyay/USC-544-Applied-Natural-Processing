Here is an example README file that could be included to document how to run the code:

# Named Entity Recognition Model Evaluation

This code evaluates pre-trained NER models on the CoNLL-2003 dataset.

## Usage

If you want to run the model for a specific task just on the test set (as required in the questions 2 and 3), you can run the specific task.py file.

For example, in order to answer question 2, by evaluating the bi-lstm model without glove embeddings (which is task 1) on test set, you can run:

`python task1.py`

Run the code for a specific dataset and task with:

```bash
python main.py --task TASK --dataset DATASET --verbose VERBOSE
```

Where:

- `--task`: Task number (1, 2, or 3) 
- `--dataset`: 1 for validation, 2 for test
- `--verbose`: 0 for no output, 1 for verbose

Example:

```bash 
python program.py --task 1 --dataset 1 --verbose 1
```

## Tasks

The following pre-trained models are evaluated:

- Task 1: Bidirectional LSTM 
- Task 2: Bidirectional LSTM + GloVe embeddings
- Task 3: Transformer Encoder

## Datasets

- Validation Set (`--dataset 1`)
- Test set (`--dataset 2`)

## Verbose

This controls the verbosity of the conlleval.evaluate() function. You can get more detailed outputs by using `--verbose 1`. If you want just the f1, recall, and precision scores, use `--verbose 0`.

## Output 

The code prints precision, recall, and F1 score for the chosen model on the selected dataset with chosen verbosity.

## Requirements

The code requires PyTorch and these libraries:

- argparse
- datasets
- accelerate

Install requirements with:

```
pip install -r requirements.txt
```

Let me know if you would like any changes or have any other questions! A README can be as simple or detailed as needed.