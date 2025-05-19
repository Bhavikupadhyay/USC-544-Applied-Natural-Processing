import json
import numpy as np
from collections import Counter, defaultdict
import os

def create_vocabulary(data, threshold=2, return_counter=False):
    words_counter = Counter()
    tags_counter = Counter()
    
    # update the counters for the presented data
    for datum in data:
        words_counter.update(datum['sentence'])
        tags_counter.update(datum['labels'])

    # obtain the counts of words less than threshold
    unk, count_unk = '<unk>', 0
    for word, count in words_counter.items():
        if count < threshold:
            count_unk += count

    # update the word counter by only keeping the words having frequency greater than or equal to threshold
    words_counter = {key: value for key, value in words_counter.items() if value >= threshold}

    # sorting the counters in decreasing order of frequency, which is stored in index 1
    words_counter = {key: value for key, value in
                     sorted(words_counter.items(), key=lambda datum: datum[1], reverse=True)}
    tags_counter = {key: value for key, value in sorted(tags_counter.items(), key=lambda datum: datum[1], reverse=True)}

    vocab_list, tags_list = [], []
    
    # the first element in the vocabulary must be the special <unk> tag
    vocab_list.append([unk, 0, count_unk])

    ind = 1
    # creating the word vocabulary from words_counter
    for word, count in words_counter.items():
        vocab_list.append([word, ind, count])
        ind += 1

    ind = 0
    # creating the word vocabulary from tags_counter
    for tag, count in tags_counter.items():
        tags_list.append([tag, ind, count])
        ind += 1

    if return_counter:
        return vocab_list, tags_list, tags_counter, words_counter


def create_mapper(item_list):
    item2ind = {datum[0]: datum[1] for datum in item_list}
    ind2item = {datum[1]: datum[0] for datum in item_list}

    return item2ind, ind2item

def accuracy(y_true, y_preds):
    correct, total = 0, 0

    for true_sentence, pred_sentence in zip(y_true, y_preds):
        correct += sum(true_sentence == pred_sentence)
        total += len(true_sentence)
    return correct / total

class HMM:
    def __init__(self, tags_counter, word2ind, ind2word, tag2ind, ind2tag):
        self.transition_matrix = None
        self.emission_matrix = None
        self.pi_matrix = None
        self.word2ind = word2ind
        self.ind2word = ind2word
        self.tag2ind = tag2ind
        self.ind2tag = ind2tag

        self.tags_counter = tags_counter

        self.num_tags = len(self.tag2ind)
        self.num_words = len(self.word2ind)

    def create_hmm_from_data(self, data, fill_value=1e-6):
        transition_dict = self.create_transition_dict(data)
        emission_dict = self.create_emission_dict(data)
        pi_dict = self.create_pi_dict(data)

        # initialize the hmm matrices with a default value = fill_value, which can be used as smoothing (if fill_value=0, no smoothing)
        self.transition_matrix = np.full(shape=(self.num_tags, self.num_tags), fill_value=fill_value, dtype='float')
        self.emission_matrix = np.full(shape=(self.num_tags, self.num_words), fill_value=fill_value, dtype='float')
        self.pi_matrix = np.full(shape=self.num_tags, fill_value=fill_value, dtype='float')

        # write the values from transition dict to their corresponding index in the transition_matrix which is a np.array
        for (s, s_prime), prob in transition_dict.items():
            self.transition_matrix[self.tag2ind[s], self.tag2ind[s_prime]] = prob

        # write the values from emission dict to corresponding index in emission_matrix (np.array)
        for (s, x), prob in emission_dict.items():
            if x in self.word2ind.keys():
                self.emission_matrix[self.tag2ind[s], self.word2ind[x]] = prob
            else:
                self.emission_matrix[self.tag2ind[x], self.word2ind[unk]] = prob

        for s, prob in pi_dict.items():
            self.pi_matrix[self.tag2ind[s]] = prob

    def create_hmm_from_json(self, hmm_file):
        with open(hmm_file) as hmm_file:
            hmm_data = json.load(hmm_file)

        self.transition_matrix = np.array(hmm_data['transition'])
        self.emission_matrix = np.array(hmm_data['emission'])
        self.pi_matrix = np.array(hmm_data['pi'])

    def create_transition_dict(self, data):
        transition_probs = defaultdict(float)

        for datum in data:
            num_labels = len(datum['labels'])
            for i in range(num_labels - 1):
                s = datum['labels'][i]
                s_prime = datum['labels'][i + 1]
                transition = (s, s_prime)

                transition_probs[transition] += 1. / self.tags_counter[s]

        return transition_probs

    def create_emission_dict(self, data, unk='<unk>'):
        emission_probs = defaultdict(float)

        for datum in data:
            num_words = len(datum['sentence'])

            for i in range(num_words):
                word = datum['sentence'][i]
                x = word if word in self.word2ind.keys() else unk
                s = datum['labels'][i]
                emission = (s, x)

                emission_probs[emission] += 1. / self.tags_counter[s]

        return emission_probs

    def create_pi_dict(self, data):
        pi_probs = defaultdict(int)

        for datum in data:
            s = datum['labels'][0]
            pi_probs[s] += 1. / len(data)

        return pi_probs

    def greedy_decode(self, sentence, unk='<unk>'):
        # convert the sentence from strings to indexes which can be used in the hmm matrices
        words = np.array([self.word2ind[word] if word in word2ind.keys() else word2ind[unk] for word in sentence])

        y_preds = np.zeros(shape=len(words))

        # get the first tag
        y_prev = np.argmax(self.pi_matrix * self.emission_matrix[:, words[0]])

        y_preds[0] = y_prev

        for i in range(1, len(words)):
            word = words[i]
            
            # get the next tag based on the previous tag and current word
            y_prev = np.argmax(self.transition_matrix[y_prev, :] * self.emission_matrix[:, word])
            y_preds[i] = y_prev

        # convert the indexes to tags so that they can be human-interpretable
        Y = np.array([self.ind2tag[ind] for ind in y_preds])
        return Y

    def viterbi_decode(self, sentence, unk='<unk>'):
        """
        Following the near-succinct python version in python
        Tm: hmm.transition_matrix
        Em: hmm.emission_matrix
        pi: hmm.pi_matrix
        
        O: sentence
        S: state sequence (here, equal to indexes from 0 to 44 or range(len(tags2id) as they're represented with indexes in hmm matrices)
        
        trellis = T1
        pointer = T2
        """
        
        # converting strings to indexes so that they can be used with hmm matrices
        words = np.array([self.word2ind[word] if word in word2ind.keys() else word2ind[unk] for word in sentence])
        num_words = len(words)
        
        # to hold probabilities for each state and observation pair
        T1 = np.zeros(shape=(self.num_tags, num_words), dtype='float')
        
        # to hold the back-pointer pointing to the prior state from which we obtain the current state with highest probability
        T2 = np.zeros(shape=(self.num_tags, num_words), dtype='int')

        # initializing trellis with initial probabilities
        for state in range(self.num_tags):
            T1[state, 0] = self.pi_matrix[state] * self.emission_matrix[state, words[0]]

        for obs in range(1, num_words):
            for state in range(self.num_tags):
                word = words[obs]

                k = np.argmax(T1[:, obs-1] * self.transition_matrix[:, state] * self.emission_matrix[state, word])
                T2[state, obs] = k
                T1[state, obs] = T1[k, obs-1] * self.transition_matrix[k, state] * self.emission_matrix[state, word]

        best_path = []
        k = np.argmax(T1[:, num_words-1])
        for obs in reversed(range(num_words)):
            best_path.insert(0, k)
            k = T2[k, obs]

        # convert the indexes to tags so that they can be human-interpretable
        Y = np.array([self.ind2tag[ind] for ind in best_path])
        return Y


# simply change the path below to reach the 'data' folder and load the data.
# Ensure the 'data' folder contains the three json files.
data_path = './data'
with open(os.path.join(data_path, 'train.json')) as train_file:
    train_data = json.load(train_file)

with open(os.path.join(data_path, 'dev.json')) as dev_file:
    dev_data = json.load(dev_file)

with open(os.path.join(data_path, 'test.json')) as test_file:
    test_data = json.load(test_file)

# creating the word and tags vocabulary
vocab_list, tags_list, tags_counter, _ = create_vocabulary(train_data, return_counter=True)

# creating the 'out' folder if not present
if not os.path.exists('./out'):
    os.makedirs('./out')
    
# writing the vocabulary to vocab.txt
with open('./out/vocab.txt', 'w') as vocab_file:
    for datum in vocab_list:
        vocab_file.write(f'{datum[0]}\t{datum[1]}\t{datum[2]}\n')

# creating mapper dictionaries for words and tags
word2ind, ind2word = create_mapper(vocab_list)
tag2ind, ind2tag = create_mapper(tags_list)

# creating the HMM from training data
hmm = HMM(tags_counter, word2ind, ind2word, tag2ind, ind2tag)
hmm.create_hmm_from_data(train_data, 1e-7)

with open('./out/hmm.json', 'w') as hmm_file:
    json.dump({
        'transition': hmm.transition_matrix.tolist(),
        'emission': hmm.emission_matrix.tolist(),
        'pi': hmm.pi_matrix.tolist()
    }, hmm_file)

X_dev = [datum['sentence'] for datum in dev_data]
Y_true_dev = [datum['labels'] for datum in dev_data]

# predicting tags for dev data using greedy decoding
Y_preds_greedy = [hmm.greedy_decode(sentence) for sentence in X_dev]

# predicting tags for dev data using viterbi decoding
Y_preds_viterbi = [hmm.viterbi_decode(sentence) for sentence in X_dev]

# calculating and printing the accuracies
print('Greedy Accuracy:', accuracy(Y_true_dev, Y_preds_greedy))
print('Viterbi Accuracy:', accuracy(Y_true_dev, Y_preds_viterbi))

# predicting tags for test data using greedy decoding and viterbi decoding
greedy_results, viterbi_results = [], []
for datum in test_data:
    sentence = datum['sentence']
    index = datum['index']
    greedy_labels = hmm.greedy_decode(sentence)
    viterbi_labels = hmm.viterbi_decode(sentence)

    greedy_datum = {'index': index, 'sentence': sentence, 'labels': greedy_labels.tolist()}
    viterbi_datum = {'index': index, 'sentence': sentence, 'labels': viterbi_labels.tolist()}

    greedy_results.append(greedy_datum)
    viterbi_results.append(viterbi_datum)

# writing the predicted tags for greedy decoding on test data to greedy.json
with open('./out/greedy.json', 'w') as greedy_file:
    json.dump(greedy_results, greedy_file)

# writing the predicted tags for viterbi decoding on test data to viterbi.json
with open('./out/viterbi.json', 'w') as viterbi_file:
    json.dump(viterbi_results, viterbi_file)
