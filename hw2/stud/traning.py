import json

import numpy as np
from typing import List, Tuple
from typing import List, Dict
import torch
import random
import csv
import matplotlib.pyplot as plt
import nltk
from torch import nn
from torch.utils.data import Dataset, DataLoader
from nltk.data import load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from string import punctuation
from collections import defaultdict
from functools import partial
from typing import Tuple, List, Any, Dict
from sklearn.metrics import confusion_matrix
# import seaborn as sn
# import pandas as pd
import copy
import torchmetrics

# seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# setting up nltk
nltk.download('omw-1.4')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download("tagsets")
stop_tokens = set(stopwords.words('english'))
punc_tokens = set(punctuation)
stop_tokens.update(punc_tokens)
lemmatizer = WordNetLemmatizer()

# setting the embedding dimension
EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 10

# specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print(device)
# setting unknown token to handle out of vocabulary words and padding token to pad sentences
UNK_TOKEN = '<unk>'
PAD_TOKEN = '_' #???????????????
EN_TRAIN_PATH = "./../../data/EN/train.json"
EN_DEV_PATH = "./../../data/EN/dev.json"
print(torch.version.cuda)

SEMANTIC_ROLES = ["AGENT", "ASSET", "ATTRIBUTE", "BENEFICIARY", "CAUSE", "CO_AGENT", "CO_PATIENT", "CO_THEME",
                  "DESTINATION",
                  "EXPERIENCER", "EXTENT", "GOAL", "IDIOM", "INSTRUMENT", "LOCATION", "MATERIAL", "PATIENT", "PRODUCT",
                  "PURPOSE",
                  "RECIPIENT", "RESULT", "SOURCE", "STIMULUS", "THEME", "TIME", "TOPIC", "VALUE","_"]


class SentenceDataset(Dataset):

    def __init__(self, sentences_path=None, sentences=None, lemmatization=False,
                 test=False):
        self.sentences = self.read_file(sentences_path) if sentences_path else self.read_sentences(sentences)
        features = ["words", "lemmas"]
        word_list = self.read_sentences(features)
        self.word2idx = self.create_vocabulary(word_list)
        features = ["pos"]
        pos_list = self.read_sentences(features)
        self.pos2idx = self.create_vocabulary(pos_list)
        self.SEMANTIC_ROLES = ["AGENT", "ASSET", "ATTRIBUTE", "BENEFICIARY", "CAUSE", "CO-AGENT", "CO-PATIENT",
                               "CO-THEME", "DESTINATION",
                               "EXPERIENCER", "EXTENT", "GOAL", "IDIOM", "INSTRUMENT", "LOCATION", "MATERIAL",
                               "PATIENT", "PRODUCT", "PURPOSE",
                               "RECIPIENT", "RESULT", "SOURCE", "STIMULUS", "THEME", "TIME", "TOPIC", "VALUE","_"]
        self.roles2idx = {role.lower(): idx for idx,role in enumerate(self.SEMANTIC_ROLES)}
        # print(file_output)

        '''
        self.embedding_vectors = vectors
        self.word2idx = word2idx
        self.pos_vectors = pos_vectors
        self.pos2idx = pos2idx
        self.test = test
        self.w_lemmatization = lemmatization
        self.extract_sentences(file_output)
        self.class2id = class2id
        self.id2class = {v: k for (k, v) in self.class2id.items()}
        '''

    # little function to read and store a file given the path
    def read_file(self, path):
        sentences = list()
        with open(path) as file:
            json_file = json.load(file)
            for key in json_file:
                # count = json_file[key]["predicates"].count("_")
                # length = len(json_file[key]["predicates"])
                json_file[key]["id"] = key
                if json_file[key]["predicates"].count("_") != len(json_file[key]["predicates"]) and len(
                        json_file[key]["roles"]) > 1:
                    for position in json_file[key]["roles"]:
                        roles = json_file[key]["roles"][position]
                        predicates = ["_" for i in range(len(roles))]
                        predicates[int(position)] = json_file[key]["predicates"][int(position)]
                        instance = copy.deepcopy(json_file[key])
                        instance["roles"] = roles
                        # instance["roles_bin"] = [1 if role != "_" else 0 for role in roles]
                        instance["predicates"] = predicates
                        sentences.append(self.text_preprocess(instance))
                else:
                    instance = copy.deepcopy(json_file[key])
                    if json_file[key]["predicates"].count("_") == len(json_file[key]["predicates"]):
                        roles = ["_" for i in range(len(json_file[key]["predicates"]))]
                        instance["roles"] = roles
                        # instance["roles_bin"] = [1 if role != "_" else 0 for role in roles]
                    else:
                        k = list(instance["roles"].keys())[0]
                        instance["roles"] = instance["roles"][k]
                        # instance["roles_bin"] = [1 if role != "_" else 0 for role in instance["roles"]]
                    sentences.append(self.text_preprocess(instance))
        return sentences

    # little function to read and store a set of sentences given as a list of tokens
    def read_sentences(self, features):
        word_list = set()
        for instance in self.sentences:
            for feature in features:
                for i in range(len(instance[feature])):
                    instance[feature][i] = instance[feature][i].lower()
                    word_list.add(instance[feature][i])
        return sorted(word_list)

    # function for preprocessing, which includes pos tagging and (if specified) lemmatization
    def text_preprocess(self, sentence):
        tokens_n_pos = nltk.pos_tag(sentence["lemmas"])
        # standard_tokens = [(token,self.get_standard(pos)) for token,pos in tokens_n_pos]
        sentence["pos"] = [pos for word, pos in tokens_n_pos]
        return sentence

    def create_vocabulary(self, word_list):
        word2idx = dict()
        for i, word in enumerate(word_list):
            word2idx[word] = i
        return word2idx

    # function to extract the sentences from the dictionary of samples
    def extract_sentences(self, file_output):
        self.sentences = list()  # creating a list to store the instances in the dataset
        for instance in file_output:
            processed = self.text_preprocess(
                instance)  # process every sample (sentence) with the text_preprocess function
            labels = 'UNKNOWN'  # this is needed to make the system able to store the sentences without a ground truth (for predictions)
            if 'labels' in instance:  # but if there is a ground truth we take it
                labels = processed['labels']
            self.sentences.append((processed["text"], processed["pos"], labels,
                                   id))  # append a tuple (sentence,pos,labels,id) which are all the informations we need
        if not self.test: random.Random(42).shuffle(
            self.sentences)  # for the training phase, shuffle data to avoid bias relative to data order

    # function to convert the pos extracted by nltk to the pos required by the very same library for lemmatization
    # I also use it to give pos='' to punctuation
    def get_standard(self, pos):
        if pos[0] == 'V': return wordnet.VERB
        if pos[0] == 'R': return wordnet.ADV
        if pos[0] == 'N': return wordnet.NOUN
        if pos[0] == 'J': return wordnet.ADJ
        return ''

    # function to return the number of instances contained in the dataset
    def __len__(self):
        return len(self.sentences)

    # function to get the i-th instance contained in the dataset
    def __getitem__(self, idx):
        return self.sentences[idx]

    # custom dataloader which incorporates the collate function
    def dataloader(self, batch_size):
        return DataLoader(self, batch_size=batch_size, collate_fn=partial(self.collate))

    # function to map each lemma,pos in a sentence to their indexes
    def sent2idx(self, sent, word2idx):
        return torch.tensor([word2idx[word] for word in sent])

    # custom collate function, used to create the batches to give as input to the nn
    # it's needed because we are dealing with sentences of variable length and we need padding
    # to be sure that each sentence in a batch has the same length, which is necessary
    def collate(self, data):
        X = [self.sent2idx(instance["words"], self.word2idx) for instance in data]  # extracting the input sentence
        X_len = torch.tensor([x.size(0) for x in X], dtype=torch.long).to(
            device)  # extracting the length for each sentence
        X_pos = [self.sent2idx(instance["pos"], self.pos2idx) for instance in
                 data]  # extracting pos tags for each sentence
        y = [self.sent2idx(instance["roles"], self.roles2idx) for instance in data]  # extracting labels for each sentence
        ids = [instance["id"] for instance in data]  # extracting the sentences' ids
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=1).to(
            device)  # padding all the sentences to the maximum length in the batch (forcefully max_len)
        X_pos = torch.nn.utils.rnn.pad_sequence(X_pos, batch_first=True, padding_value=1).to(
            device)  # padding all the pos tags
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.roles2idx[PAD_TOKEN]).to(
            device)  # padding all the labels
        return {
            "X": X,
            "X_len": X_len,
            "X_pos": X_pos,
            "y": y,
            "ids": ids
        }

    # function to convert the output ids to the corresponding labels
    def convert_output(self, output):
        converted = []
        for sentence in output:
            converted_sent = []
            for label in sentence:
                converted_sent.append(self.id2class[label.item()])
            converted.append(converted_sent)
        return converted


en_train_dataset = SentenceDataset(sentences_path=EN_TRAIN_PATH)
en_dev_dataset = SentenceDataset(sentences_path=EN_TRAIN_PATH)
# print(en_dataset[0])
number_words = len(en_train_dataset.word2idx)
number_pos = len(en_train_dataset.pos2idx)


# '''
class StudentModel(nn.Module):  # ,Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self, language: str,  # pos embedding vectors
                n_words,
                n_pos,
                input_dim=100,
                hidden1=128,  # dimension of the first hidden layer
                p=0.0,  # probability of dropout layer
                bidirectional=False,  # flag to decide if the LSTM must be bidirectional
                lstm_layers=1,  # layers of the LSTM
                num_classes=28):  # loss function
        super().__init__()
        self.embedding = nn.Embedding(n_words, 100)
        self.pos_embeddings = None #nn.Embedding(n_pos, 20)
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden1, dropout=p, num_layers=lstm_layers,
                            batch_first=True, bidirectional=bidirectional)
        hidden1 = hidden1 * 2 if bidirectional else hidden1  # computing the dimension of the linear layer based on if the LSTM is bidirectional or not
        self.lin1 = nn.Linear(hidden1, num_classes)
        self.dropout = nn.Dropout(p=p)
        self.num_classes = num_classes
        # load the specific model for the input language
        self.language = language
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=28)

    # forward method, automatically called when calling the instance
    # it takes the input tokens'indices, the labels'indices and the PoS tags'indices linked to input tokens
    def forward(self, X, y, X_pos=None):
        embeddings = self.embedding(X)  # expanding the words from indices to embedding vectors
        if self.pos_embeddings is not None:
            pos_embeddings = self.pos_embeddings(
                X_pos)  # in the case I'm using pos embeddings, I pass their indexes through their own embedding layer
            embeddings = torch.cat([embeddings, pos_embeddings],
                                   dim=-1)  # and then concatenate them to the corresponding words
        lstm_out = self.lstm(embeddings)[0]
        out = self.dropout(lstm_out)
        out = torch.relu(out)
        out = self.lin1(out)
        out = torch.softmax(out, dim=-1)
        return out

    def predict(self, sentence):
        """
        --> !!! STUDENT: implement here your predict function !!! <--

        Args:
            sentence: a dictionary that represents an input sentence, for example:
                - If you are doing argument identification + argument classification:
                    {
                        "words":
                            [  "In",  "any",  "event",  ",",  "Mr.",  "Englund",  "and",  "many",  "others",  "say",  "that",  "the",  "easy",  "gains",  "in",  "narrowing",  "the",  "trade",  "gap",  "have",  "already",  "been",  "made",  "."  ]
                        "lemmas":
                            ["in", "any", "event", ",", "mr.", "englund", "and", "many", "others", "say", "that", "the", "easy", "gain", "in", "narrow", "the", "trade", "gap", "have", "already", "be", "make",  "."],
                        "predicates":
                            ["_", "_", "_", "_", "_", "_", "_", "_", "_", "AFFIRM", "_", "_", "_", "_", "_", "REDUCE_DIMINISH", "_", "_", "_", "_", "_", "_", "MOUNT_ASSEMBLE_PRODUCE", "_" ],
                    },
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        "predicates":
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 ],
                    },
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "words": [...], # SAME AS BEFORE
                        "lemmas": [...], # SAME AS BEFORE
                        # NOTE: you do NOT have a "predicates" field here.
                    },

        Returns:
            A dictionary with your predictions:
                - If you are doing argument identification + argument classification:
                    {
                        "roles": list of lists, # A list of roles for each predicate in the sentence.
                    }
                - If you are doing predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list with your predicted predicate senses, one for each token in the input sentence.
                        "roles": dictionary of lists, # A list of roles for each pre-identified predicate (index) in the sentence.
                    }
                - If you are doing predicate identification + predicate disambiguation + argument identification + argument classification:
                    {
                        "predicates": list, # A list of predicate senses, one for each token in the sentence, null ("_") included.
                        "roles": dictionary of lists, # A list of roles for each predicate (index) you identify in the sentence.
                    }
        """
        pass


# trainer class
class Trainer():

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.metric = torchmetrics.F1Score(ignore_index=27).to(device)

    # train function, takes two dataloaders for trainset and devset in order to train on the trainset and
    # check at every epoch how the training is affecting performance on the dev set, to avoid overfitting
    # I use the patience mechanism to stop after a number of times that the accuracy on devset goes down
    def train(self, train_dataset, dev_dataset, patience, epochs=10):
        train_loader = train_dataset.dataloader(batch_size=16)  # instantiating the dataloaders
        dev_loader = dev_dataset.dataloader(batch_size=16)
        loss_history = [[], []]  # lists to save trainset and devset loss in order to plot the graph later
        f1_history = [[], []]  # lists to save trainset and devset accuracy in order to plot the graph later
        patience_counter = 0  # counter to implement patience
        for i in range(epochs):
            losses = []  # list to save the loss for each batch
            f1s = []
            for batch in train_loader:
                batch_x = batch["X"]  # separating first from second sentences
                batch_xlen = batch["X_len"]  # separating lengths of first and second sentences
                batch_x_pos = batch["X_pos"]
                labels = batch["y"]  # taking the ground truth
                ids = batch["ids"]
                self.optimizer.zero_grad()  # setting the gradients to zero for each batch
                logits = self.model(batch_x, labels, batch_x_pos)  # predict
                logits = logits.view(-1, logits.shape[-1])
                labels = labels.view(-1)
                loss = model.loss_fn(logits,labels)
                f1 = self.metric(logits, labels)
                f1s.append(f1)
                loss.backward()  # backpropagating the loss
                self.optimizer.step()  # adjusting the model parameters to the loss
                losses.append(loss.item())  # appending the losses to losses
            mean_f1 = sum(f1s) / len(f1s)  # computing the mean loss for each epoch
            f1_history[0].append(mean_f1)
            mean_loss = sum(losses) / len(losses)  # computing the mean loss for each epoch
            loss_history[0].append(mean_loss)  # appending the mean loss of each epoch to loss history
            metrics = {'mean_loss': mean_loss, 'f1': mean_f1}  # displaying results of the epoch
            print(f'Epoch {i}   values on training set are {metrics}')
            # the same exact process is repeated on the instances of the devset, without gradient backpropagation and optimization
            with torch.no_grad():
                losses = []  # list to save the loss for each batch
                f1s = []
                for batch in dev_loader:
                    batch_x = batch["X"]  # separating first from second sentences
                    batch_xlen = batch["X_len"]  # separating lengths of first and second sentences
                    batch_x_pos = batch["X_pos"]
                    labels = batch["y"]  # taking the ground truth
                    ids = batch["ids"]
                    logits = self.model(batch_x, labels, batch_x_pos)  # predict
                    logits = logits.view(-1, logits.shape[-1])
                    labels = labels.view(-1)
                    loss = model.loss_fn(logits, labels)
                    losses.append(loss.item())  # appending the losses to losses
                    f1 = self.metric(logits, labels)
                    f1s.append(f1)
            mean_f1 = sum(f1s) / len(f1s)  # computing the mean loss for each epoch
            f1_history[1].append(mean_f1)
            mean_loss = sum(losses) / len(losses)  # computing the mean loss for each epoch
            loss_history[1].append(mean_loss)  # appending the mean loss of each epoch to loss history
            metrics = {'mean_loss': mean_loss, 'f1': mean_f1}
            print(f'            final values on the dev set are {metrics}')
            # check for early stopping
            if len(f1_history[1]) > 1 and f1_history[1][-1] < f1_history[1][
                -2]:  # if the last f1 is lower than the previous
                patience_counter += 1  # increase patience_counter
                if patience <= patience_counter:
                    print('-----------------------------EARLY STOP--------------------------------------------')
                    break
                else:
                    print('------------------------------PATIENCE---------------------------------------------')

        return {
            'loss_history': loss_history,
            'f1_history': f1_history
        }


# '''
model = StudentModel(language="en",n_words=number_words,n_pos=number_pos,lstm_layers=5,bidirectional=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.000)  # instantiating the optimizer
trainer = Trainer(model=model, optimizer=optimizer)  # instantiating the trainer
histories = trainer.train(train_dataset=en_train_dataset,dev_dataset=en_dev_dataset,patience=0,epochs=100)    #training