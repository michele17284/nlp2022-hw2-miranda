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
#import seaborn as sn
#import pandas as pd
import copy

#seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#setting up nltk
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

#setting the embedding dimension
EMBEDDING_DIM=100
POS_EMBEDDING_DIM=10

#specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device("cpu")
print(device)
#setting unknown token to handle out of vocabulary words and padding token to pad sentences
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
EN_TRAIN_PATH = "./../../data/EN/train.json"
EN_DEV_PATH = "./../../data/EN/dev.json"
print(torch.version.cuda)


class SentenceDataset(Dataset):

    def __init__(self,vectors=None,word2idx=None,pos_vectors=None,pos2idx=None,class2id=None,sentences_path=None,sentences=None,lemmatization=False,
                 test=False):
        file_output = self.read_file(sentences_path) if sentences_path else self.read_sentences(sentences)
        #print(file_output)
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

    #little function to read and store a file given the path
    def read_file(self,path):
        sentences = list()
        with open(path) as file:
            json_file = json.load(file)
            for key in json_file:
                #count = json_file[key]["predicates"].count("_")
                #length = len(json_file[key]["predicates"])
                if json_file[key]["predicates"].count("_") != len(json_file[key]["predicates"]) and len(json_file[key]["roles"]) > 1:
                    for position in json_file[key]["roles"]:
                        roles = json_file[key]["roles"][position]
                        predicates = ["_" for i in range(len(roles))]

                        predicates[int(position)] = json_file[key]["predicates"][int(position)]
                        instance = copy.deepcopy(json_file[key])
                        instance["roles"] = roles
                        instance["predicates"] = predicates
                        sentences.append(instance)
                else:
                    instance = copy.deepcopy(json_file[key])
                    if json_file[key]["predicates"].count("_") == len(json_file[key]["predicates"]):
                        roles = ["_" for i in range(len(json_file[key]["predicates"]))]
                        instance["roles"] = roles
                    else:
                        k = list(instance["roles"].keys())[0]
                        instance["roles"] = instance["roles"][k]
                    sentences.append(instance)
        return sentences

    #little function to read and store a set of sentences given as a list of tokens
    def read_sentences(self,sentences):
        sents = list()
        for idx,line in enumerate(sentences):
            d = dict()
            d["id"] = idx
            d["text"] = line
            d["labels"] = ["O" for token in line]
            sents.append(d)
        return sents

    #function to extract the sentences from the dictionary of samples
    def extract_sentences(self,file_output):
        self.sentences = list()                 #creating a list to store the instances in the dataset
        for instance in file_output:
            processed = self.text_preprocess(instance)      #process every sample (sentence) with the text_preprocess function
            labels = 'UNKNOWN'   #this is needed to make the system able to store the sentences without a ground truth (for predictions)
            if 'labels' in instance: #but if there is a ground truth we take it
                labels = processed['labels']
            self.sentences.append((processed["text"],processed["pos"], labels, id))           #append a tuple (sentence,pos,labels,id) which are all the informations we need
        if not self.test: random.Random(42).shuffle(self.sentences)         #for the training phase, shuffle data to avoid bias relative to data order

    #function to convert the pos extracted by nltk to the pos required by the very same library for lemmatization
    #I also use it to give pos='' to punctuation
    def get_standard(self,pos):
        if pos[0] == 'V': return wordnet.VERB
        if pos[0] == 'R': return wordnet.ADV
        if pos[0] == 'N': return wordnet.NOUN
        if pos[0] == 'J': return wordnet.ADJ
        return ''

    #function for preprocessing, which includes pos tagging and (if specified) lemmatization
    def text_preprocess(self,sentence):
        tokens_n_pos = nltk.pos_tag(sentence["text"])
        standard_tokens = [(token,self.get_standard(pos)) for token,pos in tokens_n_pos]
        if self.w_lemmatization:            #choosing if applying lemmatization
            lemmatized = [(lemmatizer.lemmatize(token.lower(),pos),pos) if pos != '' else (lemmatizer.lemmatize(token.lower()),'') for token,pos in standard_tokens]
            sentence["text"] = [lemma for lemma,pos in lemmatized]
        sentence["pos"] = [pos for word,pos in standard_tokens]
        return sentence

    #function to return the number of instances contained in the dataset
    def __len__(self):
        return len(self.sentences)

    #function to get the i-th instance contained in the dataset
    def __getitem__(self, idx):
        return self.sentences[idx]

    #custom dataloader which incorporates the collate function
    def dataloader(self,batch_size):
        return DataLoader(self,batch_size=batch_size,collate_fn=partial(self.collate))

    #function to map each lemma,pos in a sentence to their indexes
    def sent2idx(self ,sent, word2idx):
        return torch.tensor([word2idx[word] for word in sent])

    #custom collate function, used to create the batches to give as input to the nn
    #it's needed because we are dealing with sentences of variable length and we need padding
    #to be sure that each sentence in a batch has the same length, which is necessary
    def collate(self, data):
        X = [self.sent2idx(instance[0], self.word2idx) for instance in data]                            #extracting the input sentence
        X_len = torch.tensor([x.size(0) for x in X], dtype=torch.long).to(device)                       #extracting the length for each sentence
        X_pos = [self.sent2idx(instance[1], self.pos2idx) for instance in data]                         #extracting pos tags for each sentence
        y = [self.sent2idx(instance[2], self.class2id) for instance in data]                            #extracting labels for each sentence
        ids = [instance[3] for instance in data]                                                        #extracting the sentences' ids
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=1).to(device)            #padding all the sentences to the maximum length in the batch (forcefully max_len)
        X_pos = torch.nn.utils.rnn.pad_sequence(X_pos, batch_first=True, padding_value=1).to(device)    #padding all the pos tags
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.class2id[PAD_TOKEN]).to(device)              #padding all the labels
        return X, X_len,X_pos,y, ids

    #function to convert the output ids to the corresponding labels
    def convert_output(self,output):
        converted = []
        for sentence in output:
            converted_sent = []
            for label in sentence:
                converted_sent.append(self.id2class[label.item()])
            converted.append(converted_sent)
        return converted


#dataset = SentenceDataset(sentences_path=EN_DEV_PATH)
