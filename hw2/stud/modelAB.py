
import json
import numpy as np
from typing import List, Tuple, Optional, Union
import random
import nltk
from torch import nn
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from string import punctuation
from functools import partial
from typing import Tuple, List, Any, Dict
import copy
import torchmetrics
import pytorch_lightning as pl
import torch
from transformers import BertTokenizer, BertModel
import os
# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging

# logging.basicConfig(level=logging.INFO)

# import matplotlib.pyplot as plt

nltk.download('averaged_perceptron_tagger')
# seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#
# stop_tokens = set(stopwords.words('english'))
# punc_tokens = set(punctuation)
# stop_tokens.update(punc_tokens)
lemmatizer = WordNetLemmatizer()

# setting the embedding dimension
EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 10

# specify the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
# setting unknown token to handle out of vocabulary words and padding token to pad sentences
UNK_TOKEN = '<unk>'
# PAD_TOKEN = '_' #???????????????
PAD_TOKEN = '<pad>'
BERT_PATH = "./model/bert-base-cased"
#BERT_PATH = "../../model/bert-base-cased"
print(torch.version.cuda)

SEMANTIC_ROLES = ["AGENT", "ASSET", "ATTRIBUTE", "BENEFICIARY", "CAUSE", "CO_AGENT", "CO_PATIENT", "CO_THEME",
                  "DESTINATION",
                  "EXPERIENCER", "EXTENT", "GOAL", "IDIOM", "INSTRUMENT", "LOCATION", "MATERIAL", "PATIENT", "PRODUCT",
                  "PURPOSE",
                  "RECIPIENT", "RESULT", "SOURCE", "STIMULUS", "THEME", "TIME", "TOPIC", "VALUE", "_"]

torch.set_printoptions(threshold=10_000)

#VERBATLAS_PATH = "./VerbAtlas/VerbAtlas"
VERBATLAS_PATH = "./hw2/stud/VerbAtlas/VerbAtlas"

import csv


def readVerbAtlas(verbAtlasPath):
    frame_info = os.path.join(verbAtlasPath, "VA_frame_info.tsv")
    frame_info = open(frame_info)
    frame_pas = os.path.join(verbAtlasPath, "VA_frame_pas.tsv")
    read_tsv = csv.reader(frame_info, delimiter="\t")
    id2frame = {}
    frame2id = {}
    for idx, row in enumerate(read_tsv):
        if idx != 0:
            id2frame[row[0]] = row[1].upper()
            frame2id[row[1].upper()] = row[0]
    return id2frame, frame2id






class PredicateDataset(Dataset):

    def __init__(self, verbAtlas_path, sentences_path=None, sentences=None,
                 test=False):
        self.test = test

        self.SEMANTIC_ROLES = ["<pad>", "AGENT", "ASSET", "ATTRIBUTE", "BENEFICIARY", "CAUSE", "CO-AGENT", "CO-PATIENT",
                               "CO-THEME", "DESTINATION",
                               "EXPERIENCER", "EXTENT", "GOAL", "IDIOM", "INSTRUMENT", "LOCATION", "MATERIAL",
                               "PATIENT", "PRODUCT", "PURPOSE",
                               "RECIPIENT", "RESULT", "SOURCE", "STIMULUS", "THEME", "TIME", "TOPIC", "VALUE", "_"]
        self.roles2idx = {role.lower(): idx for idx, role in enumerate(self.SEMANTIC_ROLES)}
        self.roles2idx["<pred>"] = 0
        id2frame, frame2id = readVerbAtlas(verbAtlas_path)
        print(len(frame2id))
        self.id2frame = id2frame
        self.frame2id = frame2id
        self.frame2idx = {frame: idx for idx, (id, frame) in enumerate(self.id2frame.items())}
        self.idx2frame = {idx: frame for frame, idx in self.frame2idx.items()}
        self.frame2idx["_"] = len(self.frame2idx)
        self.idx2frame[len(self.idx2frame)] = "_"
        self.frame2idx["<pad>"] = len(self.frame2idx)
        self.idx2frame[len(self.idx2frame)] = "<pad>"
        self.idx2roles = {idx: role.lower() for idx, role in enumerate(self.SEMANTIC_ROLES)}
        self.sentences = self.read_sentences(sentences_path=sentences_path, sentences_plain=sentences)
        self.bert_preprocess(self.sentences)

    # little function to read and store a file given the path
    def read_sentences(self, sentences_path, sentences_plain):
        sentences = list()
        if sentences_path:
            with open(sentences_path) as file:
                json_file = json.load(file)
        else:
            json_file = {0: sentences_plain}
        for key in json_file:
            json_file[key]["id"] = key
            instance = json_file[key]
            instance["attention_mask"] = [1] * len(instance["words"])
            if sentences_plain: instance["predicates"] = ["_"] * len(instance["words"])
            sentences.append(instance)
        return sentences


    def bert_preprocess(self, sentences):  # TODO tokenize with bert
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH,local_files_only=True)
        #tokenizer = BertTokenizer.from_pretrained("bert-base-cased", local_files_only=True)
        for sentence in sentences:
            text = "[CLS] " + " ".join(sentence["words"]) + " [SEP]"
            non_joined_text = ["[CLS]"] + sentence["words"] + ["[SEP]"]
            encoded = tokenizer.convert_tokens_to_ids(non_joined_text)
            segments_ids = [1] * len(non_joined_text)
            sentence["segment_ids"] = segments_ids
            sentence["encoded_words"] = encoded
            sentence["tokenized_predicates"] = ["<pad>"] + sentence["predicates"] + ["<pad>"]
            sentence["attention_mask"] = [0] + sentence["attention_mask"] + [0]
            # assert len(non_joined_text) == len(sentence["attention_mask"])
        return sentences

    def create_vocabulary(self, word_list):
        word2idx = dict()
        for i, word in enumerate(word_list):
            word2idx[word] = i
        return word2idx


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
    def dataloader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, collate_fn=partial(self.collate), shuffle=shuffle)

    # function to map each lemma,pos in a sentence to their indexes
    def sent2idx(self, sent, word2idx):
        out = [word2idx[word] for word in sent]
        return out

    # custom collate function, used to create the batches to give as input to the nn
    # it's needed because we are dealing with sentences of variable length and we need padding
    # to be sure that each sentence in a batch has the same length, which is necessary
    def collate(self, data):
        X = [torch.tensor(instance["encoded_words"]) for instance in data]  # extracting the input sentence
        X_len = torch.tensor([x.size(0) for x in X], dtype=torch.long).to(
            device)  # extracting the length for each sentence
        segment_ids = [torch.tensor(instance["segment_ids"]) for instance in data]
        attention_mask = [torch.tensor(instance["attention_mask"], dtype=torch.bool) for instance in data]
        y = [torch.tensor(self.sent2idx(instance["tokenized_predicates"], self.frame2idx)) for instance in
             data]  # extracting labels for each sentence
        ids = [instance["id"] for instance in data]  # extracting the sentences' ids
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=100).to(
            device)  # padding all the sentences to the maximum length in the batch (forcefully max_len)
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.roles2idx[PAD_TOKEN]).to(
            device)  # padding all the labels
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0).to(device)
        segment_ids = torch.nn.utils.rnn.pad_sequence(segment_ids, batch_first=True, padding_value=0).to(device)

        return X, X_len, segment_ids, y, attention_mask, ids

    # function to convert the output ids to the corresponding labels
    def convert_output(self, output):
        return self.sent2idx(output, self.idx2frame)


class PredicatesDataModule(pl.LightningDataModule):

    def __init__(
            self,
            verbAtlas_path: str,
            data_train_path: str,
            data_dev_path: str,
            data_test_path: str,
            batch_size: int
    ) -> None:
        super().__init__()
        self.data_train_path = data_train_path
        self.data_dev_path = data_dev_path
        self.data_test_path = data_test_path
        self.batch_size = batch_size
        self.verbAtlas_path = verbAtlas_path
        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = PredicateDataset(verbAtlas_path=self.verbAtlas_path,
                                                  sentences_path=self.data_train_path)
            self.validation_dataset = PredicateDataset(verbAtlas_path=self.verbAtlas_path,
                                                       sentences_path=self.data_dev_path)
        elif stage == 'test':
            self.test_dataset = PredicateDataset(verbAtlas_path=self.verbAtlas_path,
                                                 sentences_path=self.data_test_path)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.train_dataset.dataloader(batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.validation_dataset.dataloader(batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.test_dataset.dataloader(batch_size=self.batch_size, shuffle=False)


# '''


class AB_Model(
    pl.LightningModule):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self, language: str,  # pos embedding vectors
                 input_dim=768,
                 hidden1=768,  # dimension of the first hidden layer
                 hidden2=768,
                 p=0.0,  # probability of dropout layer
                 bidirectional=False,  # flag to decide if the LSTM must be bidirectional
                 lstm_layers=1,  # layers of the LSTM
                 ):  # loss function
        super().__init__()
        hidden1 = hidden1 * 2 if bidirectional else hidden1  # computing the dimension of the linear layer based on if the LSTM is bidirectional or not
        self.num_classes = 434
        self.hidden1 = hidden1
        self.classifier = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden2, self.num_classes)
        )
        print("NUM_CLASSES ", self.num_classes)
        # load the specific model for the input language
        self.language = language
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=self.num_classes - 1)
        self.f1 = torchmetrics.classification.F1Score(num_classes=self.num_classes, ignore_index=self.num_classes - 1).to(device)
        self.bert = BertModel.from_pretrained(BERT_PATH,local_files_only=True, output_hidden_states=True,
                                              is_decoder=True,
                                              add_cross_attention=True)

        self.bert.eval()

    # forward method, automatically called when calling the instance
    # it takes the input tokens'indices, the labels'indices and the PoS tags'indices linked to input tokens
    def forward(self, X, X_len, segment_ids, y, attention_mask, ids):
        outputs = self.bert(input_ids=X, token_type_ids=segment_ids, attention_mask=attention_mask)
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.reshape(token_embeddings, shape=(token_embeddings.size(0), token_embeddings.size(1) *
                                                                  token_embeddings.size(2), token_embeddings.size(3)))
        token_embeddings = token_embeddings.permute(1, 0, 2)
        token_vecs_cat = []
        for token in token_embeddings:
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_cat.append(sum_vec)

        token_vecs_cat = torch.stack(token_vecs_cat, dim=0)
        # lstm_out = self.lstm(token_vecs_cat)[0]
        out = self.classifier(token_vecs_cat)

        out = out.view(-1, out.shape[-1])
        logits = out
        y = y.view(-1)
        pred = torch.softmax(out, dim=-1)
        flat_preds = torch.argmax(pred, dim=1)
        flat_preds = flat_preds.view(pred.size(0), 1)
        result = {'logits': logits, 'pred': pred, "labels": y,
                  "flat_pred": flat_preds}
        # compute loss
        if y is not None:
            loss = self.loss_fn(logits, y)
            result['loss'] = loss

        return result

    def training_step(
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        forward_output = self.forward(*batch)
        self.f1(forward_output['flat_pred'], forward_output["labels"])
        self.log('train_f1', self.f1, prog_bar=True)
        self.log('train_loss', forward_output['loss'], prog_bar=True)
        return forward_output['loss']

    def validation_step(
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int
    ):
        forward_output = self.forward(*batch)
        self.f1(forward_output['flat_pred'], forward_output["labels"])
        self.log('val_f1', self.f1, prog_bar=True)
        self.log('val_loss', forward_output['loss'], prog_bar=True)

    def test_step(
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int
    ):
        forward_output = self.forward(*batch)
        self.f1(forward_output['flat_pred'], forward_output["labels"])
        self.log('test_f1', self.f1)

    def loss(self, pred, y):
        return self.loss_fn(pred, y)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001, weight_decay=0.0000)  # instantiating the optimizer
        return optimizer

    def predict(self, sentence):
        sentences = PredicateDataset(verbAtlas_path=VERBATLAS_PATH, sentences=sentence)
        batches = sentences.dataloader(shuffle=False, batch_size=10)
        out = []
        for batch in batches:
            X, X_len, segment_ids, y, attention_mask, ids = batch
            output = self.forward(X, X_len, segment_ids, y, attention_mask, ids)
            preds = output["flat_pred"].view(X.size(0), -1)
            out = sentences.convert_output(torch.flatten(preds).tolist())
        return out[1:-1]


