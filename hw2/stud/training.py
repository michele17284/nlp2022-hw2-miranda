import json
import string

import numpy as np
from typing import List, Tuple, Optional, Union
from typing import List, Dict
import torch
import random
import csv
import matplotlib.pyplot as plt
import nltk
import transformers
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
import pytorch_lightning as pl
import torch
from transformers import BertTokenizer, BertModel

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt

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
#device = torch.device("cpu")
print(device)
# setting unknown token to handle out of vocabulary words and padding token to pad sentences
UNK_TOKEN = '<unk>'
#PAD_TOKEN = '_' #???????????????
PAD_TOKEN = '<pad>'
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
        self.bert_preprocess(self.sentences)
        #word_list = self.read_sentences(features)
        #self.word2idx = self.create_vocabulary(word_list)
        #features = ["pos"]
        #pos_list = self.read_sentences(features)
        #self.pos2idx = self.create_vocabulary(pos_list)
        self.SEMANTIC_ROLES = ["<pad>","AGENT", "ASSET", "ATTRIBUTE", "BENEFICIARY", "CAUSE", "CO-AGENT", "CO-PATIENT",
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
        sentences_len = dict()
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
                        instance["predicate_position"] = position
                        instance["predicate_position"] = [0]*len(roles)
                        instance["predicate_position"][int(position)] = 1
                        # instance["roles_bin"] = [1 if role != "_" else 0 for role in roles]
                        instance["predicates"] = predicates
                        instance["attention_mask"] = [1]*len(roles)
                        instance["around_predicate"] = [0]*len(roles)
                        start = max(0,int(position)-10)
                        stop = min(len(roles),int(position)+10)
                        for i in range(start,stop): instance["around_predicate"][i] = 1
                        instance["around_predicate"][max([int(position)-10,0]):min([int(position)+10,len(roles)])] = [1]*21
                        sentences.append(self.text_preprocess(instance))
                else:
                    instance = copy.deepcopy(json_file[key])

                    if json_file[key]["predicates"].count("_") == len(json_file[key]["predicates"]):
                        instance["roles"] = ["_" for i in range(len(json_file[key]["predicates"]))]
                        instance["predicate_position"] = [1] * len(instance["roles"])
                        instance["around_predicate"] = [0] * len(instance["roles"])
                        # instance["roles_bin"] = [1 if role != "_" else 0 for role in roles]
                    else:
                        k = list(instance["roles"].keys())[0]
                        instance["roles"] = instance["roles"][k]
                        instance["predicate_position"] = [0] * len(instance["roles"])
                        instance["predicate_position"][int(k)] = 1
                        instance["around_predicate"] = [0] * len(instance["roles"])
                        start = max(0, int(position) - 10)
                        stop = min(len(instance["roles"]), int(position) + 10)
                        for i in range(start, stop): instance["around_predicate"][i] = 1
                    instance["attention_mask"] = [1] * len(instance["roles"])

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

    def bert_preprocess(self,sentences):    #TODO tokenize with bert
        tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        for sentence in sentences:
            text = "[CLS] "+ " ".join(sentence["words"])+" [SEP]"
            non_joined_text = ["[CLS]"]+sentence["words"]+["[SEP]"]
            roles = ["<pad>"]+sentence["roles"]+["<pad>"]
            tokenized = tokenizer.tokenize(text)
            x_index, y_index = 1,1
            tokenized_roles = ['<pad>']
            '''
            while x_index < len(tokenized)-1:
                if tokenized[x_index].startswith('#'):
                    tokenized_roles.append('_')
                    x_index += 1
                elif len(tokenized[x_index]) > 1 and tokenized[x_index][0] in string.punctuation :
                    temp = tokenized[x_index]
                    tokenized[x_index] = temp[0]
                    tokenized.insert(x_index,temp[1:])
                else:
                    tokenized_roles.append(roles[y_index])
                    x_index += 1
                    y_index += 1
            '''
            tokenized_roles.append('<pad>')
            encoded = tokenizer.convert_tokens_to_ids(non_joined_text)
            segments_ids = [1] * len(non_joined_text)
            sentence["segment_ids"] = segments_ids
            sentence["encoded_words"] = encoded
            sentence["tokenized_roles"] = ["<pad>"]+sentence["roles"]+["<pad>"]
            sentence["predicate_position"] = [0] + sentence["predicate_position"] + [0]
            sentence["attention_mask"] = [0]+sentence["attention_mask"]+[0]
            sentence["around_predicate"] = [0]+sentence["around_predicate"]+[0]
        return sentences

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
    def dataloader(self, batch_size, shuffle=False):
        return DataLoader(self, batch_size=batch_size, collate_fn=partial(self.collate))

    # function to map each lemma,pos in a sentence to their indexes
    def sent2idx(self, sent, word2idx):
        return torch.tensor([word2idx[word] for word in sent])

    # custom collate function, used to create the batches to give as input to the nn
    # it's needed because we are dealing with sentences of variable length and we need padding
    # to be sure that each sentence in a batch has the same length, which is necessary
    def collate(self, data):
        X = [torch.tensor(instance["encoded_words"]) for instance in data]  # extracting the input sentence
        X_len = torch.tensor([x.size(0) for x in X], dtype=torch.long).to(
            device)  # extracting the length for each sentence
        #X_pos = [self.sent2idx(instance["pos"], self.pos2idx) for instance in data]  # extracting pos tags for each sentence
        segment_ids = [torch.tensor(instance["segment_ids"]) for instance in data]
        predicate_position = [torch.tensor(instance["predicate_position"]) for instance in data]
        attention_mask = [torch.tensor(instance["attention_mask"],dtype=torch.bool) for instance in data]
        y = [self.sent2idx(instance["tokenized_roles"], self.roles2idx) for instance in data]  # extracting labels for each sentence
        around_predicate = [torch.tensor(instance["around_predicate"],dtype=torch.bool) for instance in data]
        ids = [instance["id"] for instance in data]  # extracting the sentences' ids
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=100).to(device)  # padding all the sentences to the maximum length in the batch (forcefully max_len)
        #X_pos = torch.nn.utils.rnn.pad_sequence(X_pos, batch_first=True, padding_value=1).to(device)  # padding all the pos tags
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.roles2idx[PAD_TOKEN]).to(device)  # padding all the labels
        predicate_position = torch.nn.utils.rnn.pad_sequence(predicate_position, batch_first=True, padding_value=100).to(device)  # padding all the labels
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0).to(device)
        around_predicate = torch.nn.utils.rnn.pad_sequence(around_predicate, batch_first=True, padding_value=0).to(device)
        segment_ids = torch.nn.utils.rnn.pad_sequence(segment_ids, batch_first=True, padding_value=0).to(device)

        return X, X_len, segment_ids, y, predicate_position,attention_mask, around_predicate,ids


    # function to convert the output ids to the corresponding labels
    def convert_output(self, output):
        converted = []
        for sentence in output:
            converted_sent = []
            for label in sentence:
                converted_sent.append(self.id2class[label.item()])
            converted.append(converted_sent)
        return converted

class SentencesDataModule(pl.LightningDataModule):

    def __init__(
        self,
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

        self.train_dataset = None
        self.validation_dataset = None
        self.test_dataset = None

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == 'fit':
            self.train_dataset = SentenceDataset(self.data_train_path)
            self.validation_dataset = SentenceDataset(self.data_dev_path)
        elif stage == 'test':
            self.test_dataset = SentenceDataset(self.data_test_path)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.train_dataset.dataloader(batch_size=self.batch_size, shuffle=False)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.validation_dataset.dataloader(batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.test_dataset.dataloader(batch_size=self.batch_size, shuffle=False)



en_train_dataset = SentenceDataset(sentences_path=EN_TRAIN_PATH)
en_dev_dataset = SentenceDataset(sentences_path=EN_TRAIN_PATH)
# print(en_dataset[0])
#number_words = len(en_train_dataset.word2idx)
#number_pos = len(en_train_dataset.pos2idx)



# '''


class StudentModel(pl.LightningModule):                                 #TODO this is now a model for role classification, let's make it identification + classification

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
                 num_classes=29):  # loss function
        super().__init__()
        #self.embedding = nn.Embedding(n_words, input_dim)
        #self.pos_embeddings = None  # nn.Embedding(n_pos, 20)
        #self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden1, dropout=p, num_layers=lstm_layers,
        #                    batch_first=True, bidirectional=bidirectional)
        hidden1 = hidden1 * 2 if bidirectional else hidden1  # computing the dimension of the linear layer based on if the LSTM is bidirectional or not
        self.lin1 = nn.Linear(hidden1, hidden2)
        self.lin2 = nn.Linear(hidden2, num_classes)
        self.dropout = nn.Dropout(p=p)
        self.num_classes = num_classes
        self.hidden1 = hidden1
        # load the specific model for the input language
        self.language = language
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.f1 = torchmetrics.classification.F1Score(num_classes=num_classes).to(device)
        self.bert = BertModel.from_pretrained("bert-base-cased",output_hidden_states=True, is_decoder=True,
                                              add_cross_attention=True)
        #self.bert.eval()
    # forward method, automatically called when calling the instance
    # it takes the input tokens'indices, the labels'indices and the PoS tags'indices linked to input tokens
    def forward(self, X,X_len,segment_ids,y,predicate_position,attention_mask,around_predicate, ids):           #TODO highligt the predicate
        outputs = self.bert(input_ids=X,token_type_ids=segment_ids,attention_mask=attention_mask)
        hidden_states = outputs[2]
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.reshape(token_embeddings,shape=(token_embeddings.size(0),token_embeddings.size(1)*
                                                                 token_embeddings.size(2),token_embeddings.size(3)))
        token_embeddings = token_embeddings.permute(1, 0, 2)
        token_vecs_cat = []
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor
            '''
            # Concatenate the vectors (that is, append them together) from the last
            # four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)

            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)

            '''
            # Sum the vectors from the last four layers.
            sum_vec = torch.sum(token[-4:], dim=0)
            
            # Use `sum_vec` to represent `token`.
            token_vecs_cat.append(sum_vec)
            #'''
        #embeddings = self.embedding(X)  # expanding the words from indices to embedding vectors
        '''
        if self.pos_embeddings is not None:
            pos_embeddings = self.pos_embeddings(
                X_pos)  # in the case I'm using pos embeddings, I pass their indexes through their own embedding layer
            embeddings = torch.cat([embeddings, pos_embeddings],
                                   dim=-1)  # and then concatenate them to the corresponding words
        '''
        token_vecs_cat = torch.stack(token_vecs_cat, dim=0)
        #lstm_out = self.lstm(token_vecs_cat)[0]
        #out = self.dropout(token_vecs_cat)
        #out = torch.relu(out)
        out = self.lin1(token_vecs_cat)
        out = self.dropout(out)
        out = torch.relu(out)
        out = self.lin2(out)
        out = out.view(-1, out.shape[-1])
        logits = out
        y = y.view(-1)
        pred = torch.softmax(out, dim=-1)
        '''
        masked_around_y = torch.masked_select(y,attention_mask.view(-1))
        mask = torch.tensor([[True if element else False for _ in range(self.hidden1)] for element in attention_mask.view(-1)],dtype=torch.bool).to(device)
        masked_around_pred = torch.masked_select(pred,mask).view(-1,pred.size(1))
        masked_around_logits = torch.masked_select(logits,mask).view(-1,logits.size(1))

        mask_pad_y = (masked_around_y != 0)
        mask = torch.tensor(
            [[True if element else False for _ in range(self.hidden1)] for element in mask_pad_y],
            dtype=torch.bool).to(device)
        masked_y = torch.masked_select(masked_around_y, mask_pad_y)
        masked_pred = torch.masked_select(masked_around_pred,mask).view(-1,pred.size(1))
        masked_logits = torch.masked_select(masked_around_logits,mask).view(-1,logits.size(1))

        mask_y = (masked_y != 28)
        mask = torch.tensor(
            [[True if element else False for _ in range(self.hidden1)] for element in mask_y],
            dtype=torch.bool).to(device)
        masked_y = torch.masked_select(masked_y,mask_y)
        masked_pred = torch.masked_select(masked_pred,mask).view(-1,pred.size(1))
        masked_logits = torch.masked_select(masked_logits,mask).view(-1,logits.size(1))
        '''
        pps = torch.argmax(pred,dim=1)
        #print(pps)
        #print(y)
        masked_around_y = y[attention_mask.view(-1)]
        masked_around_pred = pred[attention_mask.view(-1)]
        masked_around_logits = logits[attention_mask.view(-1)]

        mask = (masked_around_y != 0) & (masked_around_y != 28)
        masked_y = masked_around_y[mask]
        masked_pred = masked_around_pred[mask]
        masked_logits = masked_around_logits[mask]

        flat_preds = torch.argmax(masked_pred,dim=1)
        #print(masked_pred.size(),pred, "PREDS")
        #print(flat_preds.size(),flat_preds, "ARGMAX")
        flat_preds = flat_preds.view(flat_preds.size(0),1)
        #print(flat_preds.shape, flat_preds, "ARGMAX")
        #print(masked_y.size(),masked_y,"Y")

        #for x in flat_preds:
        #    print(x)
        #for x in masked_y:
        #    print(x)
        #0/0
        result = {'logits': masked_logits, 'pred': masked_pred, "labels":masked_y, "flat_pred":flat_preds}

        # compute loss
        if y is not None:
            # while mathematically the CrossEntropyLoss takes as input the probability distributions,
            # torch optimizes its computation internally and takes as input the logits instead
            loss = self.loss_fn(masked_logits, masked_y)
            result['loss'] = loss

        return result

    def training_step(                                              #TODO understand why you don't get epoch-wise train F1
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        forward_output = self.forward(*batch)
        self.f1(forward_output['flat_pred'], forward_output["labels"])

        self.log('train_f1', self.f1, prog_bar=True, on_epoch=True)
        self.log('train_loss', forward_output['loss'], prog_bar=True, on_epoch=True)
        return forward_output['loss']

    def validation_step(
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int
    ):
        forward_output = self.forward(*batch)
        self.f1(forward_output['flat_pred'], forward_output["labels"])

        self.log('val_f1', self.f1, prog_bar=True, on_epoch=True)
        self.log('val_loss', forward_output['loss'], prog_bar=True, on_epoch=True)

    def test_step(
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int
    ):
        forward_output = self.forward(*batch)
        self.f1(forward_output['flat_pred'], forward_output["labels"])
        self.log('test_f1', self.f1, prog_bar=True)

    def loss(self, pred, y):
        return self.loss_fn(pred, y)

    def configure_optimizers(self):
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.0)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001,weight_decay=0.000)  # instantiating the optimizer
        return optimizer


early_stopping = pl.callbacks.EarlyStopping(
    monitor='val_f1',  # the value that will be evaluated to activate the early stopping of the model.
    patience=10,  # the number of consecutive attempts that the model has to raise (or lower depending on the metric used) to raise the "monitor" value.
    verbose=True,  # whether to log or not information in the console.
    mode='max', # wheter we want to maximize (max) or minimize the "monitor" value.
)

check_point_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_f1',  # the value that we want to use for model selection.
    verbose=True,  # whether to log or not information in the console.
    save_top_k=3,  # the number of checkpoints we want to store.
    mode='max',  # wheter we want to maximize (max) or minimize the "monitor" value.
    filename='{epoch}-{val_f1:.4f}'  # the prefix on the checkpoint values. Metrics store by the trainer can be used to dynamically change the name.
)

sentences_dm = SentencesDataModule(
    data_train_path=EN_DEV_PATH,
    data_dev_path=EN_DEV_PATH,
    data_test_path=EN_DEV_PATH,
    batch_size=32
)

classifier = StudentModel(language="en",hidden1=768,lstm_layers=1,bidirectional=False,p=0.0)


# the PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=100,  # maximum number of epochs.
    gpus=1,  # the number of gpus we have at our disposal.
    callbacks=[early_stopping, check_point_callback]  # the callback we want our trainer to use.
)

# and finally we can let the "trainer" fit the amazon reviews classifier.
trainer.fit(model=classifier, datamodule=sentences_dm)
'''
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
'''
model = StudentModel(language="en",n_words=number_words,n_pos=number_pos,lstm_layers=5,bidirectional=True).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=0.000)  # instantiating the optimizer
trainer = Trainer(model=model, optimizer=optimizer)  # instantiating the trainer
histories = trainer.train(train_dataset=en_train_dataset,dev_dataset=en_dev_dataset,patience=0,epochs=100)    #training
#'''