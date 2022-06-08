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

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
#logging.basicConfig(level=logging.INFO)

import matplotlib.pyplot as plt
nltk.download('averaged_perceptron_tagger')
# seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#
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
BERT_PATH = "./../../model/bert-base-cased"
print(torch.version.cuda)

SEMANTIC_ROLES = ["AGENT", "ASSET", "ATTRIBUTE", "BENEFICIARY", "CAUSE", "CO_AGENT", "CO_PATIENT", "CO_THEME",
                  "DESTINATION",
                  "EXPERIENCER", "EXTENT", "GOAL", "IDIOM", "INSTRUMENT", "LOCATION", "MATERIAL", "PATIENT", "PRODUCT",
                  "PURPOSE",
                  "RECIPIENT", "RESULT", "SOURCE", "STIMULUS", "THEME", "TIME", "TOPIC", "VALUE","_"]


def evaluate_argument_identification(labels, predictions, null_tag=28):
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in labels:
        gold = labels[sentence_id]["roles"]
        pred = predictions[sentence_id]["roles"]
        predicate_indices = set(gold.keys()).union(pred.keys())
        for idx in predicate_indices:
            if idx in gold and idx not in pred:
                false_negatives += sum(1 for role in gold[idx] if role != null_tag)
            elif idx in pred and idx not in gold:
                false_positives += sum(1 for role in pred[idx] if role != null_tag)
            else:  # idx in both gold and pred
                for r_g, r_p in zip(gold[idx], pred[idx]):
                    if r_g != null_tag and r_p != null_tag:
                        true_positives += 1
                    elif r_g != null_tag and r_p == null_tag:
                        false_negatives += 1
                    elif r_g == null_tag and r_p != null_tag:
                        false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_argument_classification(labels, predictions, null_tag=28):
    true_positives, false_positives, false_negatives = 0, 0, 0
    for sentence_id in labels:
        gold = labels[sentence_id]["roles"]
        pred = predictions[sentence_id]["roles"]
        predicate_indices = set(gold.keys()).union(pred.keys())

        for idx in predicate_indices:
            if idx in gold and idx not in pred:
                false_negatives += sum(1 for role in gold[idx] if role != null_tag)
            elif idx in pred and idx not in gold:
                false_positives += sum(1 for role in pred[idx] if role != null_tag)
            else:  # idx in both gold and pred
                for r_g, r_p in zip(gold[idx], pred[idx]):
                    if r_g != null_tag and r_p != null_tag:
                        if r_g == r_p:
                            true_positives += 1
                        else:
                            false_positives += 1
                            false_negatives += 1
                    elif r_g != null_tag and r_p == null_tag:
                        false_negatives += 1
                    elif r_g == null_tag and r_p != null_tag:
                        false_positives += 1

    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / (true_positives + false_negatives)
    f1 = 2 * (precision * recall) / (precision + recall)
    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }

class SentenceDataset(Dataset):

    def __init__(self, sentences_path=None, sentences=None, lemmatization=False,
                 test=False):
        self.test = test
        self.sentences = self.read_sentences(sentences_path=sentences_path,sentences_plain=sentences)

        self.bert_preprocess(self.sentences)
        self.SEMANTIC_ROLES = ["<pad>","AGENT", "ASSET", "ATTRIBUTE", "BENEFICIARY", "CAUSE", "CO-AGENT", "CO-PATIENT",
                               "CO-THEME", "DESTINATION",
                               "EXPERIENCER", "EXTENT", "GOAL", "IDIOM", "INSTRUMENT", "LOCATION", "MATERIAL",
                               "PATIENT", "PRODUCT", "PURPOSE",
                               "RECIPIENT", "RESULT", "SOURCE", "STIMULUS", "THEME", "TIME", "TOPIC", "VALUE","_"]
        self.roles2idx = {role.lower(): idx for idx,role in enumerate(self.SEMANTIC_ROLES)}
        self.roles2idx["<pred>"] = 0

        self.idx2roles = {idx: role.lower() for idx,role in enumerate(self.SEMANTIC_ROLES)}

    # little function to read and store a file given the path
    def read_sentences(self, sentences_path, sentences_plain):
        sentences = list()
        sentences_len = dict()
        if sentences_path:
            with open(sentences_path) as file:
                json_file = json.load(file)
        else:
            json_file = {0:sentences_plain}
        for key in json_file:
            # count = json_file[key]["predicates"].count("_")
            # length = len(json_file[key]["predicates"])
            json_file[key]["id"] = key
            if self.test:
                json_file[key]["roles"] = {}
                for idx,predicate in enumerate(json_file[key]["predicates"]):
                    if predicate != "_":
                        json_file[key]["roles"][idx] = ["_"]*(idx)+["_"]*(len(json_file[key]["predicates"])-idx)  #TODO find a solution for this
                if not json_file[key]["roles"]:
                    json_file[key]["roles"] = ["_" for i in range(len(json_file[key]["predicates"]))]
            if json_file[key]["predicates"].count("_") != len(json_file[key]["predicates"]):# and len(json_file[key]["roles"]) > 1:
                for position in json_file[key]["roles"]:
                    instance = copy.deepcopy(json_file[key])
                    roles = json_file[key]["roles"][position]
                    predicates = ["_" for i in range(len(roles))]
                    predicates[int(position)] = json_file[key]["predicates"][int(position)]
                    predicates.insert(int(position),"<pred>")
                    predicates.insert(int(position)+2,"<pred>")
                    roles.insert(int(position), "<pred>")
                    roles.insert(int(position) + 2, "<pred>")
                    instance["roles"] = roles
                    instance["predicate_position"] = int(position)
                    instance["predicates"] = predicates
                    attention_mask = [1]*len(roles)
                    attention_mask[int(position)] = 0
                    attention_mask[int(position)+2] = 0
                    instance["attention_mask"] = attention_mask


                    instance["around_predicate"] = [0]*len(roles)
                    around_number = 10
                    start = max(0, int(position) - 4)
                    stop = min(len(roles), int(position) + 6)
                    for i in range(start, stop): instance["around_predicate"][i] = 1
                    count = instance["around_predicate"].count(1)

                    if count < around_number:
                        if start == 0:
                            for i in range(stop + (around_number - count)):
                                instance["around_predicate"][i] = 1
                        elif stop == len(roles):
                            for i in range(len(roles) - around_number, len(roles)):
                                instance["around_predicate"][i] = 1
                    #instance["around_predicate"][max([int(position)-10,0]):min([int(position)+10,len(roles)])] = [1]*20
                    sentences.append(self.text_preprocess(instance))
            else:
                instance = copy.deepcopy(json_file[key])

                #if json_file[key]["predicates"].count("_") == len(json_file[key]["predicates"]):
                instance["roles"] = ["_" for i in range(len(json_file[key]["predicates"]))]
                instance["predicate_position"] = -1
                instance["around_predicate"] = [0] * len(instance["roles"])
                # instance["roles_bin"] = [1 if role != "_" else 0 for role in roles]
                '''
                else:
                    k = list(instance["roles"].keys())[0]
                    instance["roles"] = instance["roles"][k]
                    instance["predicate_position"] = int(k)
                    instance["around_predicate"] = [0] * len(instance["roles"])
                    start = max(0, int(k) - 3)
                    stop = min(len(instance["roles"]), int(k) + 3)
                    for i in range(start, stop): instance["around_predicate"][i] = 1
                
                '''
                instance["attention_mask"] = [1] * len(instance["roles"])
                sentences.append(self.text_preprocess(instance))
        return sentences



    # function for preprocessing, which includes pos tagging and (if specified) lemmatization
    def text_preprocess(self, sentence):
        tokens_n_pos = nltk.pos_tag(sentence["lemmas"])
        # standard_tokens = [(token,self.get_standard(pos)) for token,pos in tokens_n_pos]
        sentence["pos"] = [pos for word, pos in tokens_n_pos]
        return sentence

    def bert_preprocess(self,sentences):    #TODO tokenize with bert
        #tokenizer = BertTokenizer.from_pretrained("bert-base-cased",local_files_only=True)
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH, local_files_only=True)
        for sentence in sentences:
            text = "[CLS] "+ " ".join(sentence["words"])+" [SEP]"
            non_joined_text = ["[CLS]"]+sentence["words"]+["[SEP]"]
            if sentence["predicate_position"] != -1:
                non_joined_text.insert(sentence["predicate_position"], "<pred>")
                non_joined_text.insert(sentence["predicate_position"] + 2, "<pred>")
            encoded = tokenizer.convert_tokens_to_ids(non_joined_text)
            segments_ids = [1] * len(non_joined_text)
            sentence["segment_ids"] = segments_ids
            sentence["encoded_words"] = encoded
            sentence["tokenized_roles"] = ["<pad>"]+sentence["roles"]+["<pad>"]
            sentence["attention_mask"] = [0]+sentence["attention_mask"]+[0]
            sentence["around_predicate"] = [0]+sentence["around_predicate"]+[0]
            #assert len(non_joined_text) == len(sentence["attention_mask"])
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
        return DataLoader(self, batch_size=batch_size, collate_fn=partial(self.collate), shuffle=shuffle)

    # function to map each lemma,pos in a sentence to their indexes
    def sent2idx(self, sent, word2idx,is_tensor=False):
        return [word2idx[word] for word in sent]

    # custom collate function, used to create the batches to give as input to the nn
    # it's needed because we are dealing with sentences of variable length and we need padding
    # to be sure that each sentence in a batch has the same length, which is necessary
    def collate(self, data):
        X = [torch.tensor(instance["encoded_words"]) for instance in data]  # extracting the input sentence
        X_len = torch.tensor([x.size(0) for x in X], dtype=torch.long).to(
            device)  # extracting the length for each sentence
        #X_pos = [self.sent2idx(instance["pos"], self.pos2idx) for instance in data]  # extracting pos tags for each sentence
        segment_ids = [torch.tensor(instance["segment_ids"]) for instance in data]
        predicate_position = torch.tensor([torch.tensor(instance["predicate_position"]) for instance in data],device=device)
        attention_mask = [torch.tensor(instance["attention_mask"],dtype=torch.bool) for instance in data]
        y = [torch.tensor(self.sent2idx(instance["tokenized_roles"], self.roles2idx)) for instance in data]  # extracting labels for each sentence
        around_predicate = [torch.tensor(instance["around_predicate"],dtype=torch.bool) for instance in data]
        ids = [instance["id"] for instance in data]  # extracting the sentences' ids
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=100).to(device)  # padding all the sentences to the maximum length in the batch (forcefully max_len)
        #X_pos = torch.nn.utils.rnn.pad_sequence(X_pos, batch_first=True, padding_value=1).to(device)  # padding all the pos tags
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.roles2idx[PAD_TOKEN]).to(device)  # padding all the labels
        #predicate_position = torch.nn.utils.rnn.pad_sequence(predicate_position, batch_first=True, padding_value=100).to(device)  # padding all the labels
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0).to(device)
        around_predicate = torch.nn.utils.rnn.pad_sequence(around_predicate, batch_first=True, padding_value=0).to(device)
        segment_ids = torch.nn.utils.rnn.pad_sequence(segment_ids, batch_first=True, padding_value=0).to(device)

        return X, X_len, segment_ids, y, predicate_position,attention_mask, around_predicate,ids


    # function to convert the output ids to the corresponding labels
    def convert_output(self, output,predicate_position):
        converted = {
            "roles":{}
        }
        for idx,position in enumerate(predicate_position):
            converted["roles"][position] = self.sent2idx(output[idx],self.idx2roles)
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
        self.num_classes = num_classes
        self.hidden1 = hidden1
        self.classifier = nn.Sequential(
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(hidden2, num_classes)
        )

        # load the specific model for the input language
        self.language = language
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=0)
        self.f1 = torchmetrics.classification.F1Score(num_classes=num_classes,ignore_index=0).to(device)
        self.bert = BertModel.from_pretrained(BERT_PATH,local_files_only=True,output_hidden_states=True, is_decoder=True,
                                              add_cross_attention=True)
        self.bert.eval()
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
        out = self.classifier(token_vecs_cat)

        out = out.view(-1, out.shape[-1])
        logits = out
        y = y.view(-1)
        pred = torch.softmax(out, dim=-1)
        #'''
        mask = around_predicate.view(-1)
        masked_around_y = y[mask]
        masked_around_pred = pred[mask]
        masked_around_logits = logits[mask]
        '''
        mask = (y != 0) & (y != 28)
        masked_y = y[mask]
        masked_pred = pred[mask]
        masked_logits = logits[mask]
        #'''
        flat_preds = torch.argmax(masked_around_pred,dim=1)
        #print(masked_pred.size(),pred, "PREDS")
        #print(flat_preds.size(),flat_preds, "ARGMAX")
        flat_preds = flat_preds.view(masked_around_pred.size(0),1)
        #print(flat_preds.shape, flat_preds, "ARGMAX")
        #print(masked_y.size(),masked_y,"Y")

        #for x in flat_preds:
        #    print(x)
        #for x in masked_y:
        #    print(x)
        #0/0
        result = {'logits': masked_around_logits, 'pred': masked_around_pred, "labels":masked_around_y,
                  "flat_pred":flat_preds, "mask":mask,
                  "predicate_position":predicate_position}

        # compute loss
        if y is not None:
            # while mathematically the CrossEntropyLoss takes as input the probability distributions,
            # torch optimizes its computation internally and takes as input the logits instead
            loss = self.loss_fn(masked_around_logits, masked_around_y)
            result['loss'] = loss

        return result

    def training_step(
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int
    ) -> torch.Tensor:
        forward_output = self.forward(*batch)
        self.f1(forward_output['flat_pred'], forward_output["labels"])
        #print("\n TRAINING F1 PER CLASS: ",self.f1_per_class(forward_output['flat_pred'], forward_output["labels"]))
        self.log('train_f1', self.f1, prog_bar=True)

        #self.log('train_f1_per_class', self.f1_per_class, prog_bar=True)
        self.log('train_loss', forward_output['loss'],prog_bar=True)
        return forward_output['loss']

    def validation_step(
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int
    ):
        forward_output = self.forward(*batch)
        self.f1(forward_output['flat_pred'], forward_output["labels"])
        #print("\n VALIDATION F1 PER CLASS: ",self.f1_per_class(forward_output['flat_pred'], forward_output["labels"]))

        predicate_position = forward_output["predicate_position"]
        '''
        preds = forward_output["flat_pred"].view(predicate_position.size(0), -1)
        labels = forward_output["labels"].view(predicate_position.size(0), -1)
        converted_preds = dict()
        converted_labels = dict()
        for idx, position in enumerate(predicate_position):
            converted_preds[idx] = {"roles": {position: preds[idx]}}
            converted_labels[idx] = {"roles": {position: labels[idx]}}
        '''
        #print(evaluate_argument_classification(converted_labels, converted_preds))
        self.log('val_f1', self.f1, prog_bar=True)
        #self.log('val_f1_per_class', self.f1_per_class, prog_bar=True)
        self.log('val_loss', forward_output['loss'],prog_bar=True)

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
        #optimizer = torch.optim.SGD(self.parameters(), lr=0.1, momentum=0.0)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001,weight_decay=0.00001)  # instantiating the optimizer
        return optimizer

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
        sentences = SentenceDataset(sentences=sentence,test=True)
        batches = sentences.dataloader(shuffle=False,batch_size=32)
        out = {
            "roles":{}
        }
        for batch in batches:
            X, X_len, segment_ids, y, predicate_position, attention_mask, around_predicate, ids = batch
            output = self.forward(X,X_len,segment_ids,y,predicate_position,attention_mask,around_predicate, ids)
            preds = output["flat_pred"].view(X.size(0),-1)
            masks = output["mask"].view(X.size(0),-1)
            n_values = X.size(0)*X.size(1)
            reconstructed = torch.tensor([28]*n_values,dtype=torch.long, device=device).view(X.size(0),X.size(1))
               # torch.zeros(size=X.size(),dtype=torch.long,device=device)
            #reconstructed[around_predicate] = preds
            for idx in range(len(around_predicate)):
                reconstructed[idx][around_predicate[idx]] = preds[idx]
                #reconstructed[idx] = torch.flatten(reconstructed[idx])
            #reconstructed = preds
            '''
            for idx,position in enumerate(predicate_position):
                if position.item() != -1:
                    #out["roles"][position.item()] = [sentences.idx2roles[role.item()] for role in torch.flatten(reconstructed[idx])][1:-1]
                    ["roles"][position.item()] = sentences.convert_output(torch.flatten(reconstructed[idx])[1:-1])
                    del out["roles"][position.item()][position.item()]
                    del out["roles"][position.item()][position.item()+1]
                #print("a")
            '''
            out = sentences.convert_output(reconstructed.tolist(),predicate_position.tolist())
        return out


early_stopping = pl.callbacks.EarlyStopping(
    monitor='val_f1',  # the value that will be evaluated to activate the early stopping of the model.
    patience=5,  # the number of consecutive attempts that the model has to raise (or lower depending on the metric used) to raise the "monitor" value.
    verbose=True,  # whether to log or not information in the console.
    mode='max', # wheter we want to maximize (max) or minimize the "monitor" value.
)

check_point_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_f1',  # the value that we want to use for model selection.
    verbose=True,  # whether to log or not information in the console.
    save_top_k=1,  # the number of checkpoints we want to store.
    mode='max',  # wheter we want to maximize (max) or minimize the "monitor" value.
    filename='{epoch}-{val_f1:.4f}'  # the prefix on the checkpoint values. Metrics store by the trainer can be used to dynamically change the name.
)
#'''
sentences_dm = SentencesDataModule(
    data_train_path=EN_TRAIN_PATH,
    data_dev_path=EN_DEV_PATH,
    data_test_path=EN_DEV_PATH,
    batch_size=32
)
#'''
classifier = StudentModel(language="en",hidden1=768,lstm_layers=1,bidirectional=False,p=0.5)


# the PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=20,  # maximum number of epochs.
    gpus=1,  # the number of gpus we have at our disposal.
    callbacks=[early_stopping, check_point_callback]  # the callback we want our trainer to use.
)

# and finally we can let the "trainer" fit the amazon reviews classifier.
trainer.fit(model=classifier, datamodule=sentences_dm)


model_path = "../../model/model.ckpt"
loaded = StudentModel.load_from_checkpoint(model_path,language="en").to(device)



def read_dataset(path: str):
    with open(path) as f:
        dataset = json.load(f)

    sentences, labels = {}, {}
    for sentence_id, sentence in dataset.items():
        sentence_id = sentence_id
        sentences[sentence_id] = {
            "words": sentence["words"],
            "lemmas": sentence["lemmas"],
            "pos_tags": sentence["pos_tags"],
            "dependency_heads": [int(head) for head in sentence["dependency_heads"]],
            "dependency_relations": sentence["dependency_relations"],
            "predicates": sentence["predicates"],
        }

        labels[sentence_id] = {
            "predicates": sentence["predicates"],
            "roles": {int(p): r for p, r in sentence["roles"].items()}
            if "roles" in sentence
            else dict(),
        }

    return sentences, labels

sentences,labels = read_dataset(EN_DEV_PATH)
for idx,key in enumerate(sentences):
    prediction = loaded.predict(sentences[key])["roles"]
    lab = labels[key]["roles"]
    print("PREDICTED",prediction)
    print("GROUND TRUTH",lab)



