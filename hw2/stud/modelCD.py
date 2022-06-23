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

#import matplotlib.pyplot as plt
nltk.download('averaged_perceptron_tagger')
# seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#
#stop_tokens = set(stopwords.words('english'))
#punc_tokens = set(punctuation)
#stop_tokens.update(punc_tokens)
#lemmatizer = WordNetLemmatizer()

# setting the embedding dimension
EMBEDDING_DIM = 100
POS_EMBEDDING_DIM = 10

# specify the device
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print(device)
# setting unknown token to handle out of vocabulary words and padding token to pad sentences
UNK_TOKEN = '<unk>'
PAD_TOKEN = '<pad>'
print(torch.version.cuda)
BERT_PATH = "./model/bert-base-cased"
#BERT_PATH = "../../model/bert-base-cased"

SEMANTIC_ROLES = ["AGENT", "ASSET", "ATTRIBUTE", "BENEFICIARY", "CAUSE", "CO_AGENT", "CO_PATIENT", "CO_THEME",
                  "DESTINATION",
                  "EXPERIENCER", "EXTENT", "GOAL", "IDIOM", "INSTRUMENT", "LOCATION", "MATERIAL", "PATIENT", "PRODUCT",
                  "PURPOSE",
                  "RECIPIENT", "RESULT", "SOURCE", "STIMULUS", "THEME", "TIME", "TOPIC", "VALUE","_"]




class SentenceDataset(Dataset):

    def __init__(self, sentences_path=None, sentence=None, predicates=None,
                 test=False):
        self.test = test
        self.sentences = self.read_sentences(sentences_path=sentences_path,single_sentence=sentence,input_predicates=predicates)

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
    def read_sentences(self, sentences_path, single_sentence, input_predicates):
        sentences = list()
        if sentences_path:
            with open(sentences_path) as file:
                json_file = json.load(file)
        else:
            json_file = {0:single_sentence}
        for key in json_file:
            json_file[key]["id"] = key
            #'''
            if input_predicates:
                json_file[key]["predicates"] = input_predicates[key]
            if self.test:
                json_file[key]["roles"] = {}
                for idx,predicate in enumerate(json_file[key]["predicates"]):
                    if predicate != "_":
                        json_file[key]["roles"][idx] = ["_"]*(idx)+["_"]*(len(json_file[key]["predicates"])-idx)  #TODO find a solution for this
                if not json_file[key]["roles"]:
                    json_file[key]["roles"] = ["_" for i in range(len(json_file[key]["predicates"]))]
            #'''

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
                    #'''
                    around_number = 10
                    start = max(0, int(position) - 5)
                    stop = min(len(roles), int(position) + 5)
                    for i in range(start, stop): instance["around_predicate"][i] = 1
                    count = instance["around_predicate"].count(1)
                    #'''
                    '''
                    if count < around_number:
                        if start == 0:
                            for i in range(stop + (around_number - count)):
                                instance["around_predicate"][i] = 1
                        elif stop == len(roles):
                            for i in range(len(roles) - around_number, len(roles)):
                                instance["around_predicate"][i] = 1
                    #'''
                    #instance["around_predicate"][max([int(position)-10,0]):min([int(position)+10,len(roles)])] = [1]*20
                    sentences.append(instance)
            else:
                instance = copy.deepcopy(json_file[key])

                #if json_file[key]["predicates"].count("_") == len(json_file[key]["predicates"]):
                instance["roles"] = ["_" for i in range(len(json_file[key]["predicates"]))]
                instance["predicate_position"] = -1
                instance["around_predicate"] = [0] * len(instance["roles"])
                instance["attention_mask"] = [1] * len(instance["roles"])
                sentences.append(instance)
        return sentences

    def bert_preprocess(self,sentences):    #TODO tokenize with bert
        tokenizer = BertTokenizer.from_pretrained(BERT_PATH,local_files_only=True)
        #tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
        for sentence in sentences:
            non_joined_text = ["[CLS]"]+sentence["words"]+["[SEP]"]
            if sentence["predicate_position"] != -1:
                non_joined_text.insert(sentence["predicate_position"]+1, "<pred>")
                non_joined_text.insert(sentence["predicate_position"]+3, "<pred>")
                sentence["lemmas"].insert(sentence["predicate_position"] + 1, "<pred>")
                sentence["lemmas"].insert(sentence["predicate_position"] + 3, "<pred>")
            encoded = tokenizer.convert_tokens_to_ids(non_joined_text)
            segments_ids = [1] * len(non_joined_text)
            sentence["segment_ids"] = segments_ids
            sentence["encoded_words"] = encoded
            sentence["tokenized_roles"] = ["<pad>"]+sentence["roles"]+["<pad>"]
            sentence["lemmas"] = ["<pad>"] + sentence["lemmas"] + ["<pad>"]
            sentence["attention_mask"] = [0]+sentence["attention_mask"]+[0]
            sentence["around_predicate"] = [0]+sentence["around_predicate"]+[0]
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
        return [word2idx[word] for word in sent]

    # custom collate function, used to create the batches to give as input to the nn
    # it's needed because we are dealing with sentences of variable length and we need padding
    # to be sure that each sentence in a batch has the same length, which is necessary
    def collate(self, data):
        X = [torch.tensor(instance["encoded_words"]) for instance in data]  # extracting the input sentence
        X_len = torch.tensor([x.size(0) for x in X], dtype=torch.long).to(
            device)  # extracting the length for each sentence
        segment_ids = [torch.tensor(instance["segment_ids"]) for instance in data]
        predicate_position = torch.tensor([torch.tensor(instance["predicate_position"]) for instance in data],device=device)
        attention_mask = [torch.tensor(instance["attention_mask"],dtype=torch.bool) for instance in data]
        y = [torch.tensor(self.sent2idx(instance["tokenized_roles"], self.roles2idx)) for instance in data]  # extracting labels for each sentence
        around_predicate = [torch.tensor(instance["around_predicate"],dtype=torch.bool) for instance in data]
        ids = [instance["id"] for instance in data]  # extracting the sentences' ids
        X = torch.nn.utils.rnn.pad_sequence(X, batch_first=True, padding_value=100).to(device)  # padding all the sentences to the maximum length in the batch (forcefully max_len)
        y = torch.nn.utils.rnn.pad_sequence(y, batch_first=True, padding_value=self.roles2idx[PAD_TOKEN]).to(device)  # padding all the labels
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
          if position != -1:
            converted["roles"][position] = self.sent2idx(output[idx],self.idx2roles)[1:-1]
            del converted["roles"][position][position]
            del converted["roles"][position][position+1]
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
            self.train_dataset = SentenceDataset(sentences_path=self.data_train_path)
            self.validation_dataset = SentenceDataset(sentences_path=self.data_dev_path)
        elif stage == 'test':
            self.test_dataset = SentenceDataset(sentences_path=self.data_test_path)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return self.train_dataset.dataloader(batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.validation_dataset.dataloader(batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        return self.test_dataset.dataloader(batch_size=self.batch_size, shuffle=False)




# print(en_dataset[0])
#number_words = len(en_train_dataset.word2idx)
#number_pos = len(en_train_dataset.pos2idx)



# '''


class CD_Model(pl.LightningModule):                                 #TODO this is now a model for role classification, let's make it identification + classification

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self, language: str,  # pos embedding vectors
                 hidden1=768,  # dimension of the first hidden layer
                 hidden2=768,
                 p=0.0,
                 num_classes=29):  # loss function
        super().__init__()
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
            sum_vec = torch.sum(token[-4:], dim=0)
            token_vecs_cat.append(sum_vec)

        token_vecs_cat = torch.stack(token_vecs_cat, dim=0).view(X.size(0),X.size(1),token_embeddings.size(2))
        out = self.classifier(token_vecs_cat)

        out = out.view(-1, out.shape[-1])
        logits = out
        y = y.view(-1)
        pred = torch.softmax(logits, dim=-1)
        #'''
        mask = around_predicate.view(-1)
        masked_around_y = y[mask]
        masked_around_pred = pred[mask]
        masked_around_logits = logits[mask]
        masked_flat_preds = torch.argmax(masked_around_pred,dim=1)
        masked_flat_preds = masked_flat_preds.view(masked_around_pred.size(0),1)
        #'''
        flat_preds = torch.argmax(pred,dim=1)
        flat_preds = flat_preds.view(pred.size(0),1)
        result = {'logits': logits, 'pred': pred, "labels":y,
                  "flat_pred":flat_preds, "masked_flat_pred":masked_flat_preds, "mask":mask,
                  "predicate_position":predicate_position}

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
        self.log('train_loss', forward_output['loss'],prog_bar=True)
        return forward_output['loss']

    def validation_step(
            self,
            batch: Tuple[torch.Tensor],
            batch_idx: int
    ):
        forward_output = self.forward(*batch)
        self.f1(forward_output['flat_pred'], forward_output["labels"])
        self.log('val_f1', self.f1, prog_bar=True)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=0.00001,weight_decay=0.0000)  # instantiating the optimizer
        return optimizer

    def predict(self, sentence, predicates=None):
        sentences = SentenceDataset(sentence=sentence,test=True,predicates=predicates)
        batches = sentences.dataloader(shuffle=False,batch_size=10)
        out = {
            "roles":{}
        }
        for batch in batches:
            X, X_len, segment_ids, y, predicate_position, attention_mask, around_predicate, ids = batch
            output = self.forward(X,X_len,segment_ids,y,predicate_position,attention_mask,around_predicate, ids)
            preds = output["flat_pred"].view(X.size(0),-1)
            n_values = X.size(0)*X.size(1)
            '''
            reconstructed = torch.tensor([28]*n_values,dtype=torch.long, device=device).view(X.size(0),X.size(1))
            for idx in range(len(around_predicate)):
                reconstructed[idx][around_predicate[idx]] = preds[idx][around_predicate[idx]]
            '''
            out = sentences.convert_output(preds.tolist(),predicate_position.tolist())
        return out

'''
early_stopping = pl.callbacks.EarlyStopping(
    monitor='val_f1',  # the value that will be evaluated to activate the early stopping of the model.
    patience=2,  # the number of consecutive attempts that the model has to raise (or lower depending on the metric used) to raise the "monitor" value.
    verbose=True,  # whether to log or not information in the console.
    mode='max', # wheter we want to maximize (max) or minimize the "monitor" value.
)

check_point_callback = pl.callbacks.ModelCheckpoint(
    monitor='val_f1',  # the value that we want to use for model selection.
    verbose=True,  # whether to log or not information in the console.
    save_top_k=1,  # the number of checkpoints we want to store.
    mode='max',  # wheter we want to maximize (max) or minimize the "monitor" value.
    filename='model_CD{epoch}-{val_f1:.4f}'  # the prefix on the checkpoint values. Metrics store by the trainer can be used to dynamically change the name.
)



cd_classifier = CD_Model(language="en",p=0.5)
# the PyTorch Lightning Trainer
trainer = pl.Trainer(
    max_epochs=20,  # maximum number of epochs.
    gpus=1,  # the number of gpus we have at our disposal.
    callbacks=[early_stopping, check_point_callback]  # the callback we want our trainer to use.
)
EN_TRAIN_PATH = "./../../data/EN/train.json"
EN_DEV_PATH = "./../../data/EN/dev.json"

cd_dm = SentencesDataModule(
    data_train_path=EN_TRAIN_PATH,
    data_dev_path=EN_DEV_PATH,
    data_test_path=EN_DEV_PATH,
    batch_size=16
)

# and finally we can let the "trainer" fit the amazon reviews classifier.
trainer.fit(model=cd_classifier, datamodule=cd_dm)
model_path = "../../model/modelCD.ckpt"
classifier = CD_Model.load_from_checkpoint(model_path,language="en").to(device)

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
EN_TRAIN_PATH = "./../../data/EN/train.json"
EN_DEV_PATH = "./../../data/EN/dev.json"
sentences,labels = read_dataset(EN_DEV_PATH)


for idx,key in enumerate(sentences):
    prediction = cd_classifier.predict(sentences[key])["roles"]
    lab = labels[key]["roles"]
    print("PREDICTED",prediction)
    print("GROUND TRUTH",lab)

#'''