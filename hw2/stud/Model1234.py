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


import training_ab
import training_cd
from hw2.model import Model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_AB_PATH = "../../model/modelAB.ckpt"
MODEL_CD_PATH = "../../model/modelCD.ckpt"


class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self):  # loss function
        super().__init__()

        self.modelAB = training_ab.AB_Model.load_from_checkpoint(MODEL_AB_PATH,language="en").to(device)
        self.modelCD = training_cd.CD_Model.load_from_checkpoint(MODEL_CD_PATH,language="en").to(device)



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
        predicates = self.modelAB.predict(sentence=sentence)
        converted_predicates = {0:predicates}
        predictions = self.modelCD.predict(sentence=sentence,predicates=converted_predicates)
        return {
            "predicates":predicates,
            "roles":predictions["roles"]
        }

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

model = StudentModel()


sentences, labels = read_dataset(EN_DEV_PATH)
for idx, key in enumerate(sentences):
    prediction = model.predict(sentences[key])
    lab = labels[key]
    print("PREDICTED", prediction)
    print("GROUND TRUTH", lab)