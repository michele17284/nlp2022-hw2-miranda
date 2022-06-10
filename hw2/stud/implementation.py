
import json
from typing import List, Tuple, Optional, Union
import random
import nltk
from torch import nn
from torch.utils.data import Dataset, DataLoader
from nltk.corpus import wordnet
from functools import partial
from typing import Tuple, List, Any, Dict
import copy
import torchmetrics
import pytorch_lightning as pl
from transformers import BertTokenizer, BertModel
import torch
from torch.utils.data import DataLoader
import nltk

nltk.download('averaged_perceptron_tagger')
from model import Model


PAD_TOKEN = '<pad>'
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EN_TRAIN_PATH = "./../../data/EN/train.json"
EN_DEV_PATH = "./../../data/EN/dev.json"
BERT_PATH = "./model/bert-base-cased"
MODEL_CD_PATH = "./model/modelCD.ckpt"
MODEL_AB_PATH = "./model/modelAB.ckpt"
from stud import training_ab
from stud import training_cd


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_AB_PATH = "./model/modelAB.ckpt"
MODEL_CD_PATH = "./model/modelCD.ckpt"





def build_model_34(language: str, device: str) -> Model:
    """
    The implementation of this function is MANDATORY.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 3 and 4 of the SRL pipeline.
            3: Argument identification.
            4: Argument classification.
    """

    loaded = training_cd.CD_Model.load_from_checkpoint(MODEL_CD_PATH, language=language)
    return loaded


def build_model_234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 2, 3 and 4 of the SRL pipeline.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    raise NotImplementedError


def build_model_1234(language: str, device: str) -> Model:
    """
    The implementation of this function is OPTIONAL.
    Args:
        language: the model MUST be loaded for the given language
        device: the model MUST be loaded on the indicated device (e.g. "cpu")
    Returns:
        A Model instance that implements steps 1, 2, 3 and 4 of the SRL pipeline.
            1: Predicate identification.
            2: Predicate disambiguation.
            3: Argument identification.
            4: Argument classification.
    """
    return StudentModel(language=language,device=torch.device(device))


class Baseline(Model):
    """
    A very simple baseline to test that the evaluation script works.
    """

    def __init__(self, language: str, return_predicates=False):
        self.language = language
        self.baselines = Baseline._load_baselines()
        self.return_predicates = return_predicates

    def predict(self, sentence):
        predicate_identification = []
        for pos in sentence["pos_tags"]:
            prob = self.baselines["predicate_identification"].get(pos, dict()).get(
                "positive", 0
            ) / self.baselines["predicate_identification"].get(pos, dict()).get(
                "total", 1
            )
            if random.random() < prob:
                predicate_identification.append(True)
            else:
                predicate_identification.append(False)

        predicate_disambiguation = []
        predicate_indices = []
        for idx, (lemma, is_predicate) in enumerate(
            zip(sentence["lemmas"], predicate_identification)
        ):
            if (
                not is_predicate
                or lemma not in self.baselines["predicate_disambiguation"]
            ):
                predicate_disambiguation.append("_")
            else:
                predicate_disambiguation.append(
                    self.baselines["predicate_disambiguation"][lemma]
                )
                predicate_indices.append(idx)

        argument_identification = []
        for dependency_relation in sentence["dependency_relations"]:
            prob = self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get("positive", 0) / self.baselines["argument_identification"].get(
                dependency_relation, dict()
            ).get(
                "total", 1
            )
            if random.random() < prob:
                argument_identification.append(True)
            else:
                argument_identification.append(False)

        argument_classification = []
        for dependency_relation, is_argument in zip(
            sentence["dependency_relations"], argument_identification
        ):
            if not is_argument:
                argument_classification.append("_")
            else:
                argument_classification.append(
                    self.baselines["argument_classification"][dependency_relation]
                )

        if self.return_predicates:
            return {
                "predicates": predicate_disambiguation,
                "roles": {i: argument_classification for i in predicate_indices},
            }
        else:
            return {"roles": {i: argument_classification for i in predicate_indices}}

    @staticmethod
    def _load_baselines(path="data/baselines.json"):
        with open(path) as baselines_file:
            baselines = json.load(baselines_file)
        return baselines








class StudentModel(Model):

    # STUDENT: construct here your model
    # this class should be loading your weights and vocabulary
    # MANDATORY to load the weights that can handle the given language
    # possible languages: ["EN", "FR", "ES"]
    # REMINDER: EN is mandatory the others are extras
    def __init__(self,language,device):  # loss function
        super().__init__()

        self.modelAB = training_ab.AB_Model.load_from_checkpoint(MODEL_AB_PATH,language=language).to(device)
        self.modelCD = training_cd.CD_Model.load_from_checkpoint(MODEL_CD_PATH,language=language).to(device)



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






