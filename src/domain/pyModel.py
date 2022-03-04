import logging
import os

from typing import Dict
from random import randint

import pandas as pd
import numpy as np

from pyrdf2vec import RDF2VecTransformer
from pyrdf2vec.walkers import RandomWalker
from pyrdf2vec.embedders import Word2Vec

from domain.model import AbstractMappingModel


logger = logging.getLogger(__name__)


class pyMappingModel(AbstractMappingModel):
    def __init__(self, filenames: Dict[str, str], metric: str = "cosine"):
        super().__init__(metric=metric, filenames=filenames)

    def cfg_w2v(self, **kwargs):
        """
        skip_graham
        0: ContinousBagOfWords
        1: skip_graham
        """
        for key, val in kwargs.items():
            logging.info("pyMappingModel.cfg_w2v(): %s: %s", str(key), str(val))
        self.cfg_w2v_kwargs = kwargs

    def cfg_rdf2v(self, **kwargs):
        for key, val in kwargs.items():
            logging.info("pyMappingModel.cfg_rdf2v(): %s: %s", str(key), str(val))
        self.cfg_rdf2v_kwargs = kwargs

    def train_embeddings(self, data: Dict[str, Dict[str, np.ndarray]]):
        logging.info("pyMappingModel.train_embeddings(): Start ...")
        for in_type in ["source", "target"]:
            logging.info("Configure embedder and Walker...")

            embedder = Word2Vec(**self.cfg_w2v_kwargs)
            # walker = [WeisfeilerLehmanWalker(depth=4)]
            walker = [RandomWalker(**self.cfg_rdf2v_kwargs)]
            self.transformer[in_type] = RDF2VecTransformer(
                embedder=embedder, walkers=walker
            )

            logging.info("pyMappingModel.train_embeddings(): Fit %s ...", in_type)
            os.environ["PYTHONHASHSEED"] = str(randint(0, 1000))
            logging.info(
                'os.environ["PYTHONHASHSEED"]: %s', str(os.environ["PYTHONHASHSEED"])
            )
            embedding = self.transformer[in_type].fit_transform(
                data[in_type]["kg"], data[in_type]["entities"]
            )
            new_node_names = list()
            for uri_ref_node in data[in_type]["entities"]:
                new_node_names.append(uri_ref_node.toPython().rsplit("/", 1)[1])
            self.embeddings[in_type] = pd.DataFrame(embedding, index=new_node_names)
            self.embeddings[in_type].columns = [
                "w2v-dim-" + str(col) for col in self.embeddings[in_type].columns
            ]

        if self.embeddings["source"].shape[1] != self.embeddings["target"].shape[1]:
            raise Exception(
                "MappingModel - train: trained embeddings do differ in dimensions !"
            )
