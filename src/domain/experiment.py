import os
import logging
import uuid

from typing import Dict
from datetime import datetime
import pandas as pd

from domain.model import AbstractMappingModel
from domain.analyzer import Analyzer

# from domain.input_data import InputData

from service_layer.transformer import absolute_orientation, Gavish_Donoho
from service_layer.transformer import norm_vectors, translate
from service_layer.utils import split_alignments


logger = logging.getLogger(__name__)


class Experiment(object):
    def __init__(
        self,
        config: Dict,
        model: AbstractMappingModel,
        indata_resultDict: Dict[str, str],
    ):
        """

        Parameters
        ----------
        config
        model
        indata_resultDict: Dict[str,str]
        """
        self.cfg = config
        self.model = model
        self.indata_resultDict = indata_resultDict

    def run(self):
        logger.info("Configure the model ...")

        # read from file and define self alignment ...
        logger.info("Train the embeddings ...")
        self.model.train_embeddings()
        self.model.filter_to_embeddings()
        logger.info("Split the data ...")
        train, test = split_alignments(
            alignments=self.model.alignments, train_percentage=self.cfg["train_percentage"]
        )

        if self.cfg["translate"]:
            logger.info("Translate the embeddings ...")
            self.model.embeddings["source"] = translate(self.model.embeddings["source"])
            self.model.embeddings["target"] = translate(self.model.embeddings["target"])

        if self.cfg["norm_before_absOrient"]:
            logger.info("Norm the embeddings BEFORE absolute Orientation ...")
            self.model.embeddings["source"] = norm_vectors(
                self.model.embeddings["source"]
            )
            self.model.embeddings["target"] = norm_vectors(
                self.model.embeddings["target"]
            )

        if self.cfg["plot_singular_values"]:
            logger.info("Calculate Gavish_Donoho ...")
            self.model.embeddings = Gavish_Donoho(
                self.cfg["resultdir"], self.model.embeddings
            )

        logger.info("Calculate absolute orientation ...")
        self.model.embeddings["source"] = absolute_orientation(
            self.model.embeddings, train
        )

        if self.cfg["norm_after_absOrient"]:
            logger.info("Norm the embeddings AFTER absolute Orientation ...")
            self.model.embeddings["source"] = norm_vectors(
                self.model.embeddings["source"]
            )
            self.model.embeddings["target"] = norm_vectors(
                self.model.embeddings["target"]
            )

        # EVALUATION
        logger.info("Start evaluation...")

        results_filename = os.path.join(
            self.cfg["resultdir"], uuid.uuid4().hex[:24]
        )

        analyzer = Analyzer(
            config=self.cfg, fname=results_filename,
            model=self.model, train=train, test=test
        )
        if self.cfg["write_gephi"]:
            analyzer.write_gephi_files(self.cfg["resultdir"])

        if self.cfg["create_plots"]:
            analyzer.plot_metric_test(self.cfg["resultdir"])
            analyzer.plot_metric_train(self.cfg["resultdir"])

        result_dict = self.__get_outrow__()
        result_dict.update(self.model.cfg)
        result_dict.update(analyzer.get_outrow())
        result_dict['fname'] = results_filename.split('/')[-1]
        result = pd.DataFrame(result_dict, index=[0])
        result.to_csv(results_filename + ".csv", index=False)

    def __get_outrow__(self):
        result_dict = {"timestamp": datetime.now().strftime("%m/%d/%Y-%H:%M:%S")}
        result_dict.update(self.indata_resultDict)
        result_dict.update(
            {
                "embedding_metric": self.cfg["embedding_metric"],
                "train_percentage": self.cfg["train_percentage"],
                "translate": self.cfg["translate"],
                "norm_before_absOrient": self.cfg["norm_before_absOrient"],
                "norm_after_absOrient": self.cfg["norm_after_absOrient"],
            }
        )
        return result_dict
