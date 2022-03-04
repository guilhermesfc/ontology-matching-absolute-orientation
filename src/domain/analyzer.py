import os
import pandas as pd
import numpy as np
import logging
from typing import Dict

from matplotlib import pyplot as plt
from domain.model import AbstractMappingModel
from service_layer.utils import compute_precision_recall_f1score

logger = logging.getLogger(__name__)


class Analyzer(object):
    def __init__(
        self,
        config: Dict,
        fname: str,
        model: AbstractMappingModel,
        train: pd.DataFrame,
        test: pd.DataFrame,
    ):
        self.cfg = config
        self.fname = fname
        self.model = model
        self.train = train
        self.test = test

        logger.info("Analyzer: top_n = %s", str(self.cfg["analyzer_cfg"]["top_n"]))

        logger.info("Analyzer: Predict and validate train-dataset ...")
        prediction_train = self.model.predict(
            source_entities=self.train.index.tolist(), n_top=self.cfg["analyzer_cfg"]["top_n"]
        )
        self.validation_train = self.train.join(prediction_train, how="inner")

        logger.info("Analyzer: Predict and validate test-dataset ...")
        prediction_test = self.model.predict(
            source_entities=self.test.index.tolist(), n_top=self.cfg["analyzer_cfg"]["top_n"]
        )
        self.validation_test = self.test.join(prediction_test, how="inner")
        logger.info("Analyzer: prediction done")

        # for synthetic data: evaluate performance on test set without alignment noise:
        if "target_original" in self.validation_test.columns:
            self.validation_test.rename({"target": "target_noise",
                                         "target_original": "target"},
                                         axis=1,
                                         inplace=True)
            logger.info("Analyzer: on test set swap target w/ noise to target w/o noise")

        if self.cfg["analyzer_cfg"]["save_predictions"]:
            # save data to analyze the predictions using a notebook:
            self.validation_train.to_csv(
                self.fname + "_matches_train.csv", sep=";"
            )
            self.validation_test.to_csv(
                 self.fname + "_matches_test.csv", sep=";"
            )
            logger.info("Analyzer: predictions written to csv ")

    def get_outrow(self):

        result_dict = dict()

        precision, recall, f1_score = compute_precision_recall_f1score(
            self.validation_train, self.cfg["analyzer_cfg"]["threshold"], self.cfg["analyzer_cfg"]["top_n"]
        )
        result_dict["train-precision"] = precision
        result_dict["train-recall"] = recall
        result_dict["train-f1"] = f1_score

        precision, recall, f1_score = compute_precision_recall_f1score(
            self.validation_test, self.cfg["analyzer_cfg"]["threshold"], self.cfg["analyzer_cfg"]["top_n"]
        )
        result_dict["test-precision"] = precision
        result_dict["test-recall"] = recall
        result_dict["test-f1"] = f1_score

        logger.info("Metric threshold = %s", str(self.cfg["analyzer_cfg"]["threshold"]))
        logger.info("Top N matches = %s", str(self.cfg["analyzer_cfg"]["top_n"]))

        logger.info("\t TRAINING")
        logger.info("\t \t precision = %s %%", str(result_dict["train-precision"]))
        logger.info("\t \t recall    = %s %%", str(result_dict["train-recall"]))

        logger.info("\t TESTING")
        logger.info("\t \t precision = %s %%", str(result_dict["test-precision"]))
        logger.info("\t \t recall    = %s %%", str(result_dict["test-recall"]))

        return result_dict

    def plot_metric_train(self, outpath):
        # histogram
        plt.title("Metric distribution of TRAIN")
        plt.xlabel("Metric")
        plt.ylabel("Probability density")
        plt.hist(self.validation_train["metric_1"], bins=50, density=True)
        plot_filename = os.path.join(outpath, "Hist_metric_train.png")
        plt.savefig(plot_filename)
        plt.show()

    def plot_metric_test(self, outpath):
        # histogram
        plt.title("Metric distribution of TEST")
        plt.xlabel("Metric")
        plt.ylabel("Probability density")
        plt.hist(self.validation_test["metric_1"], bins=50, density=True)
        plot_filename = os.path.join(outpath, "Hist_metric_test.png")
        plt.savefig(plot_filename)
        plt.show()

    def write_gephi_files(self, outpath):
        logger.info("Write files for gephi visualization ...")
        tmp_train = self.validation_train
        tmp_train = tmp_train.assign(property="Train - Wrong")
        indicies = tmp_train["target"] == tmp_train["prediction_target_1"]
        tmp_train.loc[indicies, "property"] = "Train - Correct"

        tmp_test = self.validation_test
        tmp_test = tmp_test.assign(property="Test - Wrong")
        indicies = tmp_test["target"] == tmp_test["prediction_target_1"]
        tmp_test.loc[indicies, "property"] = "Test - Correct"

        gephi_filename = os.path.join(outpath, "gephi4SourceGraph.csv")
        gephi_data = pd.concat(
            [tmp_train[["property", "metric_1"]], tmp_test[["property", "metric_1"]]]
        )
        gephi_data.to_csv(gephi_filename, index_label="Node")

        tmp_train.index = tmp_train["target"]
        tmp_test.index = tmp_test["target"]
        gephi_filename = os.path.join(outpath, "gephi4TargetGraph.csv")
        gephi_data = pd.concat(
            [tmp_train[["property", "metric_1"]], tmp_test[["property", "metric_1"]]]
        )
        gephi_data.to_csv(gephi_filename, index_label="Node")
