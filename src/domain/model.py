import logging
import abc

from typing import List, Dict
from math import isclose
from scipy import spatial
from scipy.spatial.distance import cdist

import pandas as pd
import numpy as np


logger = logging.getLogger(__name__)


class AbstractMappingModel(object):
    def __init__(self, metric: str, filenames: Dict[str, str]):
        """Constructor

        Parameters
        ----------
        metric: str
        filenames: Dict[str, str]
            Must contain the following keys:
                - alignment
        """
        self.filenames = filenames
        self.metric = metric
        self.embeddings = dict()
        self.alignments = None
        self.cfg_w2v_kwargs = None
        self.cfg_rdf2v_kwargs = None
        self.transformer = dict()
        logging.info("MappingModel.__init__() ...")
        logging.info("MappingModel.metric = %s", self.metric)

    @property
    def alignments(self):
        return self._alignments

    @alignments.setter
    def alignments(self, value):
        self._alignments = value

    @alignments.getter
    def alignments(self):
        if self._alignments is None:
            if self.filenames is not None:
                self._alignments = pd.read_csv(self.filenames["alignment"], index_col=0)
        return self._alignments

    @abc.abstractmethod
    def cfg_w2v(self, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def cfg_rdf2v(self, **kwargs):
        raise NotImplementedError

    def filter_to_embeddings(self):
        idx = self.alignments.index.intersection(self.embeddings["source"].index)
        if len(self.alignments.index) > len(idx):
            logging.warning(
                "Length of alignments {} > length of idx {}".format(
                    len(self.alignments.index), len(idx)
                )
            )
        intersection_list = list(
            set.intersection(
                set(self.alignments.loc[idx, "target"].values),
                set(self.embeddings["target"].index.to_list()),
            )
        )
        self.alignments = self.alignments[
            self.alignments["target"].isin(intersection_list)
        ]
        logging.info('Filter to embeddings done')

    def predict(self, source_entities: List[str], n_top=1) -> pd.DataFrame:
        logging.info("MappingModel.predict(): Start ...")
        uniq_source_entities = np.unique(source_entities)
        in_embeddings = self.embeddings["source"].loc[uniq_source_entities]
        real_n_top = min(self.embeddings["target"].shape[0], n_top)
        prediction = self.getKNeighbours(
            search_these_vecotrs=in_embeddings,
            in_set_of_vectors=self.embeddings["target"],
            n_top=real_n_top,
            metric=self.metric,
        )
        logging.debug("---- prediction object: {}".format(prediction))
        result = pd.DataFrame()
        for i in range(0, n_top):
            if len(prediction["vertices"]) > i:
                result["prediction_target_{}".format(i+1)] = prediction["vertices"][i].values
            else:
                result["prediction_target_{}".format(i+1)] = ''
            if len(prediction["metric"]) > i:
                result["metric_{}".format(i+1)] = prediction["metric"][i].values
            else:
                result["metric_{}".format(i+1)] = 0
        result.index = prediction["vertices"].index

        logging.info("MappingModel.predict(): Start ... done")
        return result

    @staticmethod
    def getKNeighbours(
        search_these_vecotrs: pd.DataFrame,
        in_set_of_vectors: pd.DataFrame,
        n_top: int,
        metric: str,
    ):
        """
        Args:
            A (pd.DataFrame): row-vectors n x d matrix: n=number of vectors, d=dimension of vectors
            B (pd.DataFrame): row-vectors n x d matrix: n=number of vectors, d=dimension of vectors
            n_top (int): [description]

        Returns:
            [type]: [description]
        """
        result_vertices = [None] * search_these_vecotrs.shape[0]
        result_metric = [None] * search_these_vecotrs.shape[0]
        if n_top > 1:
            logging.info("MappingModel.getKNeighbours(): Start with cdist ...")
            all_dist = cdist(
                search_these_vecotrs.values, in_set_of_vectors.values, metric=metric
            )
            cnt_total = all_dist.shape[0]
            for a_index, these_distances in enumerate(all_dist):
                if a_index % 100 == 0:
                    logging.debug(
                        "getKNeighbours -- {} / {} = {:.3f} %".format(
                            a_index, cnt_total, a_index / cnt_total
                        )
                    )
                indicies_of_sorted_distances = np.argpartition(
                    these_distances, range(n_top)
                )[:n_top]
                result_vertices[a_index] = in_set_of_vectors.index[
                    indicies_of_sorted_distances
                ]
                result_metric[a_index] = these_distances[indicies_of_sorted_distances]
            logging.info("MappingModel.getKNeighbours(): Start with cdist ... done")
        elif n_top == 1:
            logging.info("MappingModel.getKNeighbours(): Start with tree ...")
            # test if the vectors are normalized !
            if metric == "cosine":
                vec = search_these_vecotrs.iloc[0].values
                length = np.sqrt(np.square(vec).sum(axis=0))
                if not isclose(length, 1, abs_tol=1e-6):
                    raise (
                        Exception(
                            "for cosine metric transformer.norm has to be called before. length = {}".format(
                                length
                            )
                        )
                    )
            tree = spatial.KDTree(in_set_of_vectors.values.tolist())
            iter_i = 0
            for _, vec in search_these_vecotrs.iterrows():
                dist, index = tree.query(vec.values)
                result_vertices[iter_i] = in_set_of_vectors.index[index]
                result_metric[iter_i] = dist
                iter_i += 1
            logging.info("MappingModel.getKNeighbours(): Start with tree ... done")

        return {
            "vertices": pd.DataFrame(result_vertices, index=search_these_vecotrs.index),
            "metric": pd.DataFrame(result_metric, index=search_these_vecotrs.index),
        }
