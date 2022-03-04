import logging
import os

import shutil

from typing import Dict
import pandas as pd

from domain.model import AbstractMappingModel


logger = logging.getLogger(__name__)


class javaMappingModel(AbstractMappingModel):
    def __init__(
        self, jar: str, workdir: str, filenames: Dict[str, str], metric: str = "cosine"
    ):
        """Constructor

        Parameters
        ----------
        jar: str
        workdir: str
        filenames:  Dict[str, str]

        metric: str
        """
        super().__init__(metric=metric, filenames=filenames)
        self.cfg = {
            "trainingMode": "sg",
            "dimension": 200,
            "minCount": 1,
            "sample": 0.0,
            "window": 5,
            "epochs": 5,
            "numberOfWalks": 100,
            "depth": 4,
        }
        self.cfg_args = ""
        self.java_cmd = "java -jar {}".format(jar)
        self.wdir = workdir

    def get_java_cmd(self, filename):
        # return self.java_cmd + ' -graph {} -walkDirectory {}'.format(filename, self.jrdf2vec_outpath) + self.cfg_jrdf2v_args + self.cfg_w2v_args
        return self.java_cmd + " -graph {}".format(filename) + self.cfg_args

    def do_cfg(self, config: Dict[str, str]):
        """
        --- w2v paras ----
        trainingMode <cbow | sg> (default: sg)
        dimension <size_of_vector> (default: 200)
        minCount <number> (default: 1)
        sample <rate> (default: 0.0)
        window <window_size> (default: 5)
        epochs <number_of_epochs> (default: 5)
        --- rdf-walk paras ----
        numberOfWalks: <number> [default 100]
        depth: <depth> (default: 4)
        """
        # let's vary the depth parameter by hand
        #if "depth" in config.keys():
        #    raise Exception(
        #        "depth should not be set for rdf2vec! We set 'depth' == 'window'"
        #    )

        #if "window" in config.keys():
        #    config["depth"] = config["window"]

        for key, val in config.items():
            logging.info("javaMappingModel.cfg_w2v(): %s: %s", str(key), str(val))
            self.cfg_args = self.cfg_args + " -{} {}".format(key, val)
            if key != "port":
                self.cfg[key] = val
        logging.info("javaMappingModel.cfg_args: %s", str(self.cfg_args))

    def train_embeddings(self):
        logging.info("javaMappingModel.train_embeddings(): Start ...")
        for in_type in ["source", "target"]:
            this_outpath = os.path.join(self.wdir, in_type)
            if not os.path.exists(this_outpath):
                os.makedirs(this_outpath, exist_ok=True)
            else:
                shutil.rmtree(this_outpath)
                os.makedirs(this_outpath, exist_ok=True)
            os.chdir(this_outpath)
            cmd = self.get_java_cmd(self.filenames[in_type])
            logging.info("Executing jrdf2vec: %s", str(cmd))
            os.system(cmd)
            embedding = pd.read_csv(
                os.path.join(this_outpath, "walks", "vectors.txt"),
                header=None,
                delim_whitespace=True,
            )
            new_node_names = list()
            for uri_ref_node in embedding[0]:
                if isinstance(uri_ref_node, str):
                    new_node_names.append(uri_ref_node.rsplit("/")[-1])
                else:
                    print(uri_ref_node)
                    new_node_names.append(str(uri_ref_node))
            del embedding[0]
            columns = ["w2v-dim-" + str(col) for col in embedding.columns]
            self.embeddings[in_type] = pd.DataFrame(
                embedding.values, index=new_node_names, columns=columns
            )
            # self.embeddings[in_type].columns = ['w2v-dim-' + str(col) for col in self.embeddings[in_type].columns]
            logging.info("Embeddings calculated for %s", in_type)
            # NaN values in Embeddings break the absolute orientation calculation
            # fix option 1: replace NaN by 0
            #self.embeddings[in_type].fillna(0, inplace=True)
            # fix option 2: drop rows containing NaN
            row_count_before = len(self.embeddings[in_type])
            self.embeddings[in_type].dropna(axis=0, inplace=True)
            row_count_after = len(self.embeddings[in_type])
            rows_dropped = row_count_before - row_count_after
            logging.info("%s rows containing NaN values in embeddings dropped for %s", str(rows_dropped), in_type)


        if self.embeddings["source"].shape[1] != self.embeddings["target"].shape[1]:
            raise Exception(
                "MappingModel - train: trained embeddings do differ in dimensions !"
            )
