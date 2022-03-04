import os
import shutil
import uuid
import logging

from typing import Dict

from domain.experiment import Experiment
from domain.javaModel import javaMappingModel
from shutil import copyfile

from domain.input_data import PoissonTree, PoissonGraph, XMLData
from service_layer.utils import add_noise_2_alignment
from service_layer.alignment_to_csv import AlignmentToCsv


logger = logging.getLogger(__name__)


def run_experiment(config: Dict):

    # -----------------------------
    # do working directory stuff
    # -----------------------------
    if not config["keep_all"]:
        config["workdir"] = os.path.join(config["workdir"], uuid.uuid4().hex[:8])

    if not os.path.exists(config["workdir"]):
        os.makedirs(config["workdir"], exist_ok=True)

    indata_filenames = {
        "source": os.path.join(config["workdir"], "Source_Data.ttl"),
        "target": os.path.join(config["workdir"], "Target_Data.ttl"),
        "alignment": os.path.join(config["workdir"], "Alignments.csv"),
    }

    # -----------------------------
    # defining the input data
    # -----------------------------
    indata_resultDict = {"indata_type": config["indata_type"]}
    indata_resultDict.update(config["indata_cfg"])
    if config["indata_type"] in [
        "synthetic_tree_ntot",
        "synthetic_tree_depth",
        "synthetic_graph",
    ]:
        if config["indata_type"] == "synthetic_tree_ntot":
            synthetic_data = PoissonTree(
                rate=config["indata_cfg"]["rate"],
                total_number_of_nodes=config["indata_cfg"]["total_number_of_nodes"],
            )
        if config["indata_type"] == "synthetic_tree_depth":
            synthetic_data = PoissonTree(
                rate=config["indata_cfg"]["rate"],
                tree_depth=config["indata_cfg"]["tree_depth"],
            )
        if config["indata_type"] == "synthetic_graph":
            synthetic_data_source = PoissonGraph(
                edge_rate=config["indata_cfg"]["edge_rate"],
                total_number_of_nodes=config["indata_cfg"]["total_number_of_nodes"],
            )
            synthetic_data = PoissonGraph(
                edge_rate=config["indata_cfg"]["edge_rate"],
                total_number_of_nodes=config["indata_cfg"]["total_number_of_nodes"],
            )

        synthetic_data_source.create_kg()
        logger.info("Size of synthetic data (Source) is %s", str(len(synthetic_data_source.kg)))
        logger.info("Write synthetic data (Source) ...")
        synthetic_data_source.write_file(indata_filenames["source"])

        # we need the exact same kg in target to create the alignment
        synthetic_data.set_kg(synthetic_data_source.kg)
        logger.info("Size of synthetic data (Target) is %s", str(len(synthetic_data.kg)))

        logger.info("Shuffle and rename target synthetic data ...")
        alignments = synthetic_data.get_self_alignment(rename=True)
        logger.info("Alignment noise is set to %s", str(config["indata_cfg"]["alignment_noise"]))
        alignments = add_noise_2_alignment(
            alignments, percentage=config["indata_cfg"]["alignment_noise"]
        )
        logger.info("Size of alignment is %s", str(len(alignments)))
        alignments.to_csv(indata_filenames["alignment"])

        if "massage_data" in config["indata_cfg"].keys():
            if config["indata_cfg"]["massage_data"]:
                # --- reflect anatomy data ---

                if config["indata_cfg"]["add_BNodes_Literals"]:
                    # add BNodes and Literals
                    synthetic_data_source.scale_graph(is_target=False)
                    logger.info("Size of synthetic data (Source) after scaling is %s", str(len(synthetic_data_source.kg)))
                    logger.info("Write synthetic data (Source) after scaling ...")
                    synthetic_data_source.write_file(indata_filenames["source"])

                # append target data by more URIRef nodes
                # len(target_URIRef_URIRef) ~= 2 * len(source_URIRef_URIRef) as found in anatomy data
                # parametrized by target_to_source_ratio
                if "target_to_source_ratio" in config["indata_cfg"].keys():
                    for _ in range(config["indata_cfg"]["target_to_source_ratio"] - 1 ):
                        logger.info("Append target synthetic data by more URIRef nodes ...")
                        synthetic_data.create_kg(rename=True, append=False)
                        logger.info("Size of synthetic data (Target) after adding URIRefs is %s", str(len(synthetic_data.kg)))

                # remove randomly a fraction of triples in target graph
                if "remove_triples_percentage" in config["indata_cfg"].keys():
                    synthetic_data.remove_triples(config["indata_cfg"]["remove_triples_percentage"])

                if config["indata_cfg"]["add_BNodes_Literals"]:
                    # add BNodes and Literals
                    synthetic_data.scale_graph(is_target=True)
                    logger.info("Size of synthetic data (Target) after scaling is %s", str(len(synthetic_data.kg)))

        logger.info("Write synthetic data (Target) ...")
        synthetic_data.write_file(indata_filenames["target"])

    if config["indata_type"] == "oaei_anatomy":
        # we now have another file format
        indata_filenames["source"] = indata_filenames["source"].replace(
            "Source_Data.ttl", "Source_Data.rdf"
        )
        indata_filenames["target"] = indata_filenames["target"].replace(
            "Target_Data.ttl", "Target_Data.rdf"
        )

        # when we loop over data input folders,
        # we need to (re-)set the home dir
        print('--------------------------------------')
        os.chdir('/'.join(config['workdir'].split('/')[:config['workdir'].split('/').index('mt-ds-sandbox')+1]))
        print(os.getcwd())
        print('--------------------------------------')

        source_file = f".{os.sep}data{os.sep}anatomy_track{os.sep}anatomy_track-default{os.sep}mouse-human-suite{os.sep}source.rdf"
        target_file = f".{os.sep}data{os.sep}anatomy_track{os.sep}anatomy_track-default{os.sep}mouse-human-suite{os.sep}target.rdf"
        copyfile(source_file, indata_filenames["source"])
        copyfile(target_file, indata_filenames["target"])
        AlignmentToCsv.write_csv_alignment(
            alignment_api_file=f".{os.sep}data{os.sep}anatomy_track{os.sep}anatomy_track-default{os.sep}mouse-human-suite{os.sep}reference.rdf",
            file_to_write_path=indata_filenames["alignment"],
        )
        logger.info("Copied anatomy files.")

        # we now need to set the files in the indata_resultDict
        indata_resultDict["source"] = indata_filenames["source"]
        indata_resultDict["target"] = indata_filenames["target"]
        indata_resultDict["alignment"] = indata_filenames["alignment"]

    if config["indata_type"] == "multifarm":
        # indata_subtype defines which subset of data is analyzed
        indata_resultDict["indata_subtype"] = config["indata_subtype"]
        # we now have another file format
        indata_filenames["source"] = indata_filenames["source"].replace(
            "Source_Data.ttl", "Source_Data.rdf"
        )
        indata_filenames["target"] = indata_filenames["target"].replace(
            "Target_Data.ttl", "Target_Data.rdf"
        )

        # when we loop over data input folders,
        # we need to (re-)set the home dir
        print('--------------------------------------')
        os.chdir('/'.join(config['workdir'].split('/')[:config['workdir'].split('/').index('mt-ds-sandbox')+1]))
        print(os.getcwd())
        print('--------------------------------------')

        source_file = f".{os.sep}data{os.sep}multifarm{os.sep}multifarm_de_en{os.sep}" + config["indata_subtype"] + f"{os.sep}source.rdf"
        target_file = f".{os.sep}data{os.sep}multifarm{os.sep}multifarm_de_en{os.sep}" + config["indata_subtype"] + f"{os.sep}target.rdf"
        copyfile(source_file, indata_filenames["source"])
        copyfile(target_file, indata_filenames["target"])
        AlignmentToCsv.write_csv_alignment(
            alignment_api_file=f".{os.sep}data{os.sep}multifarm{os.sep}multifarm_de_en{os.sep}" + config["indata_subtype"] + f"{os.sep}reference.rdf",
            file_to_write_path=indata_filenames["alignment"],
        )
        logger.info("Copied multifarm files.")

        # we now need to set the files in the indata_resultDict
        indata_resultDict["source"] = indata_filenames["source"]
        indata_resultDict["target"] = indata_filenames["target"]
        indata_resultDict["alignment"] = indata_filenames["alignment"]

    if config["indata_type"] == "scc":

        # when we loop over data input folders,
        # we need to (re-)set the home dir
        print('--------------------------------------')
        os.chdir('/'.join(config['workdir'].split('/')[:config['workdir'].split('/').index('mt-ds-sandbox')+1]))
        print(os.getcwd())
        print('--------------------------------------')

        source_file = f".{os.sep}data{os.sep}scc{os.sep}scc_bar.ttl"
        target_file = f".{os.sep}data{os.sep}scc{os.sep}scc_apqc.ttl"
        copyfile(source_file, indata_filenames["source"])
        copyfile(target_file, indata_filenames["target"])
        AlignmentToCsv.write_ttl_to_csv_alignment(
            alignment_api_file=f".{os.sep}data{os.sep}scc{os.sep}scc_rel.ttl",
            file_to_write_path=indata_filenames["alignment"],
        )
        logger.info("Copied scc files.")

        # we now need to set the files in the indata_resultDict
        indata_resultDict["source"] = indata_filenames["source"]
        indata_resultDict["target"] = indata_filenames["target"]
        indata_resultDict["alignment"] = indata_filenames["alignment"]

    # -----------------------------
    # defining the model
    # -----------------------------
    if config["model_type"] == "java_rdf2vec":
        model = javaMappingModel(
            jar=config["java_executable"],
            metric=config["embedding_metric"],
            workdir=config["workdir"],
            filenames=indata_filenames,
        )
        model.do_cfg(config=config["model_cfg"])

    if config["model_type"] == "pyrdf2vec":
        raise NotImplementedError

    # -----------------------------
    # doing the experiment
    # -----------------------------
    experiment = Experiment(
        config=config, model=model, indata_resultDict=indata_resultDict
    )
    experiment.run()

    # -----------------------------
    # cleanup
    # -----------------------------
    if not config["keep_all"]:
        if os.path.exists(config["workdir"]):
            shutil.rmtree(config["workdir"])
