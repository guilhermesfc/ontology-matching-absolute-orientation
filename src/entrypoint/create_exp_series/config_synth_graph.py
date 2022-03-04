CONFIG = {
    # pathes and files
    "workdir": "/home/gcosta/mt-ds-sandbox-paper/results_scc_dev",
    "resultdir": "/home/gcosta/mt-ds-sandbox-paper/results_scc_dev",
    "java_executable": "/home/gcosta/mt-ds-sandbox-paper/bin/jrdf2vec-1.2-SNAPSHOT.jar",
    # general behaviour
    # keep_all: delete all intermediate artefacts (embeddings etc.) to save disk
    "keep_all": False,
    "write_gephi": False,
    "create_plots": False,
    "plot_singular_values": False,
    # num_iterations: the number of iterations to perform (results are not averaged)
    "num_iterations": 5,
    # indata definition
    "indata_type": "synthetic_graph",
    "indata_cfg": {
        # Laplace Stuff (synthetic)
        "edge_rate": 4,
        "total_number_of_nodes": 2500,
        "alignment_noise": 0,
        # "source_file": "",
        # "target_file": "",
        # "alignment_file": ""
        "massage_data": True,
        "add_BNodes_Literals": False,
        "target_to_source_ratio": 1, # integer [1 .. n]
        "remove_triples_percentage": 0 # flaot [0.0 .. 1.0]
    },
    # model
    "model_type": "java_rdf2vec",
    "model_cfg": {"dimension": 100, "window": 6, "numberOfWalks": 150, "port": 1809},
    # general experiment setup
    "embedding_metric": "euclidean",  # cosine, euclidean
    "train_percentage": 0,
    "translate": True,
    "norm_before_absOrient": True,
    "norm_after_absOrient": False,
    "analyzer_cfg": {
        "save_predictions": True,
        "threshold": 0,
        "top_n": 1,
    }
}
