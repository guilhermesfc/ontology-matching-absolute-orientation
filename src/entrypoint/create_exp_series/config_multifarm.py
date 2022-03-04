CONFIG = {
    # pathes and files
    "workdir": "/home/gcosta/mt-ds-sandbox/results_multifarm",
    "resultdir": "/home/gcosta/mt-ds-sandbox/results_multifarm",
    "java_executable": "/home/gcosta/mt-ds-sandbox/bin/jrdf2vec-1.2-SNAPSHOT.jar",
    # general behaviour
    "keep_all": False,
    "write_gephi": False,
    "create_plots": False,
    "plot_singular_values": False,
    "num_iterations": 5,
    # indata definition
    "indata_type": "multifarm",
    "indata_subtype": "confOf-confOf-de-en",
    "indata_cfg": {
        "edge_rate": 4,
        "total_number_of_nodes": 1500,
        "alignment_noise": 0.0,
        "massage_data": False,
        "add_BNodes_Literals": False,
        "target_to_source_ratio": 1, # integer [1 .. n]
        "remove_triples_percentage":0, # flaot [0.0 .. 1.0]
        "threshold": 0.0 #JUST FOR WRITTING IN OUTPUT
    },
    # model
    "model_type": "java_rdf2vec",
    "model_cfg": {
        "dimension": 200,
        "window": 6,
        "depth": 6,
        "numberOfWalks": 150,
        "port": 1811,
        "embedText": "",
    },
    # general experiment setup
    "embedding_metric": "euclidean",  # cosine, euclidean
    "train_percentage": 0.4,
    "translate": True,
    "norm_before_absOrient": False,
    "norm_after_absOrient": True,
    "analyzer_cfg": {
        "save_predictions": True, # store the matching results for train and test
        "threshold": 0.0, # flaot [0.0 .. 1.0]
        "top_n": 1, # integer [1 .. 10]
    }
}
