import logging.config
import copy
import numpy as np

from config_logging import LOGGING_CONFIG

#from config_multifarm import CONFIG
from config_synth_graph import CONFIG

from service_layer.run_experiment import run_experiment

logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger(__name__)

multifarm_subtypes = [
    "cmt-cmt-de-en", "cmt-confOf-en-de", "cmt-conference-en-de",
    "cmt-iasted-en-de", "cmt-sigkdd-en-de", "confOf-iasted-de-en",
    "confOf-sigkdd-de-en", "conference-confOf-de-en",
    "conference-conference-de-en", "conference-iasted-en-de",
    "conference-sigkdd-en-de", "iasted-sigkdd-de-en",
    "sigkdd-sigkdd-de-en", "cmt-confOf-de-en", "cmt-conference-de-en",
    "cmt-iasted-de-en", "cmt-sigkdd-de-en", "confOf-confOf-de-en",
    "confOf-iasted-en-de", "confOf-sigkdd-en-de",
    "conference-confOf-en-de", "conference-iasted-de-en",
    "conference-sigkdd-de-en", "iasted-iasted-de-en",
    "iasted-sigkdd-en-de"
]

def do_many():
    ## sweap through alignment noise levels
    #for exp_var in np.arange(0., 1., .1):
    #    cfg = copy.deepcopy(CONFIG)
    #    cfg["indata_cfg"]["alignment_noise"] = exp_var
    #    run_experiment(cfg)

    ## evaluate matching on all multifarm data subsets
    #for subtype in multifarm_subtypes:
    #    cfg = copy.deepcopy(CONFIG)
    #    cfg["indata_subtype"] = subtype
    #    run_experiment(cfg)

    ## sweap through training percentage
    #for _ in range(CONFIG["num_iterations"]):
    #    for train in np.arange(0.2, 1., .2):
    #        for subtype in multifarm_subtypes:
    #            cfg = copy.deepcopy(CONFIG)
    #            cfg["train_percentage"] = train
    #            cfg["indata_subtype"] = subtype
    #            run_experiment(cfg)

    # Training size
    #for _ in range(CONFIG["num_iterations"]):
    #    for train in np.arange(0.2, 1., .2):
    #        cfg = copy.deepcopy(CONFIG)
    #        cfg["train_percentage"] = train
    #        run_experiment(cfg)
    
    # Alignment noise
    for _ in range(CONFIG["num_iterations"]):
        for train in np.arange(0.2, 1., .2):
            for noise in np.arange(0, 1., .1):
                cfg = copy.deepcopy(CONFIG)
                cfg["train_percentage"] = train
                cfg["indata_cfg"]["alignment_noise"] = noise
                run_experiment(cfg)
    
    import time
    time.sleep(60*10)

    # Graph heterogeneity
    for _ in range(CONFIG["num_iterations"]):
        for train in np.arange(0.2, 1., .2):
            for triples_per in np.arange(0, 1., .1):
                cfg = copy.deepcopy(CONFIG)
                cfg["train_percentage"] = train
                cfg["indata_cfg"]["remove_triples_percentage"] = triples_per
                run_experiment(cfg)

if __name__ == "__main__":
    do_many()
