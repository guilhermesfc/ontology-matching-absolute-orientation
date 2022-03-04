from typing import Dict
import logging
from pyrdf2vec.graphs import KG
from random import shuffle

logger = logging.getLogger(__name__)


def backend_reader(filename) -> Dict:
    kg = KG(location=filename, file_type="rdf")
    entities = list()
    for vertex in kg._vertices:
        if vertex.name.startswith("urn:sap-mapping-tool:schema:"):
            entities.append(vertex.name)
    return {"kg": kg, "entities": entities}


def simulation_reader(filename) -> Dict:
    kg = KG(location=filename + ".ttl", file_type="ttl")
    entities = list(kg.graph.all_nodes())
    shuffle(entities)

    # entities = [node.toPython().rsplit('/')[-1] for node in list(kg.graph.all_nodes())]
    return {"kg": kg, "entities": entities}
