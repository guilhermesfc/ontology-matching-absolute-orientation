import os
import pandas as pd
from gephistreamer import graph
from gephistreamer import streamer
from rdflib import Graph


def convert_to_gephi(inGraph, validation_df, tree_type):

    node_list = list()
    for _, row in validation_df.iterrows():
        node_list.append(
            graph.Node(row["Node"], metric=row["metric_1"], prediction=row["property"])
        )

    # for node_name in list(inGraph.all_nodes()):
    #    node_list.append(graph.Node(node_name.rsplit("/")[-1]))

    edge_list = list()
    for s, p, o in inGraph:
        edge_list.append(
            graph.Edge(
                s.rsplit("/")[-1], o.rsplit("/")[-1], custom_property=p.rsplit("/")[-1]
            )
        )

    # stream = streamer.Streamer(streamer.GephiWS(hostname="localhost", port=8085, workspace='workspace0'))
    stream = streamer.Streamer(
        streamer.GephiWS(
            hostname="localhost", port=8085, workspace=tree_type.lower() + "0"
        )
    )
    stream.add_node(*node_list)
    stream.add_edge(*edge_list)


DATAPATH = "/home/dajoka/Code/corpCode/mt-ds-sandbox/results/DEBUG"

for tree_type in ["Source", "Target"]:
    filename = os.path.join(DATAPATH, tree_type + "_Poisson_tree.ttl")
    kg = Graph()
    kg.parse(location=filename, format="turtle")

    validation_filename = os.path.join(DATAPATH, "gephi4" + tree_type + "Graph.csv")
    validation_df = pd.read_csv(validation_filename)
    convert_to_gephi(kg, validation_df, tree_type)
