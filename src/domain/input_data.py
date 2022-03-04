import logging
import uuid
import abc
import random
import secrets
import collections
import numpy as np
import pandas as pd

from rdflib import Graph, URIRef, BNode, Literal
from rdflib.namespace import XSD
from rdflib.extras.external_graph_libs import rdflib_to_networkx_digraph
from lorem_text import lorem

logger = logging.getLogger(__name__)


class InputData(metaclass=abc.ABCMeta):
    def __init__(self):
        self.kg = Graph()

    @staticmethod
    def random_suffix(hex_length=8):
        return uuid.uuid4().hex[:hex_length]

    def write_file(self, filename):
        self.kg.serialize(destination=filename, format="turtle")

    def read_file(self, filename):
        self.kg.parse(location=filename, format="turtle")

    def get_self_alignment(self, rename=False):
        alignments = pd.DataFrame(columns=["target"])
        if rename:
            for node in self.kg.all_nodes():
                # this_data = pd.DataFrame([self.random_suffix(hex_length=20)], columns=['target'], index=[node.toPython()])
                this_data = pd.DataFrame(
                    ["rename_" + node.toPython()],
                    columns=["target"],
                    index=[node.toPython()],
                )
                alignments = alignments.append(this_data)

            new_graph = Graph()
            for s, p, o in self.kg:
                s_index = s.toPython()
                o_index = o.toPython()
                new_graph.add(
                    (
                        URIRef(alignments.loc[s_index]["target"]),
                        URIRef(p),
                        URIRef(alignments.loc[o_index]["target"]),
                    )
                )
            self.kg = new_graph
        else:
            for node in self.kg.all_nodes():
                this_data = pd.DataFrame(
                    [node.toPython()], columns=["target"], index=[node.toPython()]
                )
                alignments = alignments.append(this_data)

        # store a copy of the target column for test w/o noise
        alignments["target_original"] = alignments["target"]

        return alignments

    def ls(self):
        print(self.kg.serialize(format="turtle").decode("utf-8"))


class XMLData(InputData):
    def __init__(self, filename):
        super().__init__()
        # self.kg.parse(filename, format="nt")
        self.kg.parse(filename, format="xml")


class PoissonTree(InputData):
    def __init__(self, rate, tree_depth=None, total_number_of_nodes=None):
        super().__init__()
        self.tree_depth = tree_depth
        self.rate = rate
        self.total_number_of_nodes = total_number_of_nodes
        self.number_of_nodes_available = total_number_of_nodes

    """
    given total number of points and rate -> depth is growing
    """

    def create_node_fixed_n(self, node_name, level):
        last_level_nodes = [(node_name, level)]
        while self.number_of_nodes_available > 0:
            next_level_nodes = list()
            if len(last_level_nodes) == 0:
                break
            for last_level_node in last_level_nodes:
                [num_childs] = np.random.poisson(lam=self.rate, size=1)
                if num_childs > self.number_of_nodes_available:
                    num_childs = self.number_of_nodes_available
                for _ in range(int(num_childs)):
                    # child_name = "Level{}".format(last_level_node[1]+1) + self.random_suffix(hex_length=20)
                    # child_name = "Level_{}_".format(last_level_node[1]+1) + str(_)
                    child_name = self.random_suffix(hex_length=20)
                    # logger.info("Adding child {}".format(child_name))
                    self.kg.add(
                        (
                            URIRef(last_level_node[0]),
                            URIRef("haschild"),
                            URIRef(child_name),
                        )
                    )
                    self.number_of_nodes_available -= 1
                    next_level_nodes.append((child_name, last_level_node[1] + 1))
            last_level_nodes = next_level_nodes

    """
    given depth and rate -> number of points is growing    
    """

    def create_node_fixed_depth(self, node_name, level):
        if level < self.tree_depth:
            [num_childs] = np.random.poisson(lam=self.rate, size=1)
            for _ in range(num_childs):
                child_name = "Level_{}_".format(level + 1) + self.random_suffix()
                self.kg.add(
                    (URIRef(node_name), URIRef("has_child"), URIRef(child_name))
                )
                self.create_node_fixed_depth(node_name=child_name, level=level + 1)

    def create_kg(self):
        name = "Level_{}_".format(0) + self.random_suffix()
        while len(self.kg.all_nodes()) == 0:
            if self.tree_depth is not None:
                self.create_node_fixed_depth(node_name=name, level=0)

            if self.total_number_of_nodes is not None:
                self.create_node_fixed_n(node_name=name, level=0)
                all_nodes = [node.toPython() for node in self.kg.all_nodes()]
                while len(all_nodes) < 0.9 * self.total_number_of_nodes:
                    self.create_node_fixed_n(node_name=name, level=0)
                    all_nodes = [node.toPython() for node in self.kg.all_nodes()]
                    raise Warning(
                        "Created tree but it did not have enough nodes !!!! Try it again"
                    )


class PoissonGraph(InputData):

    node_names = None

    def __init__(self, total_number_of_nodes, edge_rate):
        super().__init__()
        self.total_number_of_nodes = total_number_of_nodes
        self.edge_rate = edge_rate

    def create_node_names(self, rename=False, append=False):
        # rename = True: for target to be distinguished from source
        # append = True: to increase size of target data
        self.node_names = [None] * self.total_number_of_nodes
        prefix = ''
        offset = 0
        if rename:
            prefix = 'rename_'
        if append:
            offset = self.total_number_of_nodes
        for i in range(self.total_number_of_nodes):
            self.node_names[i] = "{}Node_{}".format(prefix, i+offset)

    def create_kg(self, rename=False, append=False):
        self.create_node_names(rename=rename, append=append)
        for i in range(self.total_number_of_nodes):
            this_list = self.node_names.copy()
            this_node = this_list.pop(i)
            [num_relations] = np.random.poisson(lam=self.edge_rate, size=1)
            if num_relations > 0:
                related_nodes = random.sample(this_list, num_relations)
                for rel_node in related_nodes:
                    self.kg.add(
                        (URIRef(this_node), URIRef("is_connected"), URIRef(rel_node))
                    )

    def set_kg(self, kg):
        for triple in kg.__iter__():
            self.kg.add(triple)

    def remove_triples(self, remove_triples_percentage):
        masks = np.random.rand(len(self.kg)) < remove_triples_percentage
        logger.info("# of triples in target graph before removal %s", str(len(self.kg)))
        for mask, triple in zip(masks, self.kg.__iter__()):
            if mask:
                self.kg.remove(triple)
        logger.info("# of triples in target graph after removal %s", str(len(self.kg)))

    def _generate_BNode(self):
        hexstr = secrets.token_hex(16)
        bnode = BNode('N'+hexstr)
        return bnode

    def _generate_Literal(self):
        some_text = lorem.words(2)
        literal = Literal(some_text, datatype=XSD.string)
        return literal

    def _print_node_types(self, nx_graph):
        return collections.Counter([(type(tup[0]), type(tup[1])) for tup in list(nx_graph.edges)])

    def scale_graph(self, is_target=True):
        # Anatomy data target graph
        #node_types_data = { URIRef: 9423, Literal: 8955, BNode: 7766 }
        node_types_data = {(URIRef, URIRef): 13196,
                           (URIRef, BNode): 1662,
                           (URIRef, Literal): 9406,
                           (BNode,  URIRef): 11090}

        if not is_target:
            # Anatomy data source graph
            #node_types_data = { URIRef: 3105, Literal: 3092, BNode: 1982 }
            node_types_data = {(URIRef, BNode): 1637,
                               (URIRef, URIRef): 5957,
                               (URIRef, Literal): 3108,
                               (BNode,  URIRef): 5256}

        nx_graph = rdflib_to_networkx_digraph(self.kg)
        node_types = self._print_node_types(nx_graph)

        fraction_BNodes_front = node_types_data[(BNode, URIRef)] / node_types_data[(URIRef, URIRef)]
        fraction_BNodes_back  = node_types_data[(URIRef, BNode)] / node_types_data[(URIRef, URIRef)]
        fraction_Literals = node_types_data[(URIRef, Literal)] / node_types_data[(URIRef, URIRef)]

        print('to be fraction of BNodes front:', fraction_BNodes_front)
        print('to be fraction of BNodes back:', fraction_BNodes_back)
        print('to be fraction of Literals:', fraction_Literals)

        BNodes_front_to_add  = int(np.round(node_types[(URIRef, URIRef)] * fraction_BNodes_front, 0))
        BNodes_back_to_add   = int(np.round(node_types[(URIRef, URIRef)] * fraction_BNodes_back, 0))
        Literals_to_add      = int(np.round(node_types[(URIRef, URIRef)] * fraction_Literals, 0))

        print('# BNodes front to add:', BNodes_front_to_add)
        print('# BNodes back to add:', BNodes_back_to_add)
        print('# Literals to add:', Literals_to_add)

        for _ in range(BNodes_front_to_add):
            random_index = random.randrange(len(list(nx_graph.nodes)))
            self.kg.add((self._generate_BNode(), URIRef("is_connected"), list(nx_graph.nodes)[random_index]))

        for _ in range(BNodes_back_to_add):
            random_index = random.randrange(len(list(nx_graph.nodes)))
            self.kg.add((list(nx_graph.nodes)[random_index], URIRef("is_connected"), self._generate_BNode()))

        for _ in range(Literals_to_add):
            random_index = random.randrange(len(list(nx_graph.nodes)))
            self.kg.add((list(nx_graph.nodes)[random_index], URIRef("is_connected"), self._generate_Literal()))

        self._print_node_types(rdflib_to_networkx_digraph(self.kg))
