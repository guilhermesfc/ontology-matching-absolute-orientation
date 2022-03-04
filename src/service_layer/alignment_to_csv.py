import xml.etree.ElementTree as ET
from rdflib import Graph as RDFGraph


class AlignmentToCsv:
    @staticmethod
    def write_csv_alignment(alignment_api_file: str, file_to_write_path: str) -> None:
        namespace = {
            "a": "http://knowledgeweb.semanticweb.org/heterogeneity/alignment",
            "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
        }
        resource_key = "{http://www.w3.org/1999/02/22-rdf-syntax-ns#}resource"
        root = ET.parse(alignment_api_file).getroot()
        with open(file_to_write_path, "w+", encoding="utf8") as file_to_write:
            file_to_write.write(",target\n")  # before: "source,target\n"
            for c in root.findall("a:Alignment/a:map/a:Cell", namespaces=namespace):
                e1 = c.find("a:entity1", namespaces=namespace).attrib[resource_key]
                e2 = c.find("a:entity2", namespaces=namespace).attrib[resource_key]
                # print("E1: " + str(e1))
                # print("E2: " + str(e2))
                # print("")
                # remove http:// from strings
                e1 = e1.split("http://")[-1]
                e2 = e2.split("http://")[-1]
                file_to_write.write(f"{e1},{e2}\n")

    @staticmethod
    def write_ttl_to_csv_alignment(alignment_api_file: str, file_to_write_path: str) -> None:
        alignment = RDFGraph()
        alignment.parse(alignment_api_file, format='turtle')
        print("Alignment rdflib Graph loaded successfully with {} triples".format(len(alignment)))
        with open(file_to_write_path, "w+", encoding="utf8") as file_to_write:
            file_to_write.write(",target\n")  # before: "source,target\n"
            for trip in alignment.__iter__():
                e1 = trip[0]
                e2 = trip[2]
                # remove http:// from strings
                e1 = e1.split("/")[-1]
                e2 = e2.split("/")[-1]
                file_to_write.write(f"{e1},{e2}\n")


""" 
def main():
    AlignmentToCsv.write_csv_alignment(
        alignment_api_file="../../data/anatomy_track/anatomy_track-default/mouse-human-suite/reference.rdf",
        file_to_write_path="./alignment_test.csv")


if __name__ == "__main__":
    main()
    print("DONE")
"""
