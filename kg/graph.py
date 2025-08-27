import networkx as nx
from typing import Dict

class NXGraph:
    def __init__(self):
        self.graph = nx.Graph()
    
    def add_node(self,node_id:str,**atributes):
        self.graph.add_node(node_id,**atributes)
    
    def add_nodes(self,nodes:list[Dict]):
        for node in nodes:
            node_id = node.pop("id")
            self.add_node(node_id,**node)