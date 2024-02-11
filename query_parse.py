'''
Author: oufeifei
Date: 2023-12-01 03:48:33
LastEditors: oufeifei
LastEditTime: 2023-12-01 13:58:26
Description: 
'''

import matplotlib.pyplot as plt
import networkx as nx
from rdflib import Graph, URIRef, Literal


def networkx_to_rdf(graph):
    rdf_graph = Graph()

    for node in graph.nodes():
        node_uri = URIRef(node)  # 直接使用节点的名字作为 URI
        # rdf_graph.add((node_uri, URIRef("rdf:type"), URIRef("Node")))
        # rdf_graph.add((node_uri, URIRef("label"), Literal(str(node))))

    for edge in graph.edges(data = True):
        subject_uri = URIRef(edge[0])  # 直接使用节点的名字作为 URI
        object_uri = URIRef(edge[1])  # 直接使用节点的名字作为 URI
        rdf_graph.add((subject_uri, URIRef(edge[2]['label']), object_uri))
    
    return rdf_graph


def file_to_rdf():
    rdf_graph = Graph()
    f = open('/home/off/code/workload_graph_partition/data/G_tri_notype.n3',
             'r', encoding='utf-8')
    for line in f:
        if line.startswith('<'):
            sLine = line.rstrip('.\n').split('\t')
            s,p,o = sLine[0],sLine[1],sLine[2]
            subject_uri = URIRef(s)  # 直接使用节点的名字作为 URI
            object_uri = URIRef(o)  # 直接使用节点的名字作为 URI
            rdf_graph.add((subject_uri, URIRef(p), object_uri))
    return rdf_graph        
            
    

def rdf_query(rdf_graph,sparql_query):
    # 执行SPARQL查询
    # sparql_query = """
    #         SELECT *
    # WHERE {
    #     ?a <p1> <B> .
    #     <B> <p2> ?c .
    #     Filter (?c != <B>)
    # }
    #     """

    results = rdf_graph.query(sparql_query)

    return results

if __name__ == '__main__':
    
    # 创建一个NetworkX图
    G = nx.Graph()
    G.add_nodes_from(["A", "B", "C"])
    G.add_edges_from([("A", "B",{'label':'p1'}),("A", "C",{'label':'p1'}), ("B", "C",{'label':'p2'})])

    # 将NetworkX图转换为RDF图
    rdf_graph = networkx_to_rdf(G)

    # 输出RDF图的三元组
    for triple in rdf_graph:
        print(triple)
    
    results = rdf_query()
    # 输出查询结果
    for row in results:
        print(f"?A: {row}")