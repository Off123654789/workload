'''
Author: oufeifei
Date: 2023-11-25 21:12:45
LastEditors: oufeifei
LastEditTime: 2023-11-25 21:14:57
Description: 
'''
import networkx as nx
import json
import os
import pandas as pd
import igraph as ig
import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.pyplot as cm


def load_graph(file_path: str) -> nx.Graph:
    """
    description: load graph
    param {str} file_path: path of json file
    return {nx.Graph} networkx Graph instance
    """
    print("load graph start...", end="")
    with open(file_path, "r", encoding="utf-8", newline="\n") as f:
        d = json.load(f)
    graph = nx.node_link_graph(d)
    print("finish.")
    return graph


def write_graph(G, filename):
    """write graph

    Args:
        G : networkx graph
        filename : file path
    """
    data = nx.node_link_data(G)
    d = json.dumps(data, indent=1)
    print("write graph ...", end="")
    with open(filename, "w", encoding="utf-8", newline="\n") as w:
        w.write(d)
    print("finish.")
    
    
def read_csv_dict(filename: str):
    """json 读取
    input:
        filename: 读取文件
    output:
        dictname: 字典dictname
    """
    dictname = dict()
    fexist = os.path.isfile(filename)
    assert fexist, f"Error: {filename}文件不存在"
    df_read = pd.read_csv(filename)
    dictname = df_read.set_index("Key")["Value"].to_dict()
    print(f"{filename}读取到字典已完成")
    return dictname


def corase_graph(origin_G: nx.graph, part_dict: dict):
    """description:
    param:
        origin_G: nx.graph
        part_dict: {k:v}
    return:
        corase_graph:
            c_size:
        cross_edges:
    author: oufeifei
    """
    G = ig.Graph.from_networkx(origin_G, "name")
    respart = {i: part_dict[G.vs[i]["name"]] for i in range(G.vcount())}
    membership = list(respart.values())
    vc = ig.VertexClustering(G, membership)
    corase_graph = vc.cluster_graph(
        combine_edges={"weight": "sum"}, combine_vertices={"weight": "sum"}
    )
    ew = np.array(corase_graph.get_edge_dataframe())[:, 2]
    cross_edges = np.sum(ew)
    c_size = dict(
        zip(
            range(corase_graph.vcount()),
            list(corase_graph.get_vertex_dataframe()["weight"]),
        )
    )
    corase_graph = ig.Graph.to_networkx(corase_graph)
    return corase_graph, c_size, cross_edges


def get_node_with_max_degree(graph):
    degrees = dict(graph.outdegree())  # 获取每个节点的度
    max_degree_node = max(degrees, key=degrees.get)  # 找到具有最大度的节点
    max_degree_value = degrees[max_degree_node]  # 获取最大度的值
    return max_degree_node, max_degree_value


def read_ttl_file(file_path):
    graph = nx.DiGraph()
    lineCounter = 0
    lineProgress = 10000
    with open(file_path, 'r') as file:
        for line in file:
            # Assuming the format is: source_node - edge_label -> target_node
            source, edge_label, target = line.strip('.\n').split('\t')
            # Add nodes and edges to the graph
            # graph.add_node(source)
            # graph.add_node(target)
            graph.add_edge(source, target, label=edge_label)
            # lineCounter += 1
            # if (lineCounter % lineProgress == 0):
            #     break
    # undirected_graph = graph.to_undirected()
    # connected_components = list(nx.weakly_connected_components(graph))
    # max_component = max(connected_components, key=len)
    # subgraph1 = graph.subgraph(max_component)
    # root, max_degree_value = get_node_with_max_degree(subgraph1)
    # edges = nx.bfs_edges(subgraph1, source=root, depth_limit=5)
    # nodes = set()
    # nodes.add(root)
    # for u, v in edges:
    #     nodes.add(u)
    #     nodes.add(v)
    # subgraph = subgraph1.subgraph(nodes)
    # connected_components = list(nx.weakly_connected_components(subgraph2))
    # max_component = max(connected_components, key=len)
    # subgraph = graph.subgraph(max_component)
    return graph


def visualize_graph(graph,pos,name,partition = 0,rdfnodes= 0,nodeshape1 = '*',geonodes = 0,nodeshape2 = '+',size = (25, 15)):
    
    # pos = nx.multipartite_layout(graph, subset_key="layer")  # You can use a different layout if needed
    plt.figure(figsize=size)
    # Draw nodes
    if partition == 0:
        nx.draw_networkx_nodes(graph, pos,node_size=600)
    else:
        cmap = cm.get_cmap("jet", max(partition.values()) + 1)
        nx.draw_networkx_nodes(graph, pos,partition.keys(),cmap=cmap,node_color=[list(partition.values())],node_size=600)
    if rdfnodes:
        nx.draw_networkx_nodes(graph, pos,rdfnodes,node_shape=nodeshape1,node_color='yellow',node_size=600)
    if geonodes:
        nx.draw_networkx_nodes(graph, pos,geonodes,node_shape=nodeshape2,node_color='red',node_size=600)
    # nx.draw(graph,pos)
    # Draw edges
    edge_labels = nx.get_edge_attributes(graph,name='label')
    nx.draw_networkx_edges(graph, pos)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,font_size=15)

    # Draw node labels
    node_labels = {node: node for node in graph.nodes}
    nx.draw_networkx_labels(graph, pos, labels=node_labels,font_size= 15)
    plt.savefig(
            f"workload_graph_partition/result/{name}.png",
            bbox_inches="tight",
            pad_inches=0,
        )
    plt.show()
    plt.close()
    


def pattern_mining():
    f = open('/home/off/code/workload_graph_partition/data/G_tri_notype.n3',
             'r', encoding='utf-8')
    G = nx.DiGraph()
    data = dict()
    nodes = dict()
    for line in f:
        if line.startswith('<'):
            sLine = line.rstrip('.\n').split('\t')
            s,p,o = sLine[0],sLine[1],sLine[2]
            if not p.startswith('<has'):
                if data.get(p) is not None:
                    data[p] += 1
                else:
                    data[p] = 1
                if nodes.get(p) is not None:
                    nodes[p].add(s)
                    nodes[p].add(o)
                else:
                    nodes[p] = {s,o}
    G.add_nodes_from(list(data.keys()),weight=list(data.values()))
    labels = list(G.nodes())
    g = nx.Graph()
    for i in range(len(labels)-1):
        for j in range(i+1,len(labels)):
            ilab = labels[i]
            jlab = labels[j]
            m = len(nodes[ilab] & nodes[jlab])
            if m !=0:
                g.add_edge(ilab,jlab,label=m)
    edges = list(g.edges(data=True))
    for e in edges:
        if e[2]['label'] < 1000:
            g.remove_edge(e[0],e[1])
    pos = nx.circular_layout(g)
    visualize_graph(g,pos,'label_connectisons')         
    
    

if __name__ == "__main__":
    # ttl_file_path = '/home/geo00/DISK3/oufeifei/data/graph/subdata.ttl'  # Replace with the path to your TTL file
    # graph = read_ttl_file(ttl_file_path)
    # visualize_graph(graph)
    
    pattern_mining()
