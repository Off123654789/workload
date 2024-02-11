"""
Author: oufeifei
Date: 2023-07-18 22:46:39
LastEditors: oufeifei
LastEditTime: 2023-07-18 22:46:39
Description: 
"""
import networkx as nx
from mgmetis import metis
import evaluation
import json


def writetxt(data, filename):
    w = open(filename, "w", encoding="utf-8")
    for k in data:
        strl = str(k) + " " + "\n"
        w.write(strl)
    w.close()


def write_graph(G, filename):
    data = nx.node_link_data(G)
    d = json.dumps(data, indent=1)
    print("start")
    with open(filename, "w", encoding="utf-8", newline="\n") as w:
        w.write(d)


def load_graph(file_path: str) -> nx.Graph:
    """
    description: load graph
    param {str} file_path: path of json file
    return {nx.Graph} networkx Graph instance
    """
    with open(file_path, "r", encoding="utf-8", newline="\n") as f:
        d = json.load(f)
    graph = nx.node_link_graph(d)
    return graph


def graph_to_csr(graph: nx.Graph, flag: bool = False, desc: str = "test"):
    adjncy = []
    count = 0
    id_name_dict = {}
    name_id_dict = {}
    # map name to id
    print("mapping name to id...", end="")
    for node in graph:
        id_name_dict[count] = node
        name_id_dict[node] = count
        # adj_list.append(np.zeros(1))
        count += 1
    print("finish.")

    # get adjacency list pymetis requires
    print("constructing adjacency list...", end="")
    xadj = []  # list indicates the index of vi's nbr in adj_list
    e_weights = []
    v_weights = []
    adj_count = 0
    xadj.append(adj_count)
    flag = graph.nodes[id_name_dict[0]]
    if "weight" in flag:
        # print('weight metis')
        for i in range(len(id_name_dict.keys())):
            node = id_name_dict[i]
            v_weights.append(int(graph.nodes[node]["weight"]))
            nbrs: iter = graph[node]
            for nbr in nbrs:
                adjncy.append(name_id_dict[nbr])
                e_weights.append(int(nbrs[nbr]["weight"]))
            adj_count += len(nbrs)
            xadj.append(adj_count)
    else:
        # print("no weight metis")
        for i in range(len(id_name_dict.keys())):
            node = id_name_dict[i]
            nbrs: iter = graph[node]
            for nbr in nbrs:
                adjncy.append(name_id_dict[nbr])
                e_weights.append(int(nbrs[nbr]["weight"]))
            adj_count += len(nbrs)
            xadj.append(adj_count)
    # for node, adjdict in graph.adjacency():
    #     node_id = id_name_dict[node]
    #     tmp_list = []
    #     for adj in adjdict.keys():
    #         tmp_list.append(name_id_dict[adj])
    #     adj_list[node_id] = np.array(tmp_list)
    print("finish.")
    # if flag:
    #     writetxt(
    #         xadj,
    #         f"/home/geo00/DISK3/oufeifei/Code20230506/DPC/MyDPC/graph/{desc}_xadj.txt",
    #     )
    #     writetxt(
    #         adjncy,
    #         f"/home/geo00/DISK3/oufeifei/Code20230506/DPC/MyDPC/graph/{desc}_adjncy.txt",
    #     )
    #     writetxt(
    #         e_weights,
    #         f"/home/geo00/DISK3/oufeifei/Code20230506/DPC/MyDPC/graph/{desc}_eweights.txt",
    #     )
    #     writetxt(
    #         v_weights,
    #         f"/home/geo00/DISK3/oufeifei/Code20230506/DPC/MyDPC/graph/{desc}_vweights.txt",
    #     )
    return xadj, adjncy, v_weights, e_weights


def metis_seg(G: nx.Graph, partNum: int, ubvec: float = 1.001, recursive: bool = True):
    xadj, adjncy, v_weights, e_weights = graph_to_csr(G)
    if recursive:
        _, part = metis.part_graph_recursize(
            partNum, xadj, adjncy, ubvec=ubvec, vwgt=v_weights, adjwgt=e_weights
        )
    else:
        _, part = metis.part_graph_kway(
            partNum, xadj, adjncy, ubvec=ubvec, vwgt=v_weights, adjwgt=e_weights
        )
    node_name = list(G.nodes())
    part_dict = dict(zip(node_name, part))
    # print(ncuts)
    _, mc_size, n_cuts = evaluation.corase_graph(G, part_dict)
    print(f"our,cluster size:{mc_size}")
    print(f"ours,n_cuts: {n_cuts}")
    return part_dict


if __name__ == "__main__":
    G = load_graph(
        "/home/geo00/DISK3/oufeifei/Code20230506/DPC/MyDPC/data/Literal_or_Geonames_in_Facts_Geodata_2hop2_resG.json"
    )
    # nx.set_edge_attributes(G, 1, "weight")
    # nx.set_node_attributes(G, 1, "weight")
    xadj, adjncy, v_weights, e_weights = graph_to_csr(G, False, desc="lfr_test_500")
    ncuts, part = metis.part_graph_recursize(
        2, xadj, adjncy, ubvec=1.05, vwgt=v_weights, adjwgt=e_weights
    )
    node_name = list(G.nodes())
    part_dict = dict(zip(node_name, part))
    # print(part)
    print(ncuts)
    _, mc_size, n_cuts = evaluation.cross_edges(G, part_dict)
    print(f"our,cluster size:{mc_size}")
    print(f"ours,n_cuts: {n_cuts}")
