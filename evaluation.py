"""
Author: oufeifei
Date: 2023-05-18 15:27:43
LastEditors: oufeifei
LastEditTime: 2023-05-21 17:13:27
Description: 
"""
import networkx as nx
import numpy as np
# import pymetis_test as pmt
from collections import defaultdict
import matplotlib.pyplot as cm
from matplotlib import pyplot as plt
import main
import igraph as ig


def plot(G, pos):
    # cmap = cm.get_cmap('jet', max(partition.values())+1)
    # eweights = nx.get_edge_attributes(G, "weight")
    # vweights = nx.get_node_attributes(G, "weight")
    # node_size = [v / 200 for v in vweights.values()]
    # nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, alpha=0.5)
    # nx.draw_networkx_labels(G, pos, labels=vweights)
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=eweights)
    plt.savefig(
        f"/home/geo00/DISK3/oufeifei/Code20230506/DPC/result/test/xxxx_cluster.png"
    )


def cluster_plot(
    G,
    pos,
    pred_tags1,
    pred_tags2,
    cut,
    max_rate,
    desc="",
    node_size=500,
    edge_width=1,
    label=False,
):
    # # 真实图
    fig, ax = plt.subplots(1, 2, figsize=(30, 15))
    # nx.draw_networkx_edges(G, pos, alpha=0.2, width=edge_width, ax=ax[0])
    # nx.draw_networkx_nodes(
    #     G,
    #     pos,
    #     nodelist=G.nodes,
    #     node_color=true_tags,
    #     node_size=node_size,
    #     cmap=plt.cm.jet,
    #     alpha=0.8,
    #     ax=ax[0],
    # )
    # if label:
    #     nx.draw_networkx_labels(
    #         G, pos, dict(zip(G.nodes, G.nodes)), font_color="whitesmoke", ax=ax[0]
    #     )
    # ax[0].set_title("Truth")
    # 预测图
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=edge_width, ax=ax[0])
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=G.nodes,
        node_color=pred_tags1,
        node_size=node_size,
        cmap=plt.cm.jet,
        alpha=0.8,
        ax=ax[0],
    )
    if label:
        nx.draw_networkx_labels(
            G, pos, dict(zip(G.nodes, G.nodes)), font_color="whitesmoke", ax=ax[0]
        )
    ax[0].set_title(f"Prediction, n_cuts:{cut[0]}, max_rate:{max_rate[0]}")
    # pymetis
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=edge_width, ax=ax[1])
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=G.nodes,
        node_color=pred_tags2,
        node_size=node_size,
        cmap=plt.cm.jet,
        alpha=0.8,
        ax=ax[1],
    )
    if label:
        nx.draw_networkx_labels(
            G, pos, dict(zip(G.nodes, G.nodes)), font_color="whitesmoke", ax=ax[1]
        )
    ax[1].set_title(f"metis, n_cuts:{cut[1]}, max_rate:{max_rate[1]}")
    fig.savefig(
        f"/home/geo00/DISK3/oufeifei/Code20230506/DPC/result/test/{desc}_cluster.png",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    return fig, ax


def sinplot(G, pos, partition, cut, desc="test_metis"):
    cmap = cm.get_cmap("jet", len(set(partition.values())))
    plt.figure(figsize=(15, 10))
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=partition.keys(),
        node_size=50,
        cmap=cmap,
        node_color=list(partition.values()),
        edgecolors=None,
    )
    nx.draw_networkx_edges(G, pos, alpha=0.2)
    # if label:
    #     nx.draw_networkx_labels(
    #         G, pos, dict(zip(G.nodes, G.nodes)), font_color="whitesmoke", ax=ax[1]
    #     )
    plt.title(f"{desc}, n_cuts:{cut}")
    plt.savefig(
        f"/home/geo00/DISK3/oufeifei/Code20230506/DPC/MyDPC/result/{desc}_cluster.png",
        bbox_inches="tight",
        pad_inches=0.2,
    )
    plt.close()


def cut_edges(G, part_dict):
    res = 0
    for e, w in G.edges().items():
        i = part_dict[e[0]]
        j = part_dict[e[1]]
        if i != j:
            x = w["weight"]
            res += x
    return res


def corase_graph2(origin_G, part_dict, cluster_size):
    G = nx.Graph()
    new_nodes = set(part_dict.values())
    new_node_num = len(new_nodes)
    cut_array = np.zeros(shape=(new_node_num, new_node_num))
    for node in new_nodes:
        G.add_node(node, weight=cluster_size[node])
    # x = G.nodes[0]["weight"]
    for e, w in origin_G.edges().items():
        i = part_dict[e[0]]
        j = part_dict[e[1]]
        if i != j:
            x = w["weight"]
            cut_array[i, j] += x
    cut_weight = cut_array + cut_array.T
    for i in range(new_node_num):
        for j in range(i + 1, new_node_num):
            if cut_weight[i, j] != 0:
                G.add_edge(i, j, weight=cut_weight[i, j])
    return G


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


def cross_edges(originG, match_part_dict):
    # 计算簇大小
    c_size = main.cluster_size(originG, match_part_dict)
    G = corase_graph2(originG, match_part_dict, c_size)
    cut_edges = 0
    for e, w in G.edges().items():
        cut_edges += w["weight"]
    return G, c_size, cut_edges


def maximum_load_rate(c_size: dict):
    """description: 最大负载率
    param:
        c_size: 簇的大小
    return:
        w: 最大负载率
    author: oufeifei
    """
    max_nodes = max(c_size.values())
    balanced_nodes = sum(c_size.values()) / len(c_size.keys())
    w = max_nodes / balanced_nodes
    return w


def result_ours_eval(originG, part_dict, corase_part_dict):
    match_part_dict = dict()
    for k, v in part_dict.items():
        match_part_dict[k] = corase_part_dict[v]
    # 计算交叉边数量
    # cut_edges, corase_g = cross_edges(originG, match_part_dict)
    # print("ours计算的割边:{}".format(cut_edges))
    return match_part_dict
    # pos = nx.circular_layout(corase_g)
    # plot(corase_g, pos)
    # plt.title("algorithm:{},    cut_edges: {}".format("ours", int(cut_edges)))
    # plt.savefig("./ours_result.png")
    # plt.show()


# def result_metis_eval(g, partNum):
#     cutoff, part_dict = pmt.metis(g, partNum)
#     _, csize, cut_edges = corase_graph(g, part_dict)
#     # cut_edges2, csize2 = cross_edges(g, part_dict)
#     print("pymetis计算的割边:{},ours计算的割边:{}".format(cutoff, cut_edges))
#     print(f"ptmetis cluster size: {csize}")
#     # pos = nx.circular_layout(corase_g)
#     # plot(corase_g, pos)
#     # plt.title("algorithm:{},    cut_edges: {}".format("metis", int(cut_edges)))
#     # plt.savefig("./metis_result.png")
#     # plt.show()


def result_hash_eval(g, partNum):
    nodeslist = set(g.nodes())
    part_dict = dict()
    for i, j in enumerate(nodeslist):
        id = i % partNum
        part_dict[j] = id
    _, csize, cut_edges = corase_graph(g, part_dict)
    print("hash,计算的割边:{}".format(cut_edges))
    return part_dict


if __name__ == "__main__":
    g = pmt.load_graph(
        # "/home/geo00/DISK3/oufeifei/Code20230506/DPC/MyDPC/data/Literal_or_Geonames_in_Facts_Geodata_2hop_attr.json"
        "/home/geo00/DISK3/oufeifei/Code20230506/DPC/MyDPC/data/Ghop2_attr_resG.json"
    )

    part_dict = result_hash_eval(g, 10)
