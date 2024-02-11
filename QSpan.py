import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
from Isomorphism import signature
import query_span
from basic_graph_func import visualize_graph


class QS:
    def __init__(self,name):
        self.levels = [[{0:name}]]
        self.edges = []
        self.sups = [[0]]

    def add_level(self,level_data):
        self.levels.append(level_data)

    def add_edge(self,edge_data):
        self.edges.append(edge_data)

    def add_sup(self,edge_data):
        self.sups.append(edge_data)


def find_parents(QSQ_edges,qsq_edges,nsig):
    QSQ_ori = set([ele[0] for ele in QSQ_edges if ele[1] == nsig])
    qsq_ori = set([ele[0] for ele in qsq_edges if ele[1] == nsig])
    comm_ori = qsq_ori - QSQ_ori
    return comm_ori


def Query_Span(qsq,QSQ):
    num_levels = len(qsq.levels)
    for i in range(1,num_levels):
        if len(QSQ.levels) > i:
            sig_ilevel = [list(ele.keys())[0] for ele in QSQ.levels[i]]
        else:
            QSQ.add_level([])
            QSQ.add_sup({})
            QSQ.add_edge([])
            sig_ilevel = []
        for my_dict in qsq.levels[i]:
            nsig = list(my_dict.keys())[0]
            n = list(my_dict.values())[0]
            if nsig in sig_ilevel:
                QSQ.sups[i][nsig] += 1
            else:
                QSQ.levels[i].append(my_dict)
                QSQ.sups[i][nsig]=1
                parents = find_parents(QSQ.edges[i-1],qsq.edges[i-1],nsig)
                for j in parents:
                    QSQ.edges[i-1].append((j,nsig))
    return QSQ


def contribute_graph(QSQ):
    G = nx.DiGraph()
    num_levels = len(QSQ.edges)
    for ilayer in range(1,num_levels):
        orinodes_dict = {list(ele.keys())[0]:(ilayer,i) for i,ele in enumerate(QSQ.levels[ilayer])}
        srcnodes_dict = {list(ele.keys())[0]:(ilayer+1,i) for i,ele in enumerate(QSQ.levels[ilayer+1])}
        for edge in QSQ.edges[ilayer]:
            Con = round(QSQ.sups[ilayer+1][edge[1]]/QSQ.sups[ilayer][edge[0]],2)
            e = (orinodes_dict[edge[0]],srcnodes_dict[edge[1]])
            G.add_node(e[0],layer = ilayer,sup = QSQ.sups[ilayer][edge[0]])
            G.add_node(e[1],layer = ilayer+1,sup = QSQ.sups[ilayer+1][edge[1]])
            G.add_edge(*e,label=Con)
    # visualize_graph(G)
    return G
      

def find_pattern(contri_g,QSQ,thao,lam):
    g = contri_g.copy()
    for node in g.nodes(data = True):
        if node[1]['sup']<thao:
            contri_g.remove_node(node[0])
    g = contri_g.copy()
    # for edge in g.edges(data=True):
    #     if edge[2]['label'] >= lam:
    #         n = edge[0]
    #         if n in contri_g.nodes():
    #             contri_g.remove_node(n)

    pos = nx.multipartite_layout(contri_g, subset_key="layer")
    # visualize_graph(contri_g,pos,'FPcontri')
    FP = list()
    sup = list()
    for node in contri_g.nodes():
        fp = list(QSQ.levels[node[0]][node[1]].values())[0]
        sig = list(QSQ.levels[node[0]][node[1]].keys())[0]
        FP.append({node:[sig,fp]})
        sup.append(QSQ.sups[node[0]][sig])
    return FP,sup
    


def FP_tree(query_dataset,min_support=40,thres=0.8):
    # 例子
    
    QSQ = QS('root')
    for data in query_dataset:
        # data=[edgelist,query_count]
        q = nx.DiGraph()
        q.add_edges_from(data[0])
        qsq = query_span.query_span(q)
        for i in range(data[1]):
            QSQ = Query_Span(qsq,QSQ)

    G=contribute_graph(QSQ)
    fp,sup = find_pattern(G,QSQ,min_support,thres)
    return fp,sup




if __name__ == '__main__':
    # 例子
    edges_list1=[('?x','?y',{'label':'p1'}),('?z','?y',{'label':'p1'}),
               ('?x','?w',{'label':'p2'}),('?z','?w',{'label':'p2'})]

    edges_list2=[('?x','?y',{'label':'p1'}),('?y','?z',{'label':'p2'}),
               ('?z','?w',{'label':'p3'})]

    edges_list3=[('?x','?y',{'label':'p3'}),('?y','?z',{'label':'p4'}),
               ('?y','?w',{'label':'p5'})]
    query_dataset = [[edges_list1,60],[edges_list2,30],[edges_list3,10]]
    FP_tree(query_dataset)
