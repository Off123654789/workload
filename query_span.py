import networkx as nx
from networkx.algorithms.isomorphism import DiGraphMatcher
from Isomorphism import signature

class qs:
    def __init__(self,qname='q'):
        self.q = qname
        self.levels = []
        self.edges = []
        self.fps = {}

    def add_level(self,level_data):
        self.levels.append(level_data)

    def add_edge(self,edge_data):
        self.edges.append(edge_data)

    def add_fp(self,fp_data):
        self.fps.update(fp_data)


def find_expand_subgraph(G,pattern):
    # 创建有向图 G
    # G = nx.DiGraph()
    # g_edgeslist = [(1, 2, {'label': 'X'}), (2, 3, {'label': 'Y'}), (3, 4, {'label': 'Z'}),
    #                 (1, 5, {'label': 'X'}), (5, 3, {'label': 'Y'}), (5, 6, {'label': 'Z'}),
    #                 (6, 7, {'label': 'X'})]
    # G.add_edges_from(g_edgeslist)

    # 创建子图 H，其中节点是变量 a，边标签是常量
    H = nx.DiGraph()
    # pattern = [('a','b', {'label': 'X'}), ('b', 'c', {'label': 'Y'}), ('c', 'd', {'label': 'Z'})]
    H.add_edges_from(pattern)

    # 使用 DiGraphMatcher 查找匹配
    matcher = DiGraphMatcher(G, H,edge_match=lambda edge1, edge2: edge1['label'] == edge2['label'])

    # 获得匹配的子图
    res = []
    for node_dic in matcher.subgraph_isomorphisms_iter():
        node_dic = {v: k for k, v in node_dic.items()}
        # print("匹配结果:", node_dic)
        subgraph = [(node_dic[edge[0]], node_dic[edge[1]], edge[2]) for edge in pattern]
        res.append(subgraph)
        # print("匹配结果:", subgraph)
    
    expand_res = []
    for edges in res:
        g = nx.DiGraph()
        g.add_edges_from(edges)
        adj_edges = list(G.in_edges(list(g.nodes()),data=True))+list(G.out_edges(list(g.nodes()),data=True))
        for adj in adj_edges:
            if adj not in edges:
                temp = edges.copy()
                temp.append(adj)
                expand_res.append(temp)           
    return expand_res


def query_span(q_graph):
    qsq = qs()
    edges_list = list(q_graph.edges(data=True))
    edges_label = set([e[2]['label'] for e in edges_list])
    num_edges = len(edges_list)

    for i in range(num_edges):
        qsq.add_level([])
        qsq.add_edge(set())

    # 多增加一级存储整个查询q
    qsq.add_level([])
    # 第 0 level
    qsq.levels[0].append('root')

    # subgraphs = []
    # 第 1 level
    for i , lab in enumerate(edges_label):
        edge = [('?v1','?v2',{'label':lab})]
        sig = signature(edge)
        # qsq.levels[1].append({sig:edge})
        qsq.levels[1].append({sig:edge})
        qsq.edges[0].add((0,sig))

    # 第 2 level
    # unique_pid
    for i in range(2,num_edges):
        for my_dict in qsq.levels[i-1]:
            nsig = list(my_dict.keys())[0]
            n = list(my_dict.values())[0]
            match_glist = find_expand_subgraph(q_graph,n)
            for match_g in match_glist:
                sig = signature(match_g)
                if not qsq.levels[i]:
                    sig_ilevel= []
                else:
                    sig_ilevel = [list(ele.keys())[0] for ele in qsq.levels[i]]
                if sig not in sig_ilevel:
                    qsq.levels[i].append({sig:match_g})
                    qsq.edges[i-1].add((nsig,sig))
                else:
                    qsq.edges[i-1].add((nsig,sig))
        # print (f"第 {i-1} 级连边: {qsq.edges[i-1]}")
    sig = signature(edges_list)
    sig_ilevel = [list(ele.keys())[0] for ele in qsq.levels[num_edges-1]]
    qsq.levels[num_edges].append({sig:edges_list})
    for tp in sig_ilevel:
        qsq.edges[num_edges-1].add((tp,sig))
    # print (f"第 {num_edges-1} 级连边: {qsq.edges[num_edges-1]}")
    return qsq


def query_span2(q_graph):
    qsq = qs(q_graph.graph['name'])
    edges_list = list(q_graph.edges(data=True))
    edges_label = set([e[2]['label'] for e in edges_list])
    num_edges = len(edges_list)

    for i in range(num_edges):
        qsq.add_level([])

    # 多增加一级存储整个查询q
    qsq.add_level([])
    # 第 0 level
    qsq.levels[0].append('root')

    # subgraphs = []
    # 第 1 level
    for i , lab in enumerate(edges_label):
        edge = [('?v1','?v2',{'label':lab})]
        sig = signature(edge)
        # qsq.levels[1].append({sig:edge})
        qsq.levels[1].append((i,sig))
        qsq.add_edge(('root',i))
        qsq.add_fp({i:edge})
    

    # 第 2 level
    # unique_pid
    for i in range(2,num_edges):
        new_id = len(qsq.fps)
        for fp_sig in qsq.levels[i-1]:
            id_fp = fp_sig[0]
            match_glist = find_expand_subgraph(q_graph,qsq.fps[id_fp])
            for match_g in match_glist:
                sig = signature(match_g)
                if not qsq.levels[i]:
                    sig_ilevel= []
                else:
                    sig_ilevel = [ele[1] for ele in qsq.levels[i]]
                if sig not in sig_ilevel:
                    qsq.levels[i].append((new_id,sig))
                    qsq.add_fp({new_id:match_g})
                    qsq.add_edge((id_fp,new_id))
                    new_id +=1
                else:
                    sig_idx = sig_ilevel.index(sig)
                    i = qsq.levels[i][sig_idx][0]
                    qsq.add_edge((id_fp,i))
    sig = signature(edges_list)
    fp_id = len(qsq.fps)
    qsq.levels[num_edges].append((fp_id,sig))
    qsq.add_fp({fp_id:edges_list})
    for fp_sig in qsq.levels[num_edges-1]:
        qsq.add_edge((fp_sig[0],fp_id))

    print (f"连边: {qsq.edges[num_edges-1]}")
    return qsq


if __name__ == '__main__':
    # find_subgraph(0,0)
    # 例子
    q = nx.DiGraph()
    edges_list=[('?x','?y',{'label':'p1'}),('?z','?y',{'label':'p1'}),
               ('?x','?w',{'label':'p2'}),('?z','?w',{'label':'p2'})]
    # edges_list=[('?x','?y',{'label':'p1'}),('?y','?z',{'label':'p2'}),
    #            ('?z','?w',{'label':'p3'})]
    q.add_edges_from(edges_list)
    q.graph['name'] = 'q1'
    qsq = query_span(q)