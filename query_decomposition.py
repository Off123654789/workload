import networkx as nx
import QSpan
import basic_graph_func as bg
from Isomorphism import signature
from itertools import chain, combinations
from networkx.algorithms.isomorphism import DiGraphMatcher
from functools import reduce


def powerset(iterable):
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(1,len(s) + 1))


def enumerate_subgraphs(G):
    # 创建一个简单的有向图
    # G = nx.DiGraph()
    # G.add_edges_from([(1, 2), (1, 3), (2, 4), (3, 4), (4, 5)])
    # # 绘制原始图形
    # pos = nx.spring_layout(G)
    # bg.visualize_graph(G,pos,'G')
    # 获取所有子图
    all_subgraphs = []
    for subset_nodes in powerset(G.nodes()):
        if len(subset_nodes) > 1:
            subgraph = G.subgraph(subset_nodes)
            if nx.is_weakly_connected(subgraph):  # 可以根据需要修改条件
                all_subgraphs.append(subgraph)
    # 绘制每个强连通分量（子图）
    # for i, subgraph in enumerate(all_subgraphs):
    #     pos = nx.spring_layout(subgraph)
        # bg.visualize_graph(subgraph,pos,f'subG{i}')
    return all_subgraphs


def remove_element_by_index(nested_list, indices_to_remove):
    # 使用循环删除指定索引处的元素
    for i, j in indices_to_remove:
        if i < len(nested_list) and j < len(nested_list[i]):
            del nested_list[i][j]
    return nested_list


def decompose_star(digraph):
    graph = digraph.to_undirected()
    groups = []
    label = 0
    while True:
        groups.append([])
        max_degree_node = max(dict(graph.degree()), key=lambda x: graph.degree[x])
        # Traverse nodes with degree greater than 1
        for neighbor in graph.neighbors(max_degree_node):
                groups[label].append(neighbor)
        if groups[label] :
            groups[label].append(max_degree_node)
        subg = graph.subgraph(groups[label])
        graph.remove_edges_from(subg.edges())
        label += 1
        if len(graph.edges()) ==0:
            break
    # Create subgraphs for each group
    subgraphs = []

    for nodes in groups:
        subgraphs.append(digraph.subgraph(nodes))

    return subgraphs


def diff_G(G1,G2):
    sub = G2
    res = set(G1.nodes()) - set(G2.nodes())
    if res:
        sub.add_nodes_from(res)
        g = nx.difference(G1,sub)
    else:
        g = nx.difference(G1,sub)
    # 复制边的标签信息
    for edge in g.edges(data=True):
        g[edge[0]][edge[1]]['label'] = G1[edge[0]][edge[1]]['label']
    g.remove_nodes_from(list(nx.isolates(g)))
    return g

def FP_isomorphic(q,FP):
    i = len(q.edges())
    q_sig = signature(list(q.edges(data=True)))
    iFP = {list(fp.values())[0][0]:list(fp.values())[0][1] for fp in FP if list(fp.keys())[0][0] == i}
    sigs = list(iFP.keys())
    if q_sig in sigs:
            edges_list = iFP[q_sig]
            p = nx.DiGraph()
            p.add_edges_from(edges_list)
            isomorphic = nx.is_isomorphic(p, q, edge_match=lambda x, y: x.get('label') == y.get('label'))
            if isomorphic:
                return (i,sigs.index(q_sig))
            else:
                return -1
            # matcher = DiGraphMatcher(G, q,edge_match=lambda edge1, edge2: edge1['label'] == edge2['label'])
    else:
        return -1


def DecomposeQuery(Q,FP,maxL,QS,QS_idx,DS,res,rdfquery_stat):
    G_union = nx.compose_all(QS)
    resq = diff_G(Q,G_union)
    subgraphs = enumerate_subgraphs(resq)
    for q in subgraphs:
        isomorphic = FP_isomorphic(q,FP)
        if isomorphic != -1:
            QS1 = QS.copy()
            QS1.append(q)
            QS1_idx  = QS_idx.copy()
            QS1_idx.append(isomorphic)
            G_union1 = nx.compose(q,G_union)
            equal = nx.is_isomorphic(G_union1, Q)
            if equal:
                D = (QS1,QS1_idx)
                DS.append(D)
            else:
                DecomposeQuery(Q,FP,maxL,QS=QS1,QS_idx=QS1_idx,DS=DS,res=res,rdfquery_stat=rdfquery_stat)
                if not DS:
                    resG_union = nx.compose_all(res[0])
                    resG = diff_G(Q,resG_union)
                    star_decomp = decompose_star(resG)
                    eval_res = len(star_decomp)
                    
                    resq = diff_G(Q,G_union1)
                    star_decomp = decompose_star(resq)
                    eval = len(star_decomp)
                    values = [rdfquery_stat[i] for i in QS1_idx]
                    s = reduce(lambda x,y:x*y, values)
                    if eval < eval_res:
                        t = [QS1,QS1_idx,s]
                        res[:] = t[:]
                    elif eval == eval_res:
                        if s < res[2]:
                            t = [QS1,QS1_idx,s]
                            res[:] = t[:]



def Query_Decomposition(Q,maxL,FP,rdfquery_stat):
    
    isomorphic = FP_isomorphic(Q,FP)
    if isomorphic != -1:
        return [[Q],[isomorphic]]
    else:
        MinCost = float('inf')
        D = []
        subgraphs = enumerate_subgraphs(Q)
        res = []
        for q in subgraphs:
            isomorphic = FP_isomorphic(q,FP)
            if isomorphic != -1:
                    DS = list()
                    QS = list()
                    QS_idx = list()
                    QS_idx.append(isomorphic)
                    QS.append(q)
                    if not res:
                        res = [QS,QS_idx,rdfquery_stat[isomorphic]]
                    DecomposeQuery(Q,FP,maxL,QS,QS_idx,DS,res,rdfquery_stat)
            if DS:          
                for D1 in DS:
                    currentCost = 1
                    for q in D1[1]:
                        currentCost *= rdfquery_stat[q]
                    if currentCost < MinCost:
                        D = D1
                        MinCost = currentCost
        if not D:
            resG_union = nx.compose_all(res[0])
            resG = diff_G(Q,resG_union)
            star_decomp = decompose_star(resG)
            res[0].extend(star_decomp)
            D = res
        return D



def left_join_order(queries):
    stat = {query: len(query.edges()) for query in queries}
    visited_queries = [max(stat,key = lambda x: stat[x])]
    visited_queries_nodes = set(visited_queries[0].nodes())
    
    visited = {query:0 for query in queries}
    visited[visited_queries[0]] = 1
    cut_nodes = set()
    while True:
        next_query = 0
        for query in queries:
            if visited[query] == 0:
                max_gain = 1
                comm_var = set(query.nodes()) & (visited_queries_nodes)
                gain = len(comm_var)
                if not (len(comm_var) < max_gain):
                    next_query = query
                    max_gain = gain
        if next_query == 0:
            break
        else:
            cut_nodes |= comm_var
            visited_queries_nodes |= set(next_query.nodes)
            visited_queries.append(next_query)
            visited[next_query] = 1
    cut_nodes = tuple([i[1:] for i in cut_nodes if i.startswith('?')])
    return visited_queries, cut_nodes


    
if __name__ == '__main__':
    # find_subgraph(0,0)
    # # 例子
    q = nx.DiGraph()
    edges_list=[('?x','?y',{'label':'p1'}),('?z','?y',{'label':'p1'}),
               ('?x','?w',{'label':'p2'}),('?z','?w',{'label':'p2'})]
    # edges_list=[('?x','?y',{'label':'p1'}),('?y','?z',{'label':'p2'}),
    #            ('?z','?w',{'label':'p3'})]
    q.add_edges_from(edges_list)
    q.graph['name'] = 'q1'
    D = Query_Decomposition(q,10)
    print(D)