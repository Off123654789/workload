import networkx as nx
import basic_graph_func
import random
import QSpan
import geoFrenquent
import query_parse
import metis_test as metis
import query_decomposition


def generate_test_data(vnum,label_num):
    # 创建一个空图
    G = nx.DiGraph()

    # 添加100个节点
    nodes = [str(i) for i in range(0, vnum)]
    edges_label1 = [f'p{i}' for i in range(0,6)]
    edges_label = [f'f{i}' for i in range(0,10)]
    G.add_nodes_from(nodes)
    nx.set_node_attributes(G,1,'weight')

    # 定义社区结构，这里将图分为两个社区
    community1 = [str(i) for i in range(0, int(vnum/2))]
    community2 = [str(i) for i in range(int(vnum/2), vnum)]

    # 添加社区内的边，减少社区内边的密度
    edges_community1 = [(i, j) for i in community1 for j in community1 if i < j and random.random() > 0.7]
    edges_community11 = [(i, j) for i in community1 for j in community1 if i > j and random.random() > 0.8]
    edges_community2 = [(i, j) for i in community2 for j in community2 if i < j and random.random() > 0.8]
    edges_community22 = [(i, j) for i in community2 for j in community2 if i > j and random.random() > 0.7]

    # 添加社区间的边，使得社区间的边较少
    edges_between_communities = [(i, j) for i in community1 for j in community2 if random.random() > 0.95]

    G.add_edges_from(edges_community1 + edges_community2 + edges_community11+edges_community22 + edges_between_communities)
    nx.set_edge_attributes(G,1,'weight')
    g = G.copy()
    for e in g.edges():
        if int(e[0]) > 10 and int(e[0]) < 20 and int(e[1]) > 10 and int(e[1]) < 25:
            G.edges[e[0],e[1]]['label'] = edges_label1[random.randint(0,5)]
        else:
            G.edges[e[0],e[1]]['label'] = edges_label[random.randint(0,9)]

    # 绘制图
    pos = nx.spring_layout(G)
    basic_graph_func.visualize_graph(G,pos,'test')
    return G


def generate_test_graphquery():
    edges_list1=[('?x','?y',{'label':'p1'}),('?z','?y',{'label':'p1'}),
               ('?x','?w',{'label':'p2'}),('?z','?w',{'label':'p2'})]

    edges_list2=[('?x','?y',{'label':'p1'}),('?y','?z',{'label':'p2'}),
               ('?z','?w',{'label':'p3'})]

    edges_list3=[('?x','?y',{'label':'p3'}),('?y','?z',{'label':'p4'}),
               ('?y','?w',{'label':'p2'})]
    
    # edges_list4=[('?x','?y',{'label':'p7'}),('?y','?z',{'label':'p9'}),
    #            ('?y','?w',{'label':'p2'})]
    query_dataset = [[edges_list1,6],[edges_list2,3],[edges_list3,5]]

    return query_dataset


def generate_test_spatialquery():
    # 输入数据：事务列表
    q1 = ('1','2','8')
    q2 = ('1','2','3','4','5')
    q3 = ('1','2','9')
    q4 = ('9','10','2','3','4')
    q5 = ('1','12','3','16','15')
    q6 = ('3','4','15','16','17')
    q7 = ('9','10','16','17','18')
    transactions = [q1,q2,q3,q4,q5,q6,q7,q1,q2,q3,q4,q5,q6,q7,q1,q2,q3,q4,q5,q6,q7]
    return transactions


def query_to_rdfquery(rdf_graph,graphFP,sup):
    results = []
    stat = dict()
    for i,fp in enumerate(graphFP):
        q = list(fp.values())[0][1]
        sparql_query = "SELECT *WHERE {"
        for e in q:
            sparql_query += f"{e[0]} <{e[2]['label']}> {e[1]} ."
        
        sparql_query += "}"
        res = query_parse.rdf_query(rdf_graph,sparql_query)
        # 直接获取结果中的值
        res_set = set()
        for row in res.bindings:
            for var_name, var_value in row.items():
                res_set.add(str(var_value))
        results.append([res_set,{'sup':sup[i],'num':len(res_set),'fp':list(fp.keys())[0]}])
        stat[list(fp.keys())[0]] = len(res)
    return results,stat


def allocation(rdfquery_res,geonode_FP,pos,g,g_part):
    G = nx.Graph()
    for i,node in enumerate(rdfquery_res):
        if node[0] :
            G.add_node(i,nodetype = 'fp',weight = node[1]['sup'],fp = node[1]['fp'])
    nodes = list(G.nodes())
    for i in nodes:
        for j in nodes:
            if i != j:
                res = rdfquery_res[i][0] & rdfquery_res[j][0]
                if res:
                    G.add_edge(i,j,weight=len(res))
    
    rdfnodes = set()
    for i in nodes:
        rdfnodes.update(rdfquery_res[i][0])
    res_rdfnodes  = set(g.nodes()) - rdfnodes
    
    geonodes = set()
    for i in geonode_FP:
        geonodes.update(i[0])
    
    # basic_graph_func.visualize_graph(g,pos,'rdfnodes_part',partition=g_part,rdfnodes= rdfnodes,geonodes = geonodes)
    newidx = len(rdfquery_res)
    delidx = set()

    for i in nodes:
        for j in range(len(geonode_FP)):
            res = rdfquery_res[i][0] & set(geonode_FP[j][0])  
            if res:
                G.add_node(newidx+j,weight = geonode_FP[j][1]['sup'])
                G.add_edge(i,newidx+j,weight=len(res))
                delidx.add(j)
    
    part1 = metis.metis_seg(G,2)
    allnodes = rdfquery_res+geonode_FP
    rgnodes = {0:set(),1:set()}
    for k,v in part1.items():
        n = allnodes[k][0]
        rgnodes[v].update(n)
     
    # res = {node: G_corase_part_dict[pi] for node, pi in G_part_dict.items()}
    # gs = g.subgraph(rgnodes[0]|rgnodes[1])
    # pos = nx.spring_layout(gs)
    basic_graph_func.visualize_graph(g,pos,'first_part',partition=g_part,rdfnodes=rgnodes[0],nodeshape1='*',geonodes=rgnodes[1],nodeshape2='o')
    G1 = nx.Graph()
    for k,v in rgnodes.items():
        G1.add_node(k,weight = len(v))
    residx = {i for i in range(len(geonode_FP))} - delidx
    
    if residx:
        for j in residx:
            G1.add_node(newidx+j,weight = len(geonode_FP[j][0]))
        newnodes = list(G1.nodes())
        for i in newnodes:
            for j in newnodes:
                if i != j:
                    if (i ==0 and j == 1) or (i ==1 and j == 0):
                        continue
                    else:
                        if i<2:
                            res = rgnodes[i] & set(allnodes[j][0])
                        else:
                            res = set(allnodes[i][0]) & set(allnodes[j][0])
                        if res:
                            G1.add_edge(i,j,weight = len(res))
    part2 = metis.metis_seg(G1,2,1)
    last_res = {0:set(),1:set()}
    for k,v in part2.items():
        if k in rgnodes.keys():
            last_res[v].update(rgnodes[k])
        else:
            n = allnodes[k][0]
            last_res[v].update(n)
    for k,v in last_res.items():
        pass
    leaf = set(g.nodes()) - (last_res[0] | last_res[1])
    basic_graph_func.visualize_graph(g,pos,'last_part',partition=part,rdfnodes =last_res[0],nodeshape1='o',geonodes = last_res[1],nodeshape2='o')
    
    
def test_test():
    # G = generate_test_data(30,10)
    # basic_graph_func.write_graph(G,f"workload_graph_partition/testG")
    G = basic_graph_func.load_graph("workload_graph_partition/data/testG")
    pos_G = nx.spring_layout(G)
    part = metis.metis_seg(G,3)
    # basic_graph_func.visualize_graph(G,pos,'testG',partition=part)

    query_data = generate_test_graphquery()
    for  i,q in enumerate(query_data):
        g = nx.DiGraph()
        g.add_edges_from(q[0])
        pos = nx.spring_layout(g)
        basic_graph_func.visualize_graph(g,pos,f'q{i}',size=(5,5))
    # 频繁子图
    graph_FP,sup = QSpan.FP_tree(query_data,4)
    rdf_graph = query_parse.networkx_to_rdf(G)
    rdfquery_res, rdfquery_stat = query_to_rdfquery(rdf_graph,graph_FP,sup)
    
    # 查询分解
    q = nx.DiGraph()
    edges_list=[('?x','?y',{'label':'p1'}),('?z','?y',{'label':'p1'}),
               ('?x','?w',{'label':'p2'}),('?z','?w',{'label':'p5'})]
    
    # edges_list=[('?x','?y',{'label':'p1'}),('?y','?z',{'label':'p2'}),
    #            ('?z','?w',{'label':'p3'})]
    q.add_edges_from(edges_list)
    pos = nx.spring_layout(q)
    basic_graph_func.visualize_graph(q,pos,f'new_q',size = (5,5))
    q.graph['name'] = 'q1'
    D = query_decomposition.Query_Decomposition(q,5,graph_FP,rdfquery_stat)
    
    # 绘图
    for i,iq in enumerate(D[0]):
        pos = nx.spring_layout(iq)
        basic_graph_func.visualize_graph(iq,pos,f'iq{i}',size = (5,5))
    
    # # 连接顺序
    # queries,cut_nodes = query_decomposition.left_join_order(D[0])
    
    # pos = nx.spring_layout(G)
    # 频繁节点集
    query_transactions = generate_test_spatialquery()
    geonode_FP = geoFrenquent.geo_FP(query_transactions,3)

    newpart = allocation(rdfquery_res,geonode_FP,pos_G,G,part)
    
    
if __name__ == '__main__':
    rdf_graph = query_parse.file_to_rdf()

    sparql_query = """SELECT * WHERE { ?person  <isPoliticianOf>	?country1.?person	<hasGender>	?gender.?person	<worksAt>	?place.}"""
    res = query_parse.rdf_query(rdf_graph,sparql_query)
    x = 0