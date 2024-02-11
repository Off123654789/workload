'''
Author: oufeifei
Date: 2023-11-27 09:42:51
LastEditors: oufeifei
LastEditTime: 2023-11-27 14:14:50
Description: 
'''
from rdflib import Graph,URIRef,Namespace
import basic_graph_func as bgf
from urllib.parse import quote

# 用于分配 URI 给节点
def get_node_uri(node,node_uri_mapping):  
    # ex = Namespace("http://yago3.org/")
    if node not in node_uri_mapping:
        node_uri_mapping[node] = URIRef(quote(node))
    return node_uri_mapping[node]


def query_run():
    # 创建一个 RDF 图
    g = Graph()

    # 用于将节点映射到 URI
    node_uri_mapping = {}
    # 从文本文件加载三元组
    file_path = '/home/geo00/DISK3/oufeifei/data/graph/G_tri_intype.ttl'

    with open(file_path, 'r',encoding='utf-8') as file:
        for line in file:
            triple = line.strip('.\n').split('\t')
            subject_uri = get_node_uri(triple[0],node_uri_mapping)
            predicate_uri = get_node_uri(triple[1],node_uri_mapping)
            object_uri = get_node_uri(triple[2],node_uri_mapping)
            # 添加三元组到图中
            g.add((subject_uri, predicate_uri, object_uri))
    # 查询
    q = """select * where { ?gn <givenNameOf> ?p. ?fn
<familyNameOf> ?p. ?p <type> ”scientist”; <bornInLocation> ?city; <hasDoctoralAdvisor> ?a. ?a <bornInLocation>
?city2. ?city <locatedIn> ”Switzerland”. ?city2 <locatedIn>
”Germany”. }
    """
    res = g.query(q)
    x=0


    

if __name__ == '__main__':
    query_run()