from functools import reduce
import networkx as nx
import json
import os

thelt = 11


def write_dict_file(dictname: dict(), filename: str):
    """
    变量dictname 写入文件filename
    """
    w = open(filename, "w", encoding="utf-8")
    data = json.dumps(dictname, indent=1)
    w.write(data)
    w.close()
    print(f"{dictname}写入文件已完成")


def read_file_dict(filename: str):
    """
    input:
        filename: 读取文件
    output:
        dictname: 字典dictname
    """
    dictname = dict()
    fexist = os.path.isfile(filename)
    assert fexist, f"Error: {filename}文件不存在"
    f = open(filename, "r", encoding="utf-8")
    data = f.read()
    dictname = json.loads(data)
    f.close()
    print(f"{filename}读取到字典已完成")
    return dictname

label_map_id = read_file_dict('/home/off/code/workload_graph_partition/data/label_map.json')

def data_label():
    f = open('/home/off/code/workload_graph_partition/data/G_tri_notype.n3',
             'r', encoding='utf-8')
    data = dict()
    i = 3
    for line in f:
        if line.startswith('<'):
            sLine = line.rstrip('.\n').split('\t')
            data[sLine[1]] = i
            i += 1
    write_dict_file(data,'/home/off/code/workload_graph_partition/data/label_map.json')
            
    
def id_map(x):
    # data = {f'p{i}':i+1 for i in range(10)}
    return label_map_id[x]

def in_fac(id_edge,i):
    return (id_edge+i) % thelt


def out_fac(id_edge,i):
    return (id_edge-i) % thelt


def signature(edges_list):
    
    q_graph = nx.DiGraph()
    q_graph.add_edges_from(edges_list)
    nodes = q_graph.nodes()
    res_sig = 1
    for node in nodes:
        in_edges = list(q_graph.in_edges(node))
        out_edges = list(q_graph.out_edges(node))

        if in_edges:
            in_edges = [q_graph.get_edge_data(*e)['label'] for e in in_edges]
            in_edges = sorted(list(map(id_map,in_edges)))
            inFac  = list(map(in_fac,in_edges,range(1,len(in_edges)+1)))
            inFac_result = reduce(lambda x, y: x * y, inFac)
        else:
            inFac_result = 1

        if  out_edges:
            out_edges = [q_graph.get_edge_data(*e)['label'] for e in out_edges]
            out_edges = sorted(list(map(id_map,out_edges)))
            outFac = list(map(out_fac,out_edges,range(1,len(out_edges)+1)))
            outFac_result = reduce(lambda x, y: x * y, outFac)
        else:
            outFac_result = 1

        res_sig *= inFac_result * outFac_result

    return res_sig


if __name__ == '__main__':
    data_label()
    # q = nx.DiGraph()
    # edges_list=[('?x','?y',{'label':'p1'}),('?z','?y',{'label':'p1'}),
    #            ('?x','?w',{'label':'p2'}),('?z','?w',{'label':'p2'})]
    # sig = signature(edges_list)
    # print(f'signature of q: {sig}')