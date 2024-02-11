from annotated_types import Len
from matplotlib.pyplot import axis
from numpy import full
from pydantic import InstanceOf
import ray
import requests
from urllib import parse
import json
import query_plan
import networkx as nx
import query_decomposition as qd
import pandas as pd


                                                               

@ray.remote
def execute_query_and_filter(machine, queries):
    
    import pandas as pd
    class GstoreConnector:
        def __init__(self, ip = "192.168.12.0", port = '22', username = 'root', password = '123456', http_type='ghttp'):
            if ip == "localhost":
                self.serverIP = ip
            else:
                self.serverIP = ip
            self.serverPort = port
            self.base_url = "http://" + self.serverIP + ":" + str(self.serverPort) + "/"
            self.http_type = http_type
            if self.http_type == 'grpc':
                self.base_url += 'grpc/api'
            self.auth_params = {
                'username': username,
                'password': password
            }
            self.request = {"GET": self.get, "POST": self.post}

        def get(self, params, stream=False):
            if stream:
                return requests.get(self.base_url, params=params, stream=stream)
            else:
                return requests.get(self.base_url, params=parse.urlencode(params, quote_via=parse.quote), stream=stream).text

        def post(self, params, stream=False):
            if stream:
                return requests.post(self.base_url, json=params, stream=stream)
            else:
                return requests.post(self.base_url, json=params, stream=stream).text

        def save(self, filename, res_iter):
            with open(filename, 'wb') as fd:
                for chunk in res_iter.iter_content(4096):
                    fd.write(chunk)
            return

        def build(self, db_name, db_path, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'build',
                'db_name': db_name,
                'db_path': db_path
            }
            return self.request[request_type](query_params)

        def check(self, request_type='GET'):
            query_params = {
                'operation': 'check'
            }
            return self.request[request_type](query_params)

        def load(self, db_name, csr, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'load',
                'db_name': db_name,
                'csr': csr
            }
            return self.request[request_type](query_params)

        def monitor(self, db_name, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'monitor',
                'db_name': db_name
            }
            return self.request[request_type](query_params)

        def unload(self, db_name, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'unload',
                'db_name': db_name
            }
            return self.request[request_type](query_params)

        def drop(self, db_name, is_backup, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'drop',
                'db_name': db_name,
                'is_backup': is_backup
            }
            return self.request[request_type](query_params)

        def show(self, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'show'
            }
            return self.request[request_type](query_params)

        def usermanage(self, type, op_username, op_password, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'usermanage',
                'type': type,
                'op_username': op_username,
                'op_password': op_password
            }
            return self.request[request_type](query_params)

        def showuser(self, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'showuser'
            }
            return self.request[request_type](query_params)

        def userprivilegemanage(self, type, op_username, privileges, db_name, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'userprivilegemanage',
                'type': type,
                'op_username': op_username,
                'privileges': privileges,
                'db_name': db_name
            }
            return self.request[request_type](query_params)

        def backup(self, db_name, backup_path, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'backup',
                'db_name': db_name,
                'backup_path': backup_path
            }
            return self.request[request_type](query_params)

        def restore(self, db_name, backup_path, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'restore',
                'db_name': db_name,
                'backup_path': backup_path
            }
            return self.request[request_type](query_params)

        def query(self, db_name, format, sparql, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'query',
                'db_name': db_name,
                'format': format,
                'sparql': sparql
            }
            return self.request[request_type](query_params)

        def fquery(self, db_name, format, sparql, filename, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'fquery',
                'db_name': db_name,
                'format': format,
                'sparql': sparql
            }
            r = self.request[request_type](query_params, stream=True)
            self.save(filename, r)
            return

        def exportDB(self, db_name, db_path, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'export',
                'db_name': db_name,
                'db_path': db_path
            }
            return self.request[request_type](query_params)

        def login(self, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'login'
            }
            return self.request[request_type](query_params)

        def begin(self, db_name, isolevel, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'begin',
                'db_name': db_name,
                'isolevel': isolevel
            }
            return self.request[request_type](query_params)

        def tquery(self, db_name, tid, sparql, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'tquery',
                'db_name': db_name,
                'tid': tid,
                'sparql': sparql
            }
            return self.request[request_type](query_params)

        def commit(self, db_name, tid, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'commit',
                'db_name': db_name,
                'tid': tid
            }
            return self.request[request_type](query_params)

        def rollback(self, db_name, tid, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'rollback',
                'db_name': db_name,
                'tid': tid
            }
            return self.request[request_type](query_params)

        def getTransLog(self, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'txnlog'
            }
            return self.request[request_type](query_params)

        def checkpoint(self, db_name, request_type='GET'):
            query_params = {
                **self.auth_params,
                'operation': 'checkpoint',
                'db_name': db_name
            }
            return self.request[request_type](query_params)
    # 连接到 gStore 数据库
    connection =GstoreConnector(ip = machine[0],port = 9000,username = 'root',password = '123456',http_type='ghttp')
    connection.load(db_name = machine[1],csr=False)
    # 执行查询
    sparql_queries = queries
    num_query = len(sparql_queries)
    # spatial_query = queries[1]
    boundary_nodes = set(machine[2])
    cut_vars = set(machine[3])
    # 获取查询结果
    answers = [connection.query(db_name=machine[1], format='json', sparql=query) for query in sparql_queries]
    # 关闭数据库连接
    connection.unload(db_name = machine[1])
    # 处理查询结果
    multi_results = dict()
    for i,answer in enumerate(answers):
        try:
            answer = json.loads(answer)
            if answer['StatusCode'] != 0:
                raise ValueError(answer['StatusMsg'])
        except Exception as e:
            print(f"An error occurred: {e}")
            result = None  # 数据库不存在
        else:
            if answer['AnsNum'] == 0:
                result = None  # 答案集为0
            else:
                vars_ = answer['head']['vars']
                result = pd.DataFrame(columns = vars_)
                res = answer['results']['bindings']
                for ires in res:
                    temp = [v['value'] for v in ires.values()]
                    result.loc[len(result)] = temp
            multi_results[i] = result
    
    outer_multi_results = dict()
    inner_multi_results = dict()
    for i,qres in multi_results.items():
        vars_ = list(cut_vars & set(qres.columns) )
        # 创建一个布尔掩码
        mask = pd.Series([False] * len(qres))
        if vars_:
            mask = qres[vars_].isin(boundary_nodes).any(axis=1)
        # 应用掩码提取行
        if not qres[mask].empty:
            outer_multi_results[i] = qres[mask]
        if not qres[~mask].empty:
            inner_multi_results[i] = qres[~mask]

    del answers,sparql_queries
    
    for i,ires in inner_multi_results.items():
        
        for j,jres in multi_results.items():
            if i == j:
                continue
            ijres = ires.merge(jres)
            if not ijres.empty:
                # 创建一个布尔掩码
                vars_ = list(cut_vars & set(ijres.columns))
                if vars_:
                    mask = ijres[vars_].isin(boundary_nodes).any(axis=1)
                    # 应用掩码提取行
                    if not ijres[mask].empty:
                        outer_multi_results[i] = ijres[mask,list(ires.columns)]
                    ires = ijres[~mask,list(ires.columns)]
                else:
                    continue
            else:
                continue
            if ires.empty:
                break
    
    # 中间结果的全部连接，最终结果
    full_match = 0
    if len(inner_multi_results) == num_query:
        full_match = inner_multi_results[0]
        for i in range(1,num_query):
            full_match = full_match.merge(inner_multi_results[i])
           
    return outer_multi_results,full_match


def sparql_to_edges(sparql_query):
    # 获取大括号中的内容
    query_inside_brackets = sparql_query[sparql_query.find('{')+1:sparql_query.rfind('}')]

    # 分割获取每个三元组模式
    triples = query_inside_brackets.split('.')

    # 构建查询图
    edgeslist = []
    for triple in triples:
        triple = triple.strip()
        if triple:
            parts = triple.split()
            subject, predicate, object_ = parts[0], parts[1], parts[2]
            edgeslist.append((subject,object_,{'label':predicate}))
    return edgeslist


def graph_to_sparql(query_graph,spatial_query):
    comm_vars = list(set(query_graph.nodes()) & set(spatial_query.keys()))
    if comm_vars:
        edges = query_graph.edges(data = True)
        sparql_query = "SELECT *WHERE {"
        for e in edges:
            sparql_query += f"{e[0]} {e[2]['label']} {e[1]} ."
        if len(comm_vars) ==1:
            result_string = ", ".join(spatial_query[comm_vars[0]])
            sparql_query += f" FILTER({comm_vars[0]} IN ({result_string})"
        else:
            sparql_query += "FILTER("
            for v in comm_vars[0:-1]:
                result_string = ", ".join(spatial_query[v])
                sparql_query += f"{v} in ({result_string}) && "
            result_string = ", ".join(spatial_query[comm_vars[-1]])
            sparql_query += f"{comm_vars[-1]} in ({result_string})"
            
        sparql_query += ")}"
    else:
        edges = query_graph.edges(data = True)
        sparql_query = "SELECT *WHERE {"
        for e in edges:
            sparql_query += f"{e[0]} {e[2]['label']} {e[1]} ."      
        sparql_query += "}"
    return sparql_query


def simple_query(sparql_query,spatial_query=0):
    # ray.init(address="192.168.0.129:6379")
    ray.init(local_mode=True)
    spatial_query = {}
    ips = ('192.168.0.58','192.168.0.86')
    
    edgeslist = sparql_to_edges(sparql_query)
    query_graph = nx.DiGraph()
    query_graph.add_edges_from(edgeslist)
    queries = qd.decompose_star(query_graph)
    
    queries,cut_vars = qd.left_join_order(queries)
    boundary_nodes = ('x','w','s','y','z','b')
    
    # 分解后的全部星型查询 # 空间过滤
    queries_sparql = [graph_to_sparql(q,spatial_query) for q in queries]
    
    dbname = 'btest'
    queries = {(ip,dbname,boundary_nodes,cut_vars):(queries_sparql) for ip in ips}
    # 使用Ray并行执行查询和过滤
    # futures = [execute_query_and_filter.options(resources={f'node:{machine[0]}':1.0}).remote(machine, query) for machine, query in queries.items()]
    futures = [execute_query_and_filter.remote(machine, query) for machine, query in queries.items()]
    all_results = ray.get(futures)
    ray.shutdown()
    full_matches = []
    num_subquery = len(queries_sparql)
    semi_matches =  {key: True for key in range(num_subquery)} 
    for results in all_results:
        mul_res , full_match = results[0],results[1]
        if full_match != 0:
            full_matches.append(full_match)
        for i,res in mul_res.items():
            if isinstance(semi_matches[i],pd.DataFrame):
                semi_matches[i] = pd.concat([semi_matches[i],res], axis=0)
            else: 
                semi_matches[i] = res
        semi_matches[i].drop_duplicates()
        
    if len(semi_matches) == num_subquery:
        res = semi_matches[0]
        for i in range(1,num_subquery):
            res = res.merge(semi_matches[i])
        full_matches.append(res)
        
    if full_matches:
        num = len(full_matches)
        query_results = full_matches[0]
        for i in range(1,num):
            query_results = pd.concat([query_results,full_matches[i]], axis=0)
        query_results.drop_duplicates()
        print('---------query answers-----------')
        print(query_results)
    else:
        print('query answer num: 0 .')
    



def pattern_query(sparql_query,FP):
    pass
    
    
    

if __name__ == "__main__":
    sparql_query = '''
    select * where {
      ?x    <p3>    ?y.
      ?z    <p5>    ?y.
      ?w    ?m      ?x.
    }
    '''
    # sparql_query = '''
    # select * where {
    #     ?x    <p4>    ?y.
    #     ?x    <p5>    ?z.
    #     ?z    ?m      ?y.
    #     }
    # '''
    simple_query(sparql_query)