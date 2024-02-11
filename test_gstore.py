# coding: utf-8
# zhangxiaoyang.hit#gmail.com
# github.com/zhangxiaoyang

import json
from GstoreConnector import GstoreConnector
import pprint
import time
import ray




# 定义一个远程执行函数，用于读取工作节点上的数据
@ray.remote
def read_data_from_disk(path):
    # 在这里执行读取数据的操作
    with open(path, 'r') as file:
        data = file.read()
    return data

def readfile():
    # 初始化Ray
    # ray.init()
    # 磁盘路径，这是工作节点上的文件路径
    disk_path = '/home/off/boundary.txt'

    # 在工作节点上调用远程函数来读取数据
    result_ref = read_data_from_disk.options(resources={f'node:192.168.0.128':1.0}).remote(disk_path)

    # 获取结果
    result = ray.get(result_ref)
    print(result)


if __name__ == '__main__':
    
    
    # ray.init(local_mode=True)
    t1 = time.time()
    db_name = 'total'
    # reGstoreConnector = ray.remote(GstoreConnector)
    gc = GstoreConnector("lzx", 9000, "root", "123456",http_type='ghttp')
    #    gc.build(db_name, "/home/off/data/G_tri_notype.n3")
    # gc.drop(db_name,False)
   
#     sparql = '''select * where {
# ?c1	<dealsWith>	?c2.
# ?v1	<isConnectedTo>	?v2.
# ?v1	<isLocatedIn>	?c1.
# ?v2	<isLocatedIn>	?c2.
#     }'''
    sparql = '''select * where {
        ?person	<hasGender>	?gender.
        ?person	<hasWonPrize>	?prize.
        ?person	<graduatedFrom>	?university.
        ?worker	<worksAt>	?university.
        ?university	<isLocatedIn>	?place.
    }'''
    gc.load(db_name,csr=False)
#    answer = gc.query(db_name,'json',sparql)
    
    try:
        answer = gc.query(db_name,'json',sparql)
        answer = json.loads(answer)
        if answer['StatusCode'] != 0:
            raise ValueError(answer['StatusMsg'])
    except Exception as e:
        print(f"An error occurred: {e}")
    else:
        vars = tuple(answer['head']['vars'])
        if answer['AnsNum'] == 0:
            results = {vars:[]}
        else:
            res = answer['results']['bindings']
        answer = []
        for ires in res:
            temp = []
            for k,v in ires.items():
                temp.append(v['value'])
            answer.append(temp)
        results = {vars:answer}
    t2 = time.time()
    t = t2-t1
    print(f'time cost: {t}')
    # pprint.pprint (results)
    # ray.shutdown()