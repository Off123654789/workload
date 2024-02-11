import numpy


def left_join_order(queries):
    stat = {query: len(query.edges()) for query in queries}
    visited_queries = [max(stat,key = lambda x: stat[x])]
    visited_queries_nodes = set(visited_queries[0].nodes())
    cut_nodes = set()
    visited = {query:0 for query in queries}
    while True:
        for query in queries:
            next_query = 0
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
    return visited_queries , cut_nodes

              
            