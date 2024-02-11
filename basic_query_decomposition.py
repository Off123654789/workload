import networkx as nx


def decompose_star(digraph):
    graph = digraph.to_undirected()
    groups = []
    label = 0
    while True:
        groups.append([])
        max_degree_node = max(dict(graph.degree()), key=lambda x: graph.degree[x])
        visited = {node: 0 for node in graph.nodes}
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

if __name__ == '__main__':
    # 创建一个示例图
    example_digraph = nx.DiGraph()
    example_digraph.add_edges_from([(2, 1),(2,7), (2, 3),(3,6), (3, 4),(4,5)])

    subgraphs = decompose_graph(example_digraph)
