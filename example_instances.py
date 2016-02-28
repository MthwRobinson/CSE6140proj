import networkx as nx

ex_1 = nx.Graph()
for i in range(7):
    ex_1.add_node(i+1)
ex_1.add_edges_from([(1,2),(2,3),(3,4),(3,5),
                  (4,5),(4,6),(4,7),(5,6)])

ex_2 = nx.Graph()
for i in range(8):
    ex_2.add_node(i+1)
ex_2.add_edges_from([(1,2),(1,3),(2,3),(2,4),(2,5),
                     (4,5),(5,6),(5,3),(5,7),(3,7),(7,8)])
