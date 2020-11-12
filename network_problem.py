
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random as rd

def initialize_variables():
    global num_blue
    global num_red
    got = pd.read_csv('stormofswords.csv')
    got = got.rename(columns={'Source': "source", 'Target': "target", 'Weight': "weight"})
    global network_got
    global nodes

    network_got= nx.from_pandas_edgelist(got, edge_attr='weight')
    nodes = network_got.nodes
    nodes = list(nodes)

    for node in nodes:
        network_got.nodes[node]['state'] = 1 if rd.random() < 0.90 else 0

    num_blue = 0
    num_red = 0
    for node in nodes:
        if (network_got.nodes[node]['state'] == 1):
            num_blue += 1
        else:
            num_red += 1

def get_useful_stats():
    global network_got

    print(nx.info(network_got))

    print('average clustering ',nx.average_clustering(network_got))
    cluster = nx.clustering(network_got)
    print('largest cluster ', max(cluster.values()))
    print('largest cluster node ',max(cluster, key=cluster.get))

    centrality = nx.betweenness_centrality(network_got)
    print('largest centrality ',max(centrality.values()))
    print('largest centrality value ',max(centrality,key=centrality.get))

def degree_distribution():
    global network_got
    global nodes
    degrees = []

    for node in nodes:
        degrees.append(network_got.degree[node])

    dd = np.zeros(max(degrees)+1)
    for degree in degrees:
        dd[degree]+=1

    dd = dd/np.sum(dd)

    plt.plot(dd)
    plt.xlabel('Degree')
    plt.ylabel('Probability of Degree x')
    plt.show()

def observe():
    global network_got
    global nodes

    # nx.draw(network_got, cmap=plt.cm.binary, vmin=0, vmax=1,
    #         node_color=[network_got.node[node][’state’] for node in  nodes, pos = network_got.pos)

    color_map = []
    for node in nodes:
        if (network_got.nodes[node]['state'] == 1):
            color_map.append('blue')
        else:
            color_map.append('red')
    nx.draw_shell(network_got, node_color=color_map, with_labels=True)
    plt.show()


# Press the green button in the gutter to run the script.

def update():
    global num_blue
    global num_red
    global network_got

    listener = rd.choice(nodes)
    speaker = rd.choice(list(network_got.adj[listener]))

    network_got.nodes[listener]['state'] = network_got.nodes[speaker]['state']

    num_blue = 0
    num_red = 0
    for node in nodes:
        if (network_got.nodes[node]['state'] == 1):
            num_blue+=1
        else:
            num_red+=1

if __name__ == '__main__':
    time_steps = 0
    initialize_variables()
    get_useful_stats()
    degree_distribution()
    observe()
    results = np.zeros(20)
    for i in range(20):
        time_steps = 0
        initialize_variables()

        while(num_red!=0 and num_blue!=0):
            update()
            time_steps += 1

        results[i]= time_steps
        print(i)

    average = np.sum(results)/20
    observe()
    print(results)
    print("average", average)