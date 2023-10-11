import random

import networkx as nx


def create_waxman_graph(n, beta, alpha, path, seed=None, show=True, save=True, delay_weight=(10, 20)):
    G = nx.waxman_graph(n, beta=beta, alpha=alpha, seed=seed)
    for u, v in G.edges():
        G[u][v]['weight'] = random.randint(delay_weight[0], delay_weight[1])

    if save:
        nx.write_weighted_edgelist(G, path, delimiter=",")

        with open(path + ".nodes", 'w') as file:
            for i in range(len(G)):
                file.write(f"{i},{random.randint(30,100)},{random.randint(30,100)},{random.randint(30, 100)}\n")

    if show:
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True)
        import matplotlib.pyplot as plt
        # plt.show(dpi=500)
        plt.savefig(path + ".jpg", dpi=300)


if __name__ == '__main__':
    create_waxman_graph(30, beta=0.4, alpha=0.4, seed=12, delay_weight=(20, 50), path="src/graph/small-network.edgelist")
    create_waxman_graph(50, beta=0.1, alpha=1, seed=11, delay_weight = (20, 50), path="src/graph/medium-network.edgelist")
    create_waxman_graph(100, beta=0.3, alpha=0.2, seed=13, delay_weight=(20, 60), path="src/graph/large-network.edgelist")
