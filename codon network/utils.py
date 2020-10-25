import numpy as np
import networkx as nx
from itertools import product, permutations
from scipy.spatial.distance import pdist
from scipy.cluster import hierarchy
import matplotlib.pyplot as plt

codon2AA = {'UUU':'Phe','UUC':'Phe','UUA':'Leu','UUG':'Leu',
            'UCU':'Ser','UCC':'Ser','UCA':'Ser','UCG':'Ser',
            'UAU':'Tyr','UAC':'Tyr','UAA':'Stop','UAG':'Stop',
            'UGU':'Cys','UGC':'Cys','UGA':'Stop','UGG':'Trp',
            'CUU':'Leu','CUC':'Leu','CUA':'Leu','CUG':'Leu',
            'CCU':'Pro','CCC':'Pro','CCA':'Pro','CCG':'Pro',
            'CAU':'His','CAC':'His','CAA':'Gln','CAG':'Gln',
            'CGU':'Arg','CGC':'Arg','CGA':'Arg','CGG':'Arg',
            'AUU':'Ile','AUC':'Ile','AUA':'Ile','AUG':'Met',
            'ACU':'Thr','ACC':'Thr','ACA':'Thr','ACG':'Thr',
            'AAU':'Asn','AAC':'Asn','AAA':'Lys','AAG':'Lys',
            'AGU':'Ser','AGC':'Ser','AGA':'Arg','AGG':'Arg',
            'GUU':'Val','GUC':'Val','GUA':'Val','GUG':'Val',
            'GCU':'Ala','GCC':'Ala','GCA':'Ala','GCG':'Ala',
            'GAU':'Asp','GAC':'Asp','GAA':'Glu','GAG':'Glu',
            'GGU':'Gly','GGC':'Gly','GGA':'Gly','GGG':'Gly'}


rate = {'AG':0.175, 'UC':0.175, 'GA':0.255, 'CU':0.255,
        'AU':0.285, 'UA':0.285, 'AC':0.047, 'UG':0.047,
        'GC':0.041, 'CG':0.041, 'GU':0.141, 'CA':0.141}


def createGraph(codon2AA, rate):
    G = nx.DiGraph()
    perms = [''.join(p) for p in product('AUGC',repeat=3)]
    G.add_nodes_from(perms)
    edge_list = []
    for i in perms:
        for j in perms:
            num = 0
            for bs1, bs2 in zip(i,j):
                if bs1 == bs2:
                    num += 1
                else:
                    mutation = "".join((bs1,bs2))
            if num == 2:
                edge_list.append((i,j,rate[mutation]))
    G.add_weighted_edges_from(edge_list)
    for node in G.nodes:
        G.nodes[node]['pro'] = codon2AA[node]
    nx.write_graphml(G, 'codon_graph.xml')
    return G


def randomWalk(G, path_length, start=None):
    if start:
        path = [start]
    else:
        path = np.random.choice(list(G.nodes))
    while len(path) < path_length:
        current_node = path[-1]
        node_list = list(G.neighbors(current_node))
        probs = [G.edges[(current_node,neighbor)]['weight'] for neighbor in G.neighbors(current_node)]
        total = sum(probs)
        for i in range(len(probs)):
            probs[i] /= total
        next_node = np.random.choice(node_list, p=probs)
        path.append(next_node)
    return path


def build_Deepwalk_corpus(G, num_paths, path_length):
    walks = []
    nodes = list(G.nodes)
    for cnt in range(num_paths):
        np.random.shuffle(nodes)
        for node in nodes:
            walks.append(randomWalk(G, path_length, start=node))
    return walks


def compute_Codon_distance(nodes, embedding, metric='euclidean'):
    dis_dict = {}
    node_pairs = list(permutations(nodes, 2))
    for (n1, n2) in node_pairs:
        z1 = embedding[nodes.index(n1)]
        z2 = embedding[nodes.index(n2)]
        dis_dict[(n1, n2)] = pdist(np.vstack([z1,z2]),metric).item()
    return dis_dict


def generate_Pro2codon(codon2AA):
    pro2codon = {}
    for codon, pro in codon2AA.items():
        if pro in pro2codon.keys():
            pro2codon[pro].append(codon)
        else:
            pro2codon[pro] = [codon]
    return pro2codon


def compute_Pro_distance(codon_dist, pro2codon):
    proteins = list(pro2codon.keys())
    pro_pairs = list(permutations(proteins, 2))
    pro_dist = {}
    for (pro1, pro2) in pro_pairs:
        dist = 0.0
        for node1 in pro2codon[pro1]:
            for node2 in pro2codon[pro2]:
                dist += codon_dist[(node1, node2)]
        pro_dist[(pro1, pro2)] = dist/len(pro2codon[pro1])/len(pro2codon[pro2])
    return pro_dist


def dict2matrix(dict_):
    keys = list(dict_.keys())
    proteins = list(np.unique(keys))
    n = len(proteins)
    adj = np.zeros((n, n))
    for (pro1, pro2) in keys:
        adj[proteins.index(pro1)][proteins.index(pro2)] = dict_[(pro1, pro2)]
    return np.matrix(adj)


def plot(dist_matrix, labels):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    linkage = hierarchy.ward(dist_matrix)
    dendro = hierarchy.dendrogram(
        linkage, labels=labels, ax=ax1, leaf_rotation=90
    )
    dendro_idx = np.arange(0, len(dendro['ivl']))
    ax2.imshow(dist_matrix[dendro['leaves'], :][:, dendro['leaves']])
    ax2.set_xticks(dendro_idx)
    ax2.set_yticks(dendro_idx)
    ax2.set_xticklabels(dendro['ivl'], rotation='vertical')
    ax2.set_yticklabels(dendro['ivl'])
    fig.tight_layout()
    plt.savefig('res.png',dpi=300)
    plt.show()
    