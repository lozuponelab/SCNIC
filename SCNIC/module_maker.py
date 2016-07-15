"""Make modules of observations based on cooccurence networks and collapse table"""


# TODO: FIX BUG WHERE THINGS CAN BE IN MULTIPLE CLIQUES BUT GET LAST LABEL

from __future__ import division

import networkx as nx
import numpy as np
from biom import Table
import os


def make_modules(graph, k=3):
    """make modules with networkx k-clique communities and annotate network"""
    cliques = list(nx.k_clique_communities(graph, k))
    cliques = [list(set(i)) for i in cliques]
    for i, clique in enumerate(cliques):
        for node in clique:
            graph.node[node]['clique'] = i
    return graph, cliques


def make_modules_multik(graph, k=None):
    """make modules with networkx k-clique communities and annotate network"""
    if k is None:
        k = [2, 3, 4, 5, 6]
    communities = dict()
    for k_val in list(k):
        cliques = list(nx.k_clique_communities(graph, k_val))
        cliques = [list(i) for i in cliques]
        communities[k_val] = cliques
        for i, clique in enumerate(cliques):
            for node in clique:
                graph.node[node]['k_' + str(k_val)] = i
    return graph, {k: list(v) for k, v in communities.iteritems()}


def collapse_modules(table, cliques, prefix="module_"):
    """collapse created modules in a biom table"""
    #TODO: move writing cliques to file to it's own method
    table = table.copy()
    in_module = set()
    modules = np.zeros((len(cliques), table.shape[1]))

    # for each clique merge values and print cliques to file
    os.makedirs("modules")
    with open("cliques.txt", 'w') as f:
        for i, clique in enumerate(cliques):
            in_module = in_module | set(clique)
            f.write(prefix+str(i)+'\t'+'\t'.join([str(j) for j in clique])+'\n')
            for feature in clique:
                modules[i] += table.data(feature, axis="observation")
            module_table = table.filter(clique, axis='observation', inplace=False)
            module_table.to_json("modulemaker.py", open("modules/" + prefix + str(i) + ".biom", 'w'))

    table.filter(in_module, axis='observation', invert=True)

    # make new table
    new_table_matrix = np.concatenate((table.matrix_data.toarray(), modules))
    new_table_obs = list(table.ids(axis='observation')) + [prefix + str(i) for i in range(0, len(cliques))]
    return Table(new_table_matrix, new_table_obs, table.ids())


def collapse_modules_multik(table, cliques, prefix="module_"):
    """collapse created modules in a biom table"""
    in_module = set()
    modules = np.zeros((len(cliques), table.shape[1]))

    # for each clique merge values
    for i, clique in enumerate(cliques):
        in_module = in_module | set(clique)
        for feature in clique:
            modules[i] += table.data(feature, axis='observation')
    table.filter(in_module, axis='observation')

    # make new table
    new_table_matrix = np.concatenate((table.matrix_data.toarray(), modules))
    new_table_obs = list(table.ids(axis='observation')) + [prefix + str(i) for i in range(0, len(cliques))]
    return Table(new_table_matrix, new_table_obs, table.ids(), table.metadata(axis="observation"),
                 table.metadata(axis="sample"))
