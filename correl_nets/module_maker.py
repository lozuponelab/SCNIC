"""Make modules of observations based on cooccurence networks and collapse table"""


import os

import networkx as nx
import numpy as np

import general
from biom import load_table, Table
from scipy.stats import spearmanr, pearsonr


def paired_correlations_from_table(table, correl_method=spearmanr, p_adjust=general.bh_adjust):
    """Takes a biom table and finds correlations between all pairs of otus."""
    otus = table.ids(axis='observation')

    correls = list()

    # do all correlations
    for i in xrange(1, len(otus)):
        otu_i = table.data(otus[i], axis='observation')
        for j in xrange(i+1, len(otus)):
            otu_j = table.data(otus[j], axis='observation')
            correl = correl_method(otu_i, otu_j)
            correls.append([otus[i], otus[j], correl[0], correl[1]])

    # adjust p-value if desired
    if p_adjust is not None:
        p_adjusted = p_adjust([i[3] for i in correls])
        for i in xrange(len(correls)):
            correls[i].append(p_adjusted[i])

    header = ['feature1', 'feature2', 'r', 'p']
    if p_adjust is not None:
        header.append('adjusted_p')

    return correls, header


def make_modules(graph, k=3):
    """make modules with networkx k-clique communities and annotate network
    :type graph: networkx graph
    """
    cliques = list(nx.k_clique_communities(graph, k))
    cliques = [list(i) for i in cliques]
    for i, clique in enumerate(cliques):
        for node in clique:
            graph.node[node]['k_' + str(k)] = i
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
    in_module = set()
    modules = np.zeros((len(cliques), table.shape[1]))

    # for each clique merge values and print cliques to file
    with open("cliques.txt", 'w') as f:
        for i, clique in enumerate(cliques):
            in_module = in_module | set(clique)
            f.write(prefix+str(i)+'\t'+','.join(clique)+'\n')
            for feature in clique:
                modules[i] += table.data(feature, axis='observation')
    table.filter(in_module, axis='observation')

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
    return Table(new_table_matrix, new_table_obs, table.ids())


def module_maker(args):
    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr}
    p_methods = {'bh': general.bh_adjust, 'bon': general.bonferroni_adjust}
    correl_method = correl_methods[args.correl_method]
    if args.p_adjust is not None:
        p_adjust = p_methods[args.p_adjust]
    else:
        p_adjust = None

    # get features to be correlated and extract metadata
    table = load_table(args.input)
    metadata = general.get_metadata_from_table(table)
    print "Table loaded"

    # make new output directory and change to it
    if args.output is not None:
        os.makedirs(args.output)
        os.chdir(args.output)

    # convert to relative abundance and filter
    table_filt = general.filter_table(table, args.min_sample)
    print "Table filtered"

    # correlate feature
    correls, correl_header = paired_correlations_from_table(table_filt, correl_method, p_adjust)
    general.print_delimited('correls.txt', correls, correl_header)
    print "Features Correlated"

    # make correlation network
    net = general.correls_to_net(correls, conet=True, metadata=metadata, min_p=args.min_p)
    print "Network Generated"

    # make modules
    net, cliques = make_modules(net)
    print "Modules Formed"

    # print network
    nx.write_gml(net, 'conetwork.gml')

    # collapse modules
    coll_table = collapse_modules(table, cliques, args.prefix)
    print "Table Collapsed"

    # print new table
    coll_table.to_json('make_modules.py', open('collapsed.biom', 'w'))


if __name__ == '__main__':
    """main, takes argparser"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", help="location of input biom file")
    parser.add_argument("-o", "--output", help="output file location")
    parser.add_argument("-m", "--correl_method", help="correlation method", default="spearman")
    parser.add_argument("-a", "--p_adjust", help="p-value adjustment", default="bh")
    parser.add_argument("-s", "--min_sample", help="minimum number of samples present in", type=int)
    parser.add_argument("--prefix", help="prefix for module names in collapsed file", default="module_")
    parser.add_argument("-k", "--k_size", help="desired k-size to determine cliques", default=3, type=int)
    parser.add_argument("--min_p", help="minimum p-value to determine edges", default=.05, type=float)
    module_maker(parser.parse_args())
