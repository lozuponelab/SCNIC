"""Make modules of observations based on cooccurence networks and collapse table"""


# TODO: Add parameters log output file to output folder

from __future__ import division

import os
import sys

import networkx as nx
import numpy as np
import pysurvey as ps

import general
from biom import load_table, Table
from biom.exception import UnknownIDError
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.spatial.distance import jaccard, braycurtis, euclidean, canberra
from operator import itemgetter
from sparcc_correlations import sparcc_pvals_multi


def paired_correlations_from_table(table, correl_method=spearmanr, p_adjust=general.bh_adjust):
    """Takes a biom table and finds correlations between all pairs of otus."""
    correls = list()

    for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in table.iter_pairwise(axis='observation'):
        correl = correl_method(data_i, data_j)
        correls.append([str(otu_i), str(otu_j), correl[0], correl[1]])

    # adjust p-value if desired
    if p_adjust is not None:
        p_adjusted = p_adjust([i[3] for i in correls])
        for i in xrange(len(correls)):
            correls[i].append(p_adjusted[i])

    header = ['feature1', 'feature2', 'r', 'p']
    if p_adjust is not None:
        header.append('adjusted_p')

    return correls, header


def paired_distances_from_table(table, dist_metric=braycurtis, rar=1000):
    """Takes a biom table and finds distances between all pairs of otus"""
    dists = list()
    rar_table = table.subsample(rar)

    for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in rar_table.iter_pairwise(axis='observation'):
        dist = dist_metric(data_i, data_j)
        dists.append([str(otu_i), str(otu_j), dist])

    header = ['feature1', 'feature2', 'dist']

    return dists, header


def paired_correlations_from_table_with_outlier_removal(table, good_samples, min_keep=10, correl_method=spearmanr,
                                                        p_adjust=general.bh_adjust):
    """Takes a biom table and finds correlations between all pairs of otus."""
    correls = list()

    for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in table.iter_pairwise(axis='observation'):
        samp_union = np.union1d(good_samples[otu_i], good_samples[otu_j])
        # remove zero zero points
        # samp_union = [ind for i, ind in enumerate(samp_union) if data_i[i]!=0 and data_j[i]!=0]
        if len(samp_union) > min_keep:
            correl = correl_method(data_i[samp_union], data_j[samp_union])
            correls.append([str(otu_i), str(otu_j), correl[0], correl[1]])

    # adjust p-value if desired
    if p_adjust is not None:
        p_adjusted = p_adjust([i[3] for i in correls])
        for i in xrange(len(correls)):
            correls[i].append(p_adjusted[i])

    header = ['feature1', 'feature2', 'r', 'p']
    if p_adjust is not None:
        header.append('adjusted_p')

    return correls, header


def cscore(u, v):
    """calculates standardized c-score according to https://en.wikipedia.org/wiki/Checkerboard_score"""
    u = u.astype(bool)
    v = v.astype(bool)
    r_u = np.sum(u)
    r_v = np.sum(v)
    s_uv = np.sum(np.logical_and(u, v))
    return (r_u-s_uv)*(r_v-s_uv)/(r_u+r_v-s_uv)


def square_to_correls(cor):
    # generate correls array
    correls = list()
    for i in xrange(len(cor.index)):
        for j in xrange(i + 1, len(cor.index)):
            correls.append([cor.index[i], cor.index[j], cor.iat[i, j]])
    header = ['feature1', 'feature2', 'r']
    return correls, header


def make_modules(graph, k=3):
    """make modules with networkx k-clique communities and annotate network
    :type graph: networkx graph
    """
    cliques = list(nx.k_clique_communities(graph, k))
    cliques = [list(i) for i in cliques]
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
    table = table.copy()
    in_module = set()
    modules = np.zeros((len(cliques), table.shape[1]))

    # for each clique merge values and print cliques to file
    with open("cliques.txt", 'w') as f:
        for i, clique in enumerate(cliques):
            in_module = in_module | set(clique)
            f.write(prefix+str(i)+'\t'+'\t'.join([str(j) for j in clique])+'\n')
            for feature in clique:
                try:
                    modules[i] += table.data(feature, axis="observation")
                except UnknownIDError:
                    print feature
                    print feature in table.ids(axis="observation")
                    sys.exit("exit with UnknownIDError")
                except IndexError:
                    print table.data(feature, axis="observation").shape
                    print modules.shape
                    print i
                    print len(cliques)
                    sys.exit("exit with IndexError")
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


def module_maker(args):
    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr, 'sparcc': None, 'jaccard': jaccard, 'cscore': cscore,
                      'braycurtis': braycurtis, 'euclidean': euclidean, 'kendall': kendalltau, 'canberra': canberra}
    p_methods = {'bh': general.bh_adjust, 'bon': general.bonferroni_adjust}
    correl_method = correl_methods[args.correl_method.lower()]
    if args.p_adjust is not None:
        p_adjust = p_methods[args.p_adjust]
    else:
        p_adjust = None

    # get features to be correlated and extract metadata
    table = load_table(args.input)
    metadata = general.get_metadata_from_table(table)
    print "Table loaded: " + str(table.shape[0]) + " observations"
    print ""

    # make new output directory and change to it
    if args.output is not None:
        os.makedirs(args.output)
        os.chdir(args.output)

    # convert to relative abundance and filter
    if args.min_sample is not None:
        table_filt = general.filter_table(table, args.min_sample)
        print "Table filtered: " + str(table_filt.shape[0]) + " observations"
        print ""
    else:
        table_filt = table

    # correlate feature
    if correl_method in [spearmanr, pearsonr, kendalltau]:
        if args.outlier_removal:
            print "Correlating with outlier removal."
            # remove outlier observations
            # first attempt with just looking at individual otu's
            good_samples = general.remove_outliers(table_filt)
            print "Outliers removed: " + str(len(good_samples)) + " observations"
            print ""
            correls, correl_header = paired_correlations_from_table_with_outlier_removal(table_filt, good_samples,
                                                                                         correl_method, p_adjust)
        else:
            print "Correlating with " + args.correl_method
            # correlate feature
            correls, correl_header = paired_correlations_from_table(table_filt, correl_method, p_adjust)
    elif correl_method in [jaccard, braycurtis, euclidean]:
        print "Computing pairwise distances with " + args.correl_method
        # compute distances
        correls, correl_header = paired_distances_from_table(table_filt, correl_method)
        if args.min_p is not None:
            correls, correl_header = boostrap_distance_vals(correls, correl_header, p_adjust, bootstraps=1000, procs=args.procs)
    else:
        print "Correlating using sparcc"

        # convert to pandas dataframe
        df = general.biom_to_df(table_filt)

        # calculate correlations
        cor, cov = ps.basis_corr(df, oprint=False)

        if args.min_p is None:
            correls, correl_header = square_to_correls(cor)
        else:
            print "Bootsrapping Correlations"
            correls, correl_header = sparcc_pvals_multi(df, cor, p_adjust, procs=args.procs,
                                                        bootstraps=args.bootstraps)

    correls.sort(key=itemgetter(-1))
    general.print_delimited('correls.txt', correls, correl_header)

    print "Features Correlated"

    # make correlation network
    net = general.correls_to_net(correls, conet=True, metadata=metadata, min_p=args.min_p, min_r=args.min_r)
    print "Network Generated"
    print "number of nodes: " + str(net.number_of_nodes())
    print "number of edges: " + str(net.number_of_edges())
    print ""

    # make modules
    net, cliques = make_modules(net)
    print "Modules Formed"
    print "number of modules: " + str(len(cliques))
    print ""

    # TODO: Add clique summary

    # print network
    nx.write_gml(net, 'conetwork.gml')

    # collapse modules
    coll_table = collapse_modules(table, cliques, args.prefix)
    print "Table Collapsed"

    # print new table
    coll_table.to_json('make_modules.py', open('collapsed.biom', 'w'))
