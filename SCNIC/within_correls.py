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
import correlation_analysis as ca
import distance_analysis as da
import module_maker as mm


def within_correls(args):
    print args

    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr, 'sparcc': None, 'jaccard': jaccard,
                      'cscore': da.cscore, 'braycurtis': braycurtis, 'euclidean': euclidean, 'kendall': kendalltau,
                      'canberra': canberra}
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
            correls, correl_header = ca.paired_correlations_from_table_with_outlier_removal(table_filt, good_samples,
                                                                                            correl_method, p_adjust)
        else:
            print "Correlating with " + args.correl_method
            # correlate feature
            correls, correl_header = ca.paired_correlations_from_table(table_filt, correl_method, p_adjust)
    elif correl_method in [jaccard, braycurtis, euclidean]:
        print "Computing pairwise distances with " + args.correl_method
        # compute distances
        correls, correl_header = da.paired_distances_from_table(table_filt, correl_method)
        if args.min_p is not None:
            correls, correl_header = da.boostrap_distance_vals(correls, correl_header, p_adjust, bootstraps=1000,
                                                               procs=args.procs)
    else:
        print "Correlating using sparcc"

        # convert to pandas dataframe
        df = general.biom_to_df(table_filt)

        # calculate correlations
        cor, cov = ps.basis_corr(df, oprint=False)

        if args.min_p is None:
            correls, correl_header = ca.square_to_correls(cor)
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
    net, cliques = mm.make_modules(net)
    print "Modules Formed"
    print "number of modules: " + str(len(cliques))
    print ""

    # TODO: Add clique summary

    # print network
    nx.write_gml(net, 'conetwork.gml')

    # collapse modules
    coll_table = mm.collapse_modules(table, cliques, args.prefix)
    print "Table Collapsed"

    # print new table
    coll_table.to_json('make_modules.py', open('collapsed.biom', 'w'))
