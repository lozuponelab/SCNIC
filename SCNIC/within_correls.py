# TODO: Add parameters log output file to output folder
# TODO: Testing correls as list vs as pandas dataframe for speed

from __future__ import division

import os
import sys
import shutil

import networkx as nx
import numpy as np
import pysurvey as ps

from biom import load_table, Table
from biom.exception import UnknownIDError
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.spatial.distance import jaccard, braycurtis, euclidean, canberra
from operator import itemgetter

import general
import sparcc_correlations as sc
import correlation_analysis as ca
import distance_analysis as da
import module_maker as mm


def within_correls(args):
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

    # check if output directory already exists and if it does delete it
    if args.force:
        shutil.rmtree(args.output, ignore_errors=True)

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
        # use outlier removal
        if args.outlier_removal:
            print "Correlating with outlier removal."
            # remove outlier observations
            # first attempt with just looking at individual otu's
            good_samples = general.remove_outliers(table_filt)
            print "Outliers removed: " + str(len(good_samples)) + " observations"
            print ""
            correls = ca.paired_correlations_from_table_with_outlier_removal(table_filt, good_samples,
                                                                                            correl_method, p_adjust)
        # calculate correlations normally
        else:
            print "Correlating with " + args.correl_method
            # correlate feature
            correls = ca.paired_correlations_from_table(table_filt, correl_method, p_adjust)

    # calculate distances
    elif correl_method in [jaccard, braycurtis, euclidean]:
        table_filt_rar = table_filt.subsample(args.rarefaction_level)
        print "Computing pairwise distances with " + args.correl_method
        if args.min_p is not None:
            correls = da.bootstrap_distance_vals(table_filt_rar, args.correl_method, nprocs=args.procs,
                                                 bootstraps=args.bootstraps, p_adjust=p_adjust)
        else:
            correls = da.paired_distances_from_table(table_filt_rar, args.correl_method)
    else:
        print "Correlating using sparcc"

        # convert to pandas dataframe
        df = general.biom_to_df(table_filt)

        # calculate correlations
        cor, cov = ps.basis_corr(df, oprint=False)

        if args.min_p is None:
            correls = sc.square_to_correls(cor)
        else:
            print "Bootsrapping Correlations"
            correls = sc.sparcc_pvals_multi(df, cor, p_adjust, procs=args.procs,
                                                        bootstraps=args.bootstraps)

    correls.sort([correls.columns[-1], correls.columns[2]], inplace=True)
    correls.to_csv(open('correls.txt', 'w'), sep='\t', index=False)

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
