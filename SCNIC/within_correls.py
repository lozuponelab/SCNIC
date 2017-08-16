# TODO: Testing correls as list vs as pandas dataframe for speed

from __future__ import division

import os
import shutil
from functools import partial

import networkx as nx
import pandas as pd
import numpy as np

from biom import load_table
from sparcc_fast import sparcc_correlation, sparcc_correlation_w_bootstraps
from sparcc_fast import sparcc_functions
from sparcc_fast.sparcc_functions import basis_corr
from sparcc_fast.utils import df_to_correls
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.spatial.distance import jaccard, braycurtis, euclidean, canberra, squareform, pdist

import general
import correlation_analysis as ca
import distance_analysis as da
import module_maker as mm


def within_correls(args):
    logger = general.Logger("SCNIC_log.txt")
    logger["SCNIC analysis type"] = "within"

    # sanity check args
    if args.min_r is not None and args.min_p is not None:
        raise ValueError("arguments min_p and min_r may not be used concurrently")

    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr, 'kendall': kendalltau, 'sparcc': None}
    p_methods = {'bh': general.bh_adjust, 'bon': general.bonferroni_adjust}
    correl_method = correl_methods[args.correl_method.lower()]
    if args.p_adjust is not None:
        p_adjust = p_methods[args.p_adjust]
    else:
        p_adjust = None

    # get features to be correlated and extract metadata
    table = load_table(args.input)
    logger["input table"] = args.input
    if args.verbose:
        print "Table loaded: " + str(table.shape[0]) + " observations"
        print ""
    logger["number of samples in input table"] = table.shape[1]
    logger["number of observations in input table"] = table.shape[0]

    # check if output directory already exists and if it does delete it
    # TODO: change this so it only deletes things used by SCNIC within or overwrites
    if args.force and args.output is not None:
        shutil.rmtree(args.output, ignore_errors=True)

    # make new output directory and change to it
    if args.output is not None:
        os.makedirs(args.output)
        os.chdir(args.output)
        logger["output directory"] = args.output

    # filter
    if args.sparcc_filter is True:
        table_filt = general.sparcc_paper_filter(table)
        if args.verbose:
            print "Table filtered: " + str(table_filt.shape[0]) + " observations"
            print ""
        logger["sparcc paper filter"] = True
        logger["number of observations present after filter"] = table_filt.shape[0]
    elif args.min_sample is not None:
        table_filt = general.filter_table(table, args.min_sample)
        if args.verbose:
            print "Table filtered: " + str(table_filt.shape[0]) + " observations"
            print ""
        logger["min samples present"] = args.min_sample
        logger["number of observations present after filter"] = table_filt.shape[0]
    else:
        table_filt = table

    logger["number of processors used"] = args.procs

    # correlate features
    if args.min_r is not None:
        if correl_method in [spearmanr, pearsonr, kendalltau]:
            # calculate correlations
            if args.verbose:
                print "Correlating with " + args.correl_method
            # correlate feature
            cor, p_vals = correl_method(general.biom_to_df(table_filt))
            cor = squareform(cor, checks=False)
        else:
            if args.verbose:
                print "Correlating using sparcc"
            cor, _ = basis_corr(general.biom_to_df(table_filt))
            cor = squareform(cor, checks=False)
        dist = ca.cor_to_dist(cor)
        min_dist = ca.cor_to_dist(args.min_r)
    elif args.min_p is not None:
        raise NotImplementedError()
    else:
        raise ValueError("min_p and min_r not given, one or other needs to be set")
    logger["distance metric used"] = args.correl_method

    if args.verbose:
        print "Features Correlated"
        print ""

    # make modules
    modules = mm.make_modules(dist, min_dist, obs_ids=table_filt.ids(axis="observation"))
    logger["number of modules created"] = len(modules)
    if args.verbose:
        print "Modules Formed"
        print "number of modules: %s" % len(modules)
        print "number of observations in modules: %s" % np.sum([len(i) for i in modules])
        print ""
    mm.write_modules_to_file(modules)

    # collapse modules
    coll_table = mm.collapse_modules(table, modules)
    mm.write_modules_to_dir(table, modules)
    logger["number of observations in output table"] = coll_table.shape[0]
    if args.verbose:
        print "Table Collapsed"
        print "collapsed Table Observations: " + str(coll_table.shape[0])
        print ""
    coll_table.to_json('make_modules.py', open('collapsed.biom', 'w'))

    # print correls and make correlation network
    correls = df_to_correls(pd.DataFrame(squareform(cor), index=table_filt.ids(axis="observation"),
                                         columns=table_filt.ids(axis="observation")))
    correls.to_csv('correls.txt', sep='\t', index=False)
    metadata = general.get_metadata_from_table(table_filt)
    net = general.correls_to_net(correls, conet=True, metadata=metadata, min_p=args.min_p, min_r=args.min_r)
    for i, otus in enumerate(modules):
        for otu in otus:
            net.node[otu]["module"] = i
    if args.verbose:
        print "Network Generated"
        print "number of nodes: " + str(net.number_of_nodes())
        print "number of edges: " + str(net.number_of_edges())
    logger["number of nodes"] = net.number_of_nodes()
    logger["number of edges"] = net.number_of_edges()
    nx.write_gml(net, 'conetwork.gml')

    logger.output_log()
