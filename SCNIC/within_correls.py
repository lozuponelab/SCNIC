# TODO: Testing correls as list vs as pandas dataframe for speed

from __future__ import division

import os

import networkx as nx
import pandas as pd

from biom import load_table
from sparcc_fast.sparcc_functions import basis_corr
from scipy.stats import spearmanr, pearsonr, kendalltau
from scipy.spatial.distance import squareform
from itertools import combinations

import general


def df_to_correls(cor):
    """takes a square correlation matrix and turns it into a long form dataframe"""
    correls = pd.DataFrame(cor.stack().loc[list(combinations(cor.index, 2))], columns=['r'])
    return correls


def within_correls(args):
    logger = general.Logger("SCNIC_within_log.txt")
    logger["SCNIC analysis type"] = "within"


    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr, 'kendall': kendalltau, 'sparcc': None}
    p_methods = {'bh': general.bh_adjust, 'bon': general.bonferroni_adjust}
    correl_method = correl_methods[args.correl_method.lower()]
    if args.p_adjust is not None:
        p_adjust = p_methods[args.p_adjust]
    else:
        p_adjust = None

    # get features to be correlated
    table = load_table(args.input)
    logger["input table"] = args.input
    if args.verbose:
        print "Table loaded: " + str(table.shape[0]) + " observations"
        print ""
    logger["number of samples in input table"] = table.shape[1]
    logger["number of observations in input table"] = table.shape[0]

    # make new output directory and change to it
    if args.output is not None:
        if not os.path.isdir(args.output):
            os.makedirs(args.output)
        os.chdir(args.output)
    logger["output directory"] = os.getcwd()

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
    if correl_method in [spearmanr, pearsonr, kendalltau]:
        # calculate correlations
        if args.verbose:
            print "Correlating with " + args.correl_method
        # correlate feature
        cor, p_vals = correl_method(general.biom_to_df(table_filt))
        cor = pd.DataFrame(cor, index=table_filt.ids(axis="observation"), columns=table_filt.ids(axis="observation"))
        # cor = squareform(cor, checks=False)
        if args.sparcc_p is not None:
            raise NotImplementedError()
    else:
        if args.verbose:
            print "Correlating using sparcc"
        cor, _ = basis_corr(general.biom_to_df(table_filt))
        # cor = squareform(cor, checks=False)
    logger["distance metric used"] = args.correl_method
    if args.verbose:
        print "Features Correlated"
        print ""

    # print correls
    correls = df_to_correls(cor)
    if 'p' in correls.columns:
        correls['p-adj'] = p_adjust(correls['p'])
    correls.to_csv('correls.txt', sep='\t')
    if args.verbose:
        print "Correls.txt written"

    # make correlation network
    metadata = general.get_metadata_from_table(table_filt)
    net = general.correls_to_net(correls, metadata=metadata)
    nx.write_gml(net, 'correlation_network.gml')
    if args.verbose:
        print "Network made"
        print ""

    logger.output_log()
