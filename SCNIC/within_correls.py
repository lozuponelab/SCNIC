from __future__ import division

import os
from os import path

import networkx as nx

from biom import load_table
from scipy.stats import spearmanr, pearsonr, kendalltau

from SCNIC import general
from SCNIC import correlation_analysis as ca


def within_correls(input_loc, output_loc, correl_method='sparcc', sparcc_filter=False, min_sample=None, procs=1,
                   sparcc_p=1000, p_adjust='fdr_bh', verbose=False):
    logger = general.Logger(path.join(output_loc, "SCNIC_within_log.txt"))
    logger["SCNIC analysis type"] = "within"

    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr, 'kendall': kendalltau, 'sparcc': 'sparcc'}
    correl_method = correl_methods[correl_method.lower()]

    # get features to be correlated
    table = load_table(input_loc)
    logger["input table"] = input_loc
    if verbose:
        print("Table loaded: " + str(table.shape[0]) + " observations")
        print("")
    logger["number of samples in input table"] = table.shape[1]
    logger["number of observations in input table"] = table.shape[0]

    # make new output directory
    if output_loc is not None:
        if not path.isdir(output_loc):
            os.makedirs(output_loc)
    logger["output directory"] = path.abspath(output_loc)

    # filter
    if sparcc_filter is True:
        table_filt = general.sparcc_paper_filter(table)
        if verbose:
            print("Table filtered: %s observations" % str(table_filt.shape[0]))
            print("")
        logger["sparcc paper filter"] = True
        logger["number of observations present after filter"] = table_filt.shape[0]
    elif min_sample is not None:
        table_filt = general.filter_table(table, min_sample)
        if verbose:
            print("Table filtered: %s observations" % str(table_filt.shape[0]))
            print("")
        logger["min samples present"] = min_sample
        logger["number of observations present after filter"] = table_filt.shape[0]
    else:
        table_filt = table

    logger["number of processors used"] = procs

    # correlate features
    if correl_method in [spearmanr, pearsonr, kendalltau]:
        # calculate correlations
        if verbose:
            print("Correlating with %s" % correl_method)
        # correlate feature
        correls = ca.calculate_correlations(table_filt, correl_method, nprocs=procs, p_adjust_method=p_adjust)
    elif correl_method == 'sparcc':
        if sparcc_p is None:
            correls = ca.fastspar_correlation(table_filt, verbose=verbose, nprocs=procs)
        else:
            correls = ca.fastspar_correlation(table_filt, calc_pvalues=True, bootstraps=sparcc_p,
                                              verbose=verbose, nprocs=procs, p_adjust_method=p_adjust)
    else:
        raise ValueError("How did this even happen?")
    logger["distance metric used"] = correl_method
    if verbose:
        print("Features Correlated")
        print("")

    correls.to_csv(path.join(output_loc, 'correls.txt'), sep='\t', index_label=('feature1', 'feature2'))
    if verbose:
        print("Correls.txt written")

    # make correlation network
    metadata = general.get_metadata_from_table(table_filt)
    net = general.correls_to_net(correls, metadata=metadata)
    nx.write_gml(net, path.join(output_loc, 'correlation_network.gml'))
    if verbose:
        print("Network made")
        print("")

    logger.output_log()
