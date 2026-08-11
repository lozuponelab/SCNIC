"""
Within-table correlation workflow.

Compute correlations for a single BIOM table and build a correlation network.
"""

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
    """
    Run a within-table SCNIC correlation analysis and write outputs.

    Parameters
    ----------
    input_loc : str
    output_loc : str
    correl_method : str, default 'sparcc'
    sparcc_filter : bool, default False
    min_sample : int or None
    procs : int, default 1
    sparcc_p : int, default 1000
    p_adjust : str, default 'fdr_bh'
    verbose : bool, default False
    """
    pretty_correl_method = correl_method ## save name of actual correlation method used for naming output files 
    logger = general.Logger(path.join(output_loc, f"SCNIC_within_{pretty_correl_method}_log.txt"))
    logger["SCNIC analysis type"] = "within"
    logger["number of processors used"] = procs

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
        logger["number of observations present after sparcc filter"] = table_filt.shape[0]
    elif min_sample is not None:
        table_filt = general.filter_table(table, min_sample)
        if verbose:
            print("Table filtered: %s observations" % str(table_filt.shape[0]))
            print("")
        logger["min samples present"] = min_sample
        logger["number of observations present after min sample filter"] = table_filt.shape[0]
    else:
        table_filt = table

    # correlate features
    if correl_method in [spearmanr, pearsonr, kendalltau]:
        # calculate correlations
        if verbose:
            print("Correlating with %s" % pretty_correl_method)
        # correlate feature
        correls = ca.calculate_correlations(table_filt, correl_method, nprocs=procs, p_adjust_method=p_adjust)
    elif correl_method == 'sparcc':
        if sparcc_p is None:
            correls = ca.fastspar_correlation(table_filt, verbose=verbose, nprocs=procs)
        else:
            correls = ca.fastspar_correlation(table_filt, calc_pvalues=True, bootstraps=sparcc_p,
                                              verbose=verbose, nprocs=procs, p_adjust_method=p_adjust)
    else:
        raise ValueError(f"{pretty_correl_method} correlation calculations are NOT supported by SCNIC within analysis!\n please choose one of the following: {correl_methods.keys()}")
    
    logger["correlation metric"] = pretty_correl_method

    correls.to_csv(path.join(output_loc, f'within_{pretty_correl_method}_correls.txt'), sep='\t', index_label=('feature1', 'feature2'))
    if verbose:
        print("Features Correlated")
        print(f"{pretty_correl_method} correlations: within_{pretty_correl_method}_correls.txt written to {path.abspath(output_loc)}")
        print("")
        

    # make correlation network
    metadata = general.get_metadata_from_table(table_filt)
    net = general.correls_to_net(correls, metadata=metadata)
    nx.write_gml(net, path.join(output_loc, f'within_{pretty_correl_method}_correlation_network.gml'))
    if verbose:
        print("Network made")
        print(f"Number of nodes: {net.number_of_nodes()}")
        print(f"Number of edges: {net.number_of_edges()}")
        print(f"{pretty_correl_method} correlation network: within_{pretty_correl_method}_correlation_network.gml written to {path.abspath(output_loc)}")
        print("")

    logger.output_log()
