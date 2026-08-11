"""
Between-table correlation workflow.

Compute correlations between two BIOM tables, build a network, and output
results for downstream SCNIC analysis.
"""

import os
from os import path
from biom import load_table
from scipy.stats import spearmanr, pearsonr
import networkx as nx
import numpy as np
import shutil

from SCNIC import general
from SCNIC import correlation_analysis as ca

__author__ = 'shafferm'

_spearmanr = spearmanr


def spearmanr(x, y):
    """
    Pass-through wrapper around scipy.stats.spearmanr.

    Parameters
    ----------
    x, y : array-like

    Returns
    -------
    tuple
    """
    return _spearmanr(x, y)

## added verbose argument to be consistent with within and modules 
def between_correls(table1, table2, output_loc, max_p=None, min_r=None, correl_method='spearman', sparcc_filter=False,
                    min_sample=None, p_adjust='fdr_bh', procs=1, force=False, verbose=False):
    """
    Compute correlations between two BIOM tables and write network output.

    Parameters
    ----------
    table1 : str
    table2 : str
    output_loc : str
    max_p : float or None
    min_r : float or None
    correl_method : str, default 'spearman'
    sparcc_filter : bool, default False
    min_sample : int or None
    p_adjust : str, default 'fdr_bh'
    procs : int, default 1
    force : bool, default False
    """
    pretty_correl_method = correl_method ## save name of actual correlation method used for naming output files 
    logger = general.Logger(path.join(output_loc, f"SCNIC_between_{pretty_correl_method}_log.txt"))
    logger["SCNIC analysis type"] = "between"
    logger["number of processors used"] = procs

    # correlation and p-value adjustment methods
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr}
    correl_method = correl_methods[correl_method.lower()]

    # load tables
    table1 = load_table(table1)
    table2 = load_table(table2)
    if verbose:
        print(f"Table 1 loaded: {str(table1.shape[0])} observations\n")
        print(f"Table 2 loaded: {str(table2.shape[0])} observations\n")
    
    logger["input table 1"] = table1
    logger["number of samples in input table 1"] = table1.shape[1]
    logger["number of observations in input table 1"] = table1.shape[0]

    logger["input table 2"] = table2
    logger["number of samples in input table 2"] = table2.shape[1]
    logger["number of observations in input table 2"] = table2.shape[0]

    table1 = table1.sort()
    table2 = table2.sort()

    # make new output directory and change to it
    if force and output_loc is not None:
        shutil.rmtree(output_loc, ignore_errors=True)
    if output_loc is not None:
        if not path.isdir(output_loc):
            os.makedirs(output_loc)
    logger["output directory"] = path.abspath(output_loc)
    
    if max_p is not None:
        raise ValueError("SCNIC does not currently support module making based on p-values.") # then why is this an option?

    # filter tables
    if sparcc_filter is True:
        table1_filt = general.sparcc_paper_filter(table1)
        table2_filt = general.sparcc_paper_filter(table2)
        if verbose:
            print("Table 1 SPARCC filtered: %s observations\n" % str(table1_filt.shape[0]))
            print("Table 2 SPARCC filtered: %s observations\n" % str(table2_filt.shape[0]))
        logger["sparcc paper filter"] = True
        logger["number of observations present in table 1 after sparcc filter"] = table1_filt.shape[0]
        logger["number of observations present in table 2 after sparcc filter"] = table2_filt.shape[0]
    elif min_sample is not None:
        table1_filt = general.filter_table(table1, min_sample)
        table2_filt = general.filter_table(table2, min_sample)
        logger["min samples present"] = min_sample
        logger["number of observations present in table 1 after min sample filter"] = table1_filt.shape[0]
        logger["number of observations present in table 2 after min sample filter"] = table2_filt.shape[0]

        if verbose:
            print(f"Table 1 minimum sample (n={min_sample}) filtered: {str(table1_filt.shape[0])} observations\n")
            print(f"Table 2 minimum sample (n={min_sample}) filtered: {str(table2_filt.shape[0])} observations\n")
    else:
        table1_filt = table1
        table2_filt = table2


    if not np.array_equal(table1_filt.ids(), table2_filt.ids()):
        raise ValueError("Tables have different sets of samples present!")

    metadata = general.get_metadata_from_table(table1_filt)
    metadata.update(general.get_metadata_from_table(table2_filt))

    if correl_method in [spearmanr, pearsonr]:
        # make correlations
        logger["correlation metric"] = pretty_correl_method
        logger["p adjustment method"] = p_adjust

        if verbose:
            print(f"Correlating with {pretty_correl_method}...")

        correls = ca.between_correls_from_tables(table1_filt, table2_filt, correl_method, nprocs=procs)
        correls.sort_values(correls.columns[-1], inplace=True)
        correls['pAdjusted'] = general.p_adjust(correls['p'], method=p_adjust)
        correls.to_csv(open(path.join(output_loc, f"between_{pretty_correl_method}_correls.txt"), 'w'), sep='\t', index_label=('feature1', 'feature2'))

        if verbose:
            print("Features correlated!")
            print(f"{pretty_correl_method} correlations: between_{pretty_correl_method}_correls.txt written to {path.abspath(output_loc)}\n")
    else:
        raise ValueError(f"{pretty_correl_method} correlation calculations are NOT supported by SCNIC between analysis!\n please choose one of the following: {correl_methods.keys()}")

    # filter correlations prior to making network 
    logger["minimum r-value threshold"] = min_r
    correls_filt = general.filter_correls(correls, min_r=min_r) ## isn't an r-value threshold filter highly recommended? - or is that for module creation?

    ## remove rows where feature1 is identical to feature2 (prevents self-loops in graph)
    #correls_filt_selfLoops = correls_filt[correls_filt.index.get_level_values('feature1') != correls_filt.index.get_level_values('feature2')] # may want to keep this in the future though since these features are asvs in real life and you're comparing two separate biom tables

    # make network
    net = general.correls_to_net(correls_filt, metadata=metadata)
    logger["number of nodes"] = net.number_of_nodes()
    logger["number of edges"] = net.number_of_edges()
    nx.write_gml(net, path.join(output_loc, f"between_{pretty_correl_method}_crossnet.gml"))
    if verbose:
        print("Network made!")
        print(f"Number of nodes: {net.number_of_nodes()}")
        print(f"Number of edges: {net.number_of_edges()}")
        print(f"{pretty_correl_method} correlation network: between_{pretty_correl_method}_crossnet.gml written to {path.abspath(output_loc)}\n")

    logger.output_log()
