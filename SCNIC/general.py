"""
Common helper utilities for SCNIC.

This module provides shared functions for correlation filtering, table
manipulation, metadata extraction, and network conversion.
"""

from __future__ import division

import re

import numpy as np
import networkx as nx
from biom.table import Table
from datetime import datetime
from collections import OrderedDict
from numpy.random import multivariate_normal
from statsmodels.stats.multitest import multipletests ## madi update: location of statsmodels multipletests function (adjust p-value)


__author__ = 'shafferm'


class Logger(OrderedDict):
    """
    Simple logging container that writes timestamped key/value pairs to a file.

    Parameters
    ----------
    output : str
        Path to the log file to write.
    """
    def __init__(self, output):
        super(Logger, self).__init__()
        self.output_file = output
        self['start time'] = datetime.now()

    def output_log(self):
        """
        Write the accumulated log entries to the configured output file.

        The method adds finish time and elapsed time before writing each
        entry as "key: value".
        """
        with open(self.output_file, 'w') as f:
            self['finish time'] = datetime.now()
            self['elapsed time'] = self['finish time'] - self['start time']
            for key, value in self.items():
                f.write(key + ': ' + str(value) + '\n')


def p_adjust(pvalues, method='fdr_bh'):
    """
    Adjust p-values for multiple testing.

    Parameters
    ----------
    pvalues : array-like
        Raw p-values to adjust.
    method : str, default 'fdr_bh'
        Multiple-testing correction method used by statsmodels.

    Returns
    -------
    numpy.ndarray
        Adjusted p-values.
    """
    res = multipletests(pvalues, method=method)
    return np.array(res[1], dtype=float)


def sparcc_paper_filter(table):
    """
    Apply the SparCC paper filter to a BIOM table.

    Keeps samples with more than 500 reads and observations that average
    at least 2 reads per sample.

    Parameters
    ----------
    table : biom.table.Table
        Input BIOM observation table.

    Returns
    -------
    biom.table.Table
        Filtered table.
    """
    table = table.copy()
    table.filter(table.ids(axis='sample')[table.sum(axis='sample') > 500], axis='sample')
    table.filter(table.ids(axis='observation')[table.sum(axis='observation') / table.shape[1] >= 2], axis="observation")
    return table


def df_to_biom(df):
    """
    Convert a pandas DataFrame to a BIOM Table.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame with observations as rows and samples as columns.

    Returns
    -------
    biom.table.Table
        BIOM table with observations as observation IDs and samples as sample IDs.
    """
    return Table(np.transpose(df.values), [str(i) for i in df.columns], [str(i) for i in df.index])


def get_metadata_from_table(table, axis='observation'):
    """
    Extract a metadata dictionary from a BIOM table.

    Parameters
    ----------
    table : biom.table.Table
        Input BIOM table.
    axis : str, default 'observation'
        Axis from which to extract metadata.

    Returns
    -------
    dict
        Mapping of feature ID to metadata object.
    """
    metadata = dict()
    for _, otu_i, metadata_i in table.iter(axis=axis):
        if metadata_i is not None:
            metadata[str(otu_i)] = metadata_i
    return metadata


def underscore_to_camelcase(str_):
    """
    Convert an underscore- or hyphen-delimited string to camel case.

    Parameters
    ----------
    str_ : str
        Input string to convert.

    Returns
    -------
    str
        Camel-case string.
    """
    str_ = re.split('[-_]', str_)
    if len(str_) > 1:
        str_ = [str_[0]] + [i.capitalize() for i in str_[1:]]
    return ''.join(str_)


def filter_correls(correls, max_p=None, min_r=None, conet=False):
    """
    Filter correlation results by p-value and absolute correlation.

    Parameters
    ----------
    correls : pandas.DataFrame
        Correlation table indexed by feature pairs, with columns including
        ``r`` and optionally ``p`` or ``p_adj``.
    max_p : float, optional
        Maximum allowed p-value or adjusted p-value.
    min_r : float, optional
        Minimum absolute correlation coefficient.
    conet : bool, default False
        If True, keep only positive correlations.

    Returns
    -------
    pandas.DataFrame
        Filtered correlation table.
    """
    # TODO: allow non r column names
    # TODO: allow non p_adj column names
    if conet:
        correls = correls[correls.r > 0]

    if max_p is not None:
        # filter to only include significant correlations
        if 'p_adj' in correls.columns:
            correls = correls[correls.p_adj < max_p]
        elif 'p' in correls.columns:
            correls = correls[correls.p < max_p]
        else:
            raise ValueError("No p or p_adj in correls")

    if min_r is not None:
        correls = correls[np.abs(correls.r) > min_r]

    return correls

## attempting to add a helper function that fixes np.float64 edge values in the correlation_network.gml
## these values prevent the gml file from being read back in by nx.read_gml() which causes problems for the 
## unit tests
def _to_gml_compatible(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): _to_gml_compatible(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return [_to_gml_compatible(item) for item in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_to_gml_compatible(item) for item in value]
    return value


def correls_to_net(correls, metadata=None):
    """
    Convert correlation edge list to a NetworkX graph.

    Parameters
    ----------
    correls : pandas.DataFrame
        Correlations indexed by observation pairs.
    metadata : dict, optional
        Node metadata mapping feature IDs to attribute dictionaries.

    Returns
    -------
    networkx.Graph
        Graph with nodes for each feature and edge attributes for each correlation.
    """
    if metadata is None:
        metadata = {}
    graph = nx.Graph()
    for otu_pair, correl in correls.iterrows():
        for otu in otu_pair:
            if otu not in graph.nodes:
                graph.add_node(otu)
                if otu in metadata:
                    for key in metadata[otu]:
                        graph_key = underscore_to_camelcase(str(key))
                        value = _to_gml_compatible(metadata[otu][key])
                        if value is None:
                            continue
                        elif type(value) is str: ## ruff suggested edit instead of '==' - can also use isinstance()
                            graph.nodes[otu][graph_key] = value
                        elif hasattr(value, '__iter__'):
                            graph.nodes[otu][graph_key] = ';'.join(value)
                        else:
                            graph.nodes[otu][graph_key] = value
        graph.add_edge(*otu_pair)
        for i in correl.index:
            graph_key = underscore_to_camelcase(str(i))
            graph.edges[otu_pair][graph_key] = _to_gml_compatible(correl[i])
    return graph


def filter_table(table, min_samples):
    """
    Remove observations with fewer than the specified number of non-zero samples.

    Parameters
    ----------
    table : biom.table.Table
        Input BIOM table.
    min_samples : int
        Minimum number of samples in which an observation must be present.

    Returns
    -------
    biom.table.Table
        Filtered BIOM table.
    """
    table = table.copy()
    # first sample filter
    to_keep = [i for i in table.ids(axis='observation')
               if sum(table.data(i, axis='observation') != 0) >= min_samples]
    table.filter(to_keep, axis='observation')
    return table


def simulate_correls(corr_stren=(.99, .99), std=(1, 1, 1, 2, 2), means=(100, 100, 100, 100, 100), size=30,
                     noncors=10, noncors_mean=100, noncors_std=100):
    """
    Generate a synthetic BIOM table containing correlated and uncorrelated features. 
    Generates a correlation matrix with diagonal of stds based on input parameters and fills rest of matrix with
    uncorrelated values all with same  mean and standard deviations. Output should have a triangle of correlated
    observations and a pair all other observations should be uncorrelated. Correlation to covariance calculated by
    cor(X,Y)=cov(X,Y)/sd(X)sd(Y). 

    Parameters
    ----------
    corr_stren : tuple of float, default (.99, .99)
        Correlation strengths for the correlated triangle and pair of features.
    std : tuple of float, default (1, 1, 1, 2, 2)
        Standard deviations for the first five features.
    means : tuple of float, default (100, 100, 100, 100, 100)
        Means for the first five features.
    size : int, default 30
        Number of samples to generate.
    noncors : int, default 10
        Number of additional uncorrelated features.
    noncors_mean : float, default 100
        Mean for uncorrelated features.
    noncors_std : float, default 100
        Standard deviation for uncorrelated features.

    Returns
    -------
    biom.table.Table
        Simulated BIOM table with correlated and uncorrelated observations.
    """
    cor = [[std[0], corr_stren[0], corr_stren[0], 0., 0.],  # define the correlation matrix for the triangle and pair
           [corr_stren[0], std[1], corr_stren[0], 0., 0.],
           [corr_stren[0], corr_stren[0], std[2], 0., 0.],
           [0., 0., 0., std[3], corr_stren[1]],
           [0., 0., 0., corr_stren[1], std[4]]]
    cor = np.array(cor)
    cov = np.zeros(np.array(cor.shape) + noncors)  # generate empty covariance matrix to be filled
    for i in range(cor.shape[0]):  # fill in all but diagonal of covariance matrix, first 5
        for j in range(i + 1, cor.shape[0]):
            curr_cov = cor[i, j] * cor[i, i] * cor[j, j]
            cov[i, j] = curr_cov
            cov[j, i] = curr_cov
    for i in range(cor.shape[0]):  # fill diagonal of covariance matrix, first 5
        cov[i, i] = np.square(cor[i, i])
    means = list(means)
    for i in range(cor.shape[0], cov.shape[0]):  # fill diagonal of covariance, 6 to end and populate mean list
        cov[i, i] = noncors_std
        means.append(noncors_mean)

    # fill the count table
    counts = multivariate_normal(means, cov, size).T

    counts = np.round(counts)

    observ_ids = ["Observ_" + str(i) for i in range(cov.shape[0])]
    sample_ids = ["Sample_" + str(i) for i in range(size)]
    table = Table(counts, observ_ids, sample_ids)

    return table
