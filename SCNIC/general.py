from __future__ import division

import numpy as np
import networkx as nx
from biom.table import Table
from datetime import datetime
from collections import OrderedDict
from numpy.random import multivariate_normal
from statsmodels.sandbox.stats.multicomp import multipletests


__author__ = 'shafferm'


"""functions used widely"""


class Logger(OrderedDict):
    """"""
    # TODO: break up into sections for correls making, network making and module making
    def __init__(self, output):
        super(Logger, self).__init__()
        self.output_file = output
        self['start time'] = datetime.now()

    def output_log(self):
        with open(self.output_file, 'w') as f:
            self['finish time'] = datetime.now()
            self['elapsed time'] = self['finish time'] - self['start time']
            for key, value in self.items():
                f.write(key + ': ' + str(value) + '\n')


def p_adjust(pvalues, method='fdr_bh'):
    res = multipletests(pvalues, method=method)
    return np.array(res[1], dtype=float)


def sparcc_paper_filter(table):
    """if a observation averages more than 2 reads per sample then keep,
    if a sample has more than 500 reads then keep"""
    table = table.copy()
    table.filter(table.ids(axis='sample')[table.sum(axis='sample') > 500], axis='sample')
    table.filter(table.ids(axis='observation')[table.sum(axis='observation') / table.shape[1] >= 2], axis="observation")
    return table


def df_to_biom(df):
    return Table(np.transpose(df.as_matrix()), list(df.columns), list(df.index))


def get_metadata_from_table(table):
    metadata = dict()
    for _, otu_i, metadata_i in table.iter(axis="observation"):
        if metadata_i is not None:
            metadata[otu_i] = metadata_i
    return metadata


def underscore_to_camelcase(str_):
    str_ = str_.split('_')
    if len(str_) > 1:
        str_ = [str_[0]] + [i.capitalize() for i in str_[1:]]
    return ''.join(str_)


def correls_to_net(correls, min_p=None, min_r=None, conet=False, metadata=None):
    """correls is a pandas dataframe with a multiindex containing the correlated pair of features,
    r and optionally p and p_adj and optionally any others"""
    if metadata is None:
        metadata = []

    if conet:
        correls = correls[correls.r > 0]

    if min_p is not None:
        # filter to only include significant correlations
        if 'p_adj' in correls.columns:
            correls = correls[correls.p_adj < min_p]
        elif 'p' in correls.columns:
            correls = correls[correls.p < min_p]
        else:
            raise ValueError("No p or p_adj in correls")

    if min_r is not None:
        if conet:
            correls = correls[correls.r > min_r]
        else:
            correls = correls[np.abs(correls.r) > min_r]

    graph = nx.Graph()
    for (otu_i, otu_j), correl in correls.iterrows():
        if otu_i not in graph.node:
            graph.add_node(otu_i)
            if otu_i in metadata:
                for key in metadata[otu_i]:
                    graph_key = underscore_to_camelcase(str(key))
                    if metadata[otu_i][key] is None:
                        continue
                    if hasattr(metadata[otu_i][key], '__iter__'):
                        graph.node[otu_i][graph_key] = ';'.join(metadata[otu_i][key])
                    else:
                        graph.node[otu_i][graph_key] = metadata[correl.feature1][key]

        if otu_j not in graph.node:
            graph.add_node(otu_j)
            if otu_j in metadata:
                for key in metadata[otu_j]:
                    graph_key = underscore_to_camelcase(str(key))
                    if metadata[otu_j][key] is None:
                        continue
                    if hasattr(metadata[otu_j][key], '__iter__'):
                        graph.node[otu_j][graph_key] = ';'.join(metadata[otu_j][key])
                    else:
                        graph.node[otu_j][graph_key] = metadata[otu_j][key]
        graph.add_edge(otu_i, otu_j)
        for i in correl.index:
            graph_key = underscore_to_camelcase(str(i))
            graph.edges[otu_i, otu_j][graph_key] = correl[i]
    return graph


def filter_table(table, min_samples=None, to_file=False):
    """filter relative abundance table, by default throw away things greater than 1/3 zeros"""
    table = table.copy()
    # first sample filter
    if min_samples is not None:
        to_keep = [i for i in table.ids(axis='observation')
                   if sum(table.data(i, axis='observation') != 0) >= min_samples]
    else:
        to_keep = [i for i in table.ids(axis='observation')
                   if sum(table.data(i, axis='observation') != 0) >= table.shape[1]/3]
    table.filter(to_keep, axis='observation')

    if to_file:
        table.to_json('filter_table', open("filtered_tab.biom", 'w'))
        # open("filtered_rel_abund.txt", 'w').write(table.to_tsv())

    return table


def simulate_correls(corr_stren=(.99, .99), std=(1, 1, 1, 2, 2), means=(100, 100, 100, 100, 100), size=30,
                     noncors=10, noncors_mean=100, noncors_std=100, subsample=None, log=False):
    """
    Generates a correlation matrix with diagonal of stds based on input parameters and fills rest of matrix with
    uncorrelated values all with same  mean and standard deviations. Output should have a triangle of correlated
    observations and a pair all other observations should be uncorrelated. Correlation to covariance calculated by
    cor(X,Y)=cov(X,Y)/sd(X)sd(Y).

    Parameters
    ----------
    corr_stren: tuple of length 2, correlations in triangle and in pair
    std: tuple of length 5, standard deviations of each observation
    means: tuple of length 5, mean of each observation
    size: number of samples to generate from the multivariate normal distribution
    noncors: number of uncorrelated values
    noncors_mean: mean of uncorrelated values
    noncors_std: standard deviation of uncorrelated values
    subsample: Rarefy data to give threshold
    log: logonentiate mean values, if you are trying to detect log correlation

    Returns
    -------
    table: a biom table with (size) samples and (5+noncors) observations
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

    if log:
        # if log then log the array
        means = np.log(np.array(means))

    # fill the count table
    counts = multivariate_normal(means, cov, size).T

    if log:
        counts = np.log(counts)

    counts = np.round(counts)

    observ_ids = ["Observ_" + str(i) for i in range(cov.shape[0])]
    sample_ids = ["Sample_" + str(i) for i in range(size)]
    table = Table(counts, observ_ids, sample_ids)

    if subsample is not None:
        table = table.subsample(subsample)

    return table
