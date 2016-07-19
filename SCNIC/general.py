from __future__ import division

from scipy.stats import rankdata, linregress
import numpy as np
import networkx as nx
from biom.table import Table
import pandas as pd
from datetime import datetime
from collections import OrderedDict

__author__ = 'shafferm'


"""functions used widely"""


class Logger(OrderedDict):
    """"""
    #TODO: break up into sections for correls making, network making and module making
    def __init__(self, output):
        super(Logger, self).__init__()
        self.output_file = output
        self['start time'] = datetime.now()

    def output_log(self):
        with open(self.output_file, 'w') as f:
            self['finish time'] = datetime.now()
            self['elapsed time'] = self['finish time'] - self['start time']
            for key, value in self.iteritems():
                f.write(key + ': ' + str(value) + '\n')


def sparcc_paper_filter(table):
    """if a observation averages more than 2 reads per sample then keep,
    if a sample has more than 500 reads then keep"""
    table = table.copy()
    table.filter(table.ids(axis='sample')[table.sum(axis='sample') > 500], axis='sample')
    table.filter(table.ids(axis='observation')[table.sum(axis='observation') / table.shape[1] >= 2], axis="observation")
    return table


def df_to_biom(df):
    return Table(np.transpose(df.as_matrix()), list(df.columns), list(df.index))


def biom_to_df(biom):
    return pd.DataFrame(np.transpose(biom.matrix_data.todense()), index=biom.ids(), columns=biom.ids(axis="observation"))


def get_metadata_from_table(table):
    metadata = dict()
    for data_i, otu_i, metadata_i in table.iter(axis="observation"):
        if metadata_i is not None:
            metadata[otu_i] = metadata_i
    return metadata


def bh_adjust_old(p_vals):
    """benjamini-hochberg p-value adjustment"""
    p_vals = np.array(p_vals)
    return p_vals*len(p_vals)/rankdata(p_vals, method='max')

def bh_adjust(pvalues):
    """benjamini-hochberg p-value adjustment stolen from
    http://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python
    """
    pvalues = np.array(pvalues)
    n = pvalues.shape[0]
    new_pvalues = np.empty(n)
    values = [(pvalue, i) for i, pvalue in enumerate(pvalues)]
    values.sort()
    values.reverse()
    new_values = []
    for i, vals in enumerate(values):
        rank = n - i
        pvalue, index = vals
        new_values.append((n/rank) * pvalue)
    for i in xrange(0, int(n)-1):
        if new_values[i] < new_values[i+1]:
            new_values[i+1] = new_values[i]
    for i, vals in enumerate(values):
        pvalue, index = vals
        new_pvalues[index] = new_values[i]
    return new_pvalues


def bonferroni_adjust_old(p_vals):
    """bonferroni p-value adjustment"""
    return [i*len(p_vals) for i in p_vals]


def bonferroni_adjust(pvalues):
    pvalues = np.array(pvalues)
    n = float(pvalues.shape[0])
    new_pvalues = n * pvalues
    return new_pvalues


def print_delimited(out_fp, lines, header=None):
    """print a tab delimited file with optional header"""
    out = open(out_fp, 'w')
    if header is not None:
        out.write('\t'.join([str(i) for i in header])+'\n')
    for line in lines:
        out.write('\t'.join([str(i) for i in line])+'\n')
    out.close()


def correls_to_net(correls, min_p=None, min_r=None, conet=False, metadata=None):
    """"""
    if metadata is None:
        metadata = []

    if min_p is None and min_r is None:
        min_p = .05

    if conet:
        correls = correls[correls[correls.columns[2]] > 0]

    if min_p is not None:
        # filter to only include significant correlations
        correls = correls[correls[correls.columns[-1]] < min_p]

    if min_r is not None:
        if conet:
            correls = correls[correls[correls.columns[2]] > min_r]
        else:
            correls = correls[np.abs(dists_df[dists_df.columns[2]]) > min_r]

    graph = nx.Graph()
    for correl in correls.itertuples(index=False):
        graph.add_node(correl[0])
        if correl[0] in metadata:
            for key in metadata[correl[0]]:
                if hasattr(metadata[correl[0]][key], '__iter__'):
                    graph.node[correl[0]][key] = ';'.join(metadata[correl[0]][key])
                else:
                    graph.node[correl[0]][key] = metadata[correl[0]][key]

        graph.add_node(correl[1])
        if correl[1] in metadata:
            for key in metadata[correl[1]]:
                if hasattr(metadata[correl[0]][key], '__iter__'):
                    graph.node[correl[1]][key] = ';'.join(metadata[correl[1]][key])
                else:
                    graph.node[correl[1]][key] = metadata[correl[1]][key]
        if len(correl) == 3:
            graph.add_edge(correl[0], correl[1], r=correl[2], signpos=int(abs(correl[2]) == correl[2]))
        elif len(correl) == 4:
            graph.add_edge(correl[0], correl[1], r=correl[2],
                           p=correl[3], signpos=int(abs(correl[2]) == correl[2]))
        elif len(correl) == 5:
            graph.add_edge(correl[0], correl[1], r=correl[2],
                           p=correl[3], padj=correl[4], signpos=int(abs(correl[2]) == correl[2]))
        else:
            raise ValueError("correls should only have 3-5 members")
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


def remove_outliers(table, min_obs=10):
    """returns indicies of good samples in numpy array"""
    good_samples = dict()
    for data_i, otu_i, metadata_i in table.iter(axis="observation"):
        q75, q25 = np.percentile(data_i, (75, 25))
        iqr = q75 - q25
        med = np.median(data_i)
        good_indicies = np.array([i for i, data in enumerate(data_i) if 3 * iqr + med > data > med - 3 * iqr])
        if np.sum(good_indicies) > min_obs:
            good_samples[otu_i] = good_indicies
    return good_samples


def compare_slopes(table1, table2, otu1, otu2):
    x1 = table1.data(otu1, axis="observation")
    y1 = table1.data(otu2, axis="observation")
    x2 = table2.data(otu1, axis="observation")
    y2 = table2.data(otu2, axis="observation")
    lin1 = linregress(x1, y1)
    slope1 = lin1[0]
    lin2 = linregress(x2, y2)
    slope2 = lin2[0]
    print "slope1: " + str(slope1) + " slope2: " + str(slope2)


def plot_networkx(graph):
    """plot networkx object in matplotlib"""

    try:
        import matplotlib.pyplot as plt
    except:
        print "matplotlib not installed, please install to use plotting functions"
        return None

    graph_pos = nx.circular_layout(graph)

    nx.draw_networkx_nodes(graph, graph_pos, node_size=1000, node_color='blue', alpha=0.3)
    nx.draw_networkx_edges(graph, graph_pos, width=2, alpha=0.3, edge_color='green')
    nx.draw_networkx_labels(graph, graph_pos, font_size=12, font_family='sans-serif')

    plt.show()
