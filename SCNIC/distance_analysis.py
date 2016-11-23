from __future__ import division

import multiprocessing
import warnings
from collections import Counter
import numpy as np
from biom import Table
from functools import partial
from scipy.spatial.distance import jaccard, braycurtis, euclidean, canberra
import pandas as pd


def cscore(u, v):
    """calculates standardized c-score according to https://en.wikipedia.org/wiki/Checkerboard_score"""
    u = u.astype(bool)
    v = v.astype(bool)
    r_u = np.sum(u)
    r_v = np.sum(v)
    s_uv = np.sum(np.logical_and(u, v))
    return (r_u-s_uv)*(r_v-s_uv)/(r_u+r_v-s_uv)


def paired_distances_from_table(table, dist_metric='braycurtis'):
    """Takes a biom table and finds distances between all pairs of otus"""
    dist_methods = {'jaccard': jaccard, 'cscore': cscore, 'braycurtis': braycurtis, 'euclidean': euclidean,
                    'canberra': canberra}
    dist_metric = dist_methods[dist_metric]
    dists = list()

    for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in table.iter_pairwise(axis='observation'):
        dist = dist_metric(data_i, data_j)
        dists.append([str(otu_i), str(otu_j), dist])

    header = ['feature1', 'feature2', 'dist']
    return pd.DataFrame(dists, columns=header)


def refill_biom(row):
    tot = np.sum(row)
    rands = np.random.randint(0, len(row), size=tot)
    counts = Counter(rands)
    new_data = np.zeros(len(row))
    for val in counts:
        new_data[val] = counts[val]
    return new_data


def shuffle_table(table):
    matrix = table.matrix_data.todense()
    np.random.shuffle(matrix)
    return Table(matrix, table.ids(axis="observation"), table.ids())


def shuffle_table_matrix(sparse_matrix):
    matrix = sparse_matrix.todense()
    np.random.shuffle(matrix)
    return Table(matrix, range(matrix.shape[0]), range(matrix.shape[1]))


def refill_table(table):
    matrix = table.matrix_data.todense()
    new_matrix = np.zeros(matrix.shape)
    sums = np.sum(matrix, axis=1)
    for i in xrange(sums.shape[0]):
        rands = np.random.randint(0, matrix.shape[1], size=sums[i])
        counts = Counter(rands)
        for val in counts:
            new_matrix[i, val] = counts[val]
    return Table(new_matrix, table.ids(axis="observation"), table.ids())


def refill_table_matrix(sparse_matrix):
    matrix = sparse_matrix.todense()
    new_matrix = np.zeros(matrix.shape)
    sums = np.sum(matrix, axis=1)
    for i in xrange(sums.shape[0]):
        rands = np.random.randint(0, matrix.shape[1], size=sums[i])
        counts = Counter(rands)
        for val in counts:
            new_matrix[i, val] = counts[val]
    return Table(new_matrix, range(new_matrix.shape[0]), range(new_matrix.shape[1]))


def bootstrapped_distance(bootstrap, measured_dists, sparse_matrix, dist_metric):
    bootstrapped_table = shuffle_table_matrix(sparse_matrix)
    dists = paired_distances_from_table(bootstrapped_table, dist_metric=dist_metric)
    return dists['dist'] < measured_dists


def bootstrap_distance_vals(table, dist_metric='braycurtis', nprocs=1, bootstraps=1000, p_adjust=None):
    """"""
    dists = paired_distances_from_table(table, dist_metric=dist_metric)
    measured_dists = dists['dist']

    if nprocs == 1:
        print "Using 1 processor to calculate distances"
        multi_results = np.zeros((bootstraps, len(dists)), dtype=bool)
        for i in xrange(bootstraps):
            multi_results[i] = bootstrapped_distance(None, measured_dists, table.matrix_data, dist_metric)

    else:
        if nprocs > multiprocessing.cpu_count():
            warnings.warn("nprocs greater than CPU count, using all avaliable CPUs")
            nprocs = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(nprocs)
        print "Number of processors used: " + str(nprocs)

        pfun = partial(bootstrapped_distance, measured_dists=measured_dists, sparse_matrix=table.matrix_data,
                       dist_metric=dist_metric)
        multi_results = pool.map(pfun, range(bootstraps))
        pool.close()
        pool.join()

    dists['p'] = np.sum(multi_results, axis=0)/bootstraps

    return dists
