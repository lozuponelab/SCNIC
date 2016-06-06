import numpy as np
from scipy.spatial.distance import braycurtis


def cscore(u, v):
    """calculates standardized c-score according to https://en.wikipedia.org/wiki/Checkerboard_score"""
    u = u.astype(bool)
    v = v.astype(bool)
    r_u = np.sum(u)
    r_v = np.sum(v)
    s_uv = np.sum(np.logical_and(u, v))
    return (r_u-s_uv)*(r_v-s_uv)/(r_u+r_v-s_uv)


def paired_distances_from_table(table, dist_metric=braycurtis, rar=1000):
    """Takes a biom table and finds distances between all pairs of otus"""
    dists = list()
    rar_table = table.subsample(rar)

    for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in rar_table.iter_pairwise(axis='observation'):
        dist = dist_metric(data_i, data_j)
        dists.append([str(otu_i), str(otu_j), dist])

    header = ['feature1', 'feature2', 'dist']

    return dists, header


def boostrap_distance_vals(correls, header, nprocs=1, bootstraps=1000):
    """"""
    raise NotImplementedError
