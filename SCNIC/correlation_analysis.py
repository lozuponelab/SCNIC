import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
import warnings
from functools import partial
import pandas as pd


def cor_to_dist(cor):
    # convert from correlation to distance
    return 1 - ((cor + 1) / 2)

def between_correls_from_tables(table1, table2, correl_method=spearmanr, nprocs=1):
    """Take two biom tables and correlation"""
    correls = list()

    if nprocs == 1:
        for data_i, otu_i, _ in table1.iter(axis="observation"):
            for data_j, otu_j, _ in table2.iter(axis="observation"):
                corr = correl_method(data_i, data_j)
                correls.append([otu_i, otu_j, corr[0], corr[1]])
    else:
        import multiprocessing
        if nprocs > multiprocessing.cpu_count():
            warnings.warn("nprocs greater than CPU count, using all avaliable CPUs")
            nprocs = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(nprocs)
        print "Number of processors used: " + str(nprocs)

        correls = list()
        for data_i, otu_i, _ in table1.iter(axis="observation"):
            data_j = [data_j for data_j, otu_j, _ in table2.iter(axis="observation")]
            correls += pool.map(correl_method, [(data_i, j) for j in data_j])
            pool.close()
            pool.join()

    return pd.DataFrame(correls, columns=['feature1', 'feature2', 'r', 'p'])
