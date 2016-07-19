import general
import numpy as np
from scipy.stats import spearmanr, pearsonr, kendalltau
import warnings
from functools import partial
import pandas as pd


def biom_pairwise_iter_no_metadata(biom_pairwise_iter):
    for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in biom_pairwise_iter:
        yield (data_i, otu_i), (data_j, otu_j)


def calculate_correlation(paired_iter, correl_method):
    (data_i, otu_i), (data_j, otu_j) = paired_iter
    correl_methods = {'spearman': spearmanr, 'pearson': pearsonr, 'kendall': kendalltau}
    correl_method = correl_methods[correl_method]
    correl = correl_method(data_i, data_j)
    return [str(otu_i), str(otu_j), correl[0], correl[1]]


def paired_correlations_from_table(table, correl_method="spearman", p_adjust=general.bh_adjust, nprocs=1):
    """Takes a biom table and finds correlations between all pairs of otus."""

    if nprocs == 1:
        correls = list()
        correl_methods = {'spearman': spearmanr, 'pearson': pearsonr, 'kendall': kendalltau}
        for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in table.iter_pairwise(axis='observation'):
            correl = correl_methods[correl_method](data_i, data_j)
            correls.append([str(otu_i), str(otu_j), correl[0], correl[1]])

    else:
        import multiprocessing

        if nprocs > multiprocessing.cpu_count():
            warnings.warn("nprocs greater than CPU count, using all avaliable CPUs")
            nprocs = multiprocessing.cpu_count()

        pool = multiprocessing.Pool(nprocs)
        print "Number of processors used: " + str(nprocs)

        pfun = partial(calculate_correlation, correl_method=correl_method)
        table_iter = table.iter_pairwise(axis='observation')
        correls = pool.map(pfun, biom_pairwise_iter_no_metadata(table_iter))
        pool.close()
        pool.join()

    header = ['feature1', 'feature2', 'r', 'p']
    correls_df = pd.DataFrame(correls, columns=header)

    # adjust p-value if desired
    if p_adjust is not None:
        p_adjusted = p_adjust([i[3] for i in correls])
        correls_df['adjusted_p'] = p_adjusted

    return correls


def paired_correlations_from_table_with_outlier_removal(table, good_samples, min_keep=10, correl_method=spearmanr,
                                                        p_adjust=general.bh_adjust):
    """Takes a biom table and finds correlations between all pairs of otus."""
    correls = list()

    for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in table.iter_pairwise(axis='observation'):
        samp_union = np.union1d(good_samples[otu_i], good_samples[otu_j])
        # remove zero zero points
        # samp_union = [ind for i, ind in enumerate(samp_union) if data_i[i]!=0 and data_j[i]!=0]
        if len(samp_union) > min_keep:
            correl = correl_method(data_i[samp_union], data_j[samp_union])
            correls.append([str(otu_i), str(otu_j), correl[0], correl[1]])

    header = ['feature1', 'feature2', 'r', 'p']
    correls_df = pd.DataFrame(correls, columns=header)

    # adjust p-value if desired
    if p_adjust is not None:
        p_adjusted = p_adjust([i[3] for i in correls])
        correls_df['adjusted_p'] = p_adjusted

    return correls
