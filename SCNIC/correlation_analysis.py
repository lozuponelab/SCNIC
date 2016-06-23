import general
import numpy as np
from scipy.stats import spearmanr
import pandas as pd


def paired_correlations_from_table(table, correl_method=spearmanr, p_adjust=general.bh_adjust):
    """Takes a biom table and finds correlations between all pairs of otus."""
    correls = list()

    for (data_i, otu_i, metadata_i), (data_j, otu_j, metadata_j) in table.iter_pairwise(axis='observation'):
        correl = correl_method(data_i, data_j)
        correls.append([str(otu_i), str(otu_j), correl[0], correl[1]])

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
