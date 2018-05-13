from scipy.stats import spearmanr
import warnings
from functools import partial
import pandas as pd
from biom.table import Table
import subprocess
from itertools import combinations


def df_to_correls(cor):
    """takes a square correlation matrix and turns it into a long form dataframe"""
    correls = pd.DataFrame(cor.stack().loc[list(combinations(cor.index, 2))], columns=['r'])
    return correls


def fastspar_correlation(table: Table) -> pd.DataFrame:
    # TODO: update this to use temporary file
    table.to_dataframe().to_dense().to_csv('otu_table.tsv', sep='\t', index_label='#OTU ID')
    subprocess.run(['fastspar', '-c',  'otu_table.tsv', '-r', 'correl_table.tsv', '-a', 'covar_table.tsv'])
    return pd.read_table('correl_table.tsv', index_col=0)


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
        for data_i, otu_i, _ in table1.iter(axis="observation"):
            datas_j = (data_j for data_j, _, _ in table2.iter(axis="observation"))
            corr = partial(correl_method, b=data_i)
            corrs = pool.map(corr, datas_j)
            correls += [(otu_i, table2.ids(axis="observation")[i], corrs[i][0], corrs[i][1])
                        for i in range(len(corrs))]
        pool.close()
        pool.join()

    correls = pd.DataFrame(correls, columns=['feature1', 'feature2', 'r', 'p'])
    return correls.set_index(['feature1', 'feature2'])  # this needs to be fixed, needs to return multiindex
