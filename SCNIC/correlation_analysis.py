from scipy.stats import spearmanr, pearsonr, kendalltau
import warnings
from functools import partial
import pandas as pd


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

    return pd.DataFrame(correls, columns=['feature1', 'feature2', 'r', 'p'])
