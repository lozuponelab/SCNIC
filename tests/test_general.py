import pytest
from SCNIC.general import simulate_correls, get_metadata_from_table, filter_table, sparcc_paper_filter, \
                          p_adjust, Logger, df_to_biom, underscore_to_camelcase, filter_correls, correls_to_net
from biom.table import Table
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd

# TODO: simulate sparse table to test filtering
# TODO: include HMP table to test filtering?


@pytest.fixture()
def biom_table1():
    table1 = simulate_correls()
    assert isinstance(table1, Table)
    return table1


@pytest.fixture()
def biom_table2():
    arr = np.array([[250,   0, 100, 446,   75],
                    [  0,   0,   1,   1,    2],
                    [  2,   2,   2,   2,    2],
                    [100, 100, 500,   1, 1000],
                    [500,   5,   0,  50,  100]])
    obs_ids = ["otu_%s" % i for i in range(5)]
    samp_ids = ["samp_%s" % i for i in range(5)]
    obs_meta = [
        {'taxonomy': 'k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; f__Staphylococcaceae; g__Staphylococcus; s__'},
        {'taxonomy': 'k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; f__Paenibacillaceae; g__Paenibacillus; s__'},
        {'taxonomy': 'k__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Methylophilales; f__Methylophilaceae; g__; s__'},
        {'taxonomy': 'k__Bacteria; p__Firmicutes; c__Clostridia; o__Clostridiales; f__Lachnospiraceae; g__[Ruminococcus]; s__'},
        {'taxonomy': 'k__Bacteria; p__Actinobacteria; c__Actinobacteria; o__Actinomycetales; f__Microbacteriaceae; g__; s__'}
    ]
    return Table(arr, obs_ids, samp_ids, observation_metadata=obs_meta)


@pytest.fixture()
def unadj_ps():
    return [.01, .05, .5]


@pytest.fixture()
def df():
    arr = np.array([[250,   0],
                    [  0,   1],
                    [  2,   2]])
    obs_ids = ["otu_%s" % i for i in range(2)]
    samp_ids = ["samp_%s" % i for i in range(3)]
    return pd.DataFrame(arr, index=samp_ids, columns=obs_ids)


@pytest.fixture()
def correls():
    index = (('otu_1', 'otu_2'),
             ('otu_1', 'otu_3'),
             ('otu_2', 'otu_3'))
    columns = ['r', 'p', 'p_adj']
    data = [[.7, .0001, .01],
            [.01, .5, .9],
            [-.35, .005, .04]]
    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index), columns=columns)


@pytest.fixture()
def metadata():
    meta = {
        'otu_1': {'taxonomy': 'k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; f__Staphylococcaceae; g__Staphylococcus; s__'},
        'otu_2': {'taxonomy': 'k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; f__Paenibacillaceae; g__Paenibacillus; s__'},
        'otu_3': {'taxonomy': 'k__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Methylophilales; f__Methylophilaceae; g__; s__'},
    }
    return meta


def test_Logger(tmpdir):
    loc = tmpdir.mkdir("test")
    log_path = str(loc) + "/log.txt"
    logger = Logger(log_path)
    logger["Testing"] = "1, 2, 3"
    logger.output_log()
    log = open(log_path).readlines()
    assert len(log) == 4
    assert log[0].startswith('start time')
    assert log[1].startswith('Testing: 1, 2, 3')
    assert log[-2].startswith('finish time')
    assert log[-1].startswith('elapsed time')


def test_get_metadata_from_table(biom_table2):
    metadata = get_metadata_from_table(biom_table2)
    assert isinstance(metadata, dict)


def test_filter_table(biom_table1):
    table_filt = filter_table(biom_table1, min_samples=10)
    assert isinstance(table_filt, Table)


def test_filter_better(biom_table2):
    table_filt = filter_table(biom_table2, min_samples=4)
    assert len(table_filt.ids(axis="observation")) == 4
    assert len(table_filt.ids(axis="sample")) == 5


def test_sparcc_paper_filter(biom_table1):
    table_filt = sparcc_paper_filter(biom_table1)
    assert isinstance(table_filt, Table)


def test_sparcc_paper_filter_better(biom_table2):
    table_filt = sparcc_paper_filter(biom_table2)
    assert len(table_filt.ids(axis="observation")) == 4
    assert len(table_filt.ids(axis="sample")) == 3


def test_bonferroni_adjust(unadj_ps):
    adj_ps = np.array([.03, .15, 1])
    bon_ps = p_adjust(unadj_ps, method='b')
    assert isinstance(bon_ps, np.ndarray)
    assert_allclose(adj_ps, bon_ps)


def test_bh_adjust(unadj_ps):
    adj_ps = np.array([.03, .075, .5])
    bh_ps = p_adjust(unadj_ps, 'fdr_bh')
    assert isinstance(bh_ps, np.ndarray)
    assert_allclose(adj_ps, bh_ps)


def test_df_to_biom(df):
    test_biom = df_to_biom(df)
    assert test_biom.sum() == 255
    assert test_biom.shape == (2, 3)
    assert np.allclose(test_biom.sum(axis='observation'), (252, 3))
    assert np.allclose(test_biom.sum(axis='sample'), (250, 1, 4))


def test_underscore_to_camelcase():
    assert underscore_to_camelcase('p-adj') == 'pAdj'
    assert underscore_to_camelcase('p_adj') == 'pAdj'


def test_filter_correls(correls):
    correls_filt = filter_correls(correls, min_r=.3)
    assert len(correls_filt) == 2
    correls_filt = filter_correls(correls, min_r=.3, conet=True)
    assert len(correls_filt) == 1
    correls_filt = filter_correls(correls, min_p=.05)
    assert len(correls_filt) == 2


def test_correls_to_net(correls, metadata):
    test_net = correls_to_net(correls, metadata)
    assert len(test_net.node) == 3
    assert len(test_net.edges) == 3
