import pytest
from SCNIC.general import simulate_correls, get_metadata_from_table, filter_table, sparcc_paper_filter,\
                          bonferroni_adjust, bh_adjust
from biom.table import Table
import numpy as np
from numpy.testing import assert_allclose

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
    obs_ids = ["otu_%s" % i for i in xrange(5)]
    samp_ids = ["samp_%s" % i for i in xrange(5)]
    return Table(arr, obs_ids, samp_ids)


@pytest.fixture()
def unadj_ps():
    return [.01, .05, .5]


def test_get_metadata_from_table(biom_table1):
    metadata = get_metadata_from_table(biom_table1)
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
    adj_ps = np.array([.03, .15, 1.5])
    bon_ps = bonferroni_adjust(unadj_ps)
    assert isinstance(bon_ps, np.ndarray)
    assert_allclose(adj_ps, bon_ps)


def test_bh_adjust(unadj_ps):
    adj_ps = np.array([.03, .075, .5])
    bh_ps = bh_adjust(unadj_ps)
    assert isinstance(bh_ps, np.ndarray)
    assert_allclose(adj_ps, bh_ps)
