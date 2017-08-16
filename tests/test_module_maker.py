import pytest
from SCNIC.module_maker import make_modules, collapse_modules, write_modules_to_dir
import os
import glob
from biom.table import Table
import numpy as np
from scipy.spatial.distance import squareform


@pytest.fixture()
def biom_table1():
    arr = np.array([[250,   0, 100, 446,   75],
                    [  0,   0,   1,   1,    2],
                    [  2,   2,   2,   2,    2],
                    [100, 100, 500,   1, 1000],
                    [500,   5,   0,  50,  100]])
    obs_ids = ["otu_%s" % i for i in xrange(5)]
    samp_ids = ["samp_%s" % i for i in xrange(5)]
    return Table(arr, obs_ids, samp_ids)


@pytest.fixture()
def dist1():
    dist = np.array([[0.0, .05, 0.2, 1.0, 1.0],
                     [.05, 0.0, 0.2, 1.0, 1.0],
                     [0.2, 0.2, 0.0, 1.0, 1.0],
                     [1.0, 1.0, 1.0, 0.0, 0.7],
                     [1.0, 1.0, 1.0, 0.7, 0.0]])
    return squareform(dist)


@pytest.fixture()
def obs_ids1():
    obs_ids = ["otu_%s" % i for i in xrange(5)]
    return obs_ids


@pytest.fixture()
def modules1(dist1, obs_ids1):
    return make_modules(dist1, min_dist=.21, obs_ids=obs_ids1)


def test_make_modules(modules1):
    assert len(modules1) == 1
    assert np.sum([len(i) for i in modules1]) == 3


def test_collapse_modules(biom_table1, modules1):
    coll_table = collapse_modules(biom_table1, modules1)
    assert coll_table.shape[0] == biom_table1.shape[0]-2
    assert coll_table.shape[1] == biom_table1.shape[1]
    assert coll_table.sum() == biom_table1.sum()
    assert np.array_equal(coll_table.sum(axis="sample"), biom_table1.sum(axis="sample"))
    # set since order doesn't matter
    assert frozenset(coll_table.sum(axis="observation")) == frozenset(np.array([885, 1701, 655]))


def test_write_modules_to_dir(biom_table1, modules1, tmpdir):
    tempdir = tmpdir.mkdir("mods")
    os.chdir(str(tempdir))
    write_modules_to_dir(biom_table1, modules1)
    os.chdir("modules")
    fnames = glob.glob(str(tempdir)+"/modules/*.biom")
    assert len(fnames) == len(modules1)
