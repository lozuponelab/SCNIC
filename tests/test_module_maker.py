import pytest
from SCNIC.module_analysis import correls_to_cor, make_modules, collapse_modules, write_modules_to_dir, cor_to_dist, \
                                  write_modules_to_file, add_modules_to_metadata
import os
import glob
from biom.table import Table
import numpy as np
from scipy.spatial.distance import squareform
import pandas as pd


@pytest.fixture()
def correls():
    index = (('otu_1', 'otu_2'),
             ('otu_1', 'otu_3'),
             ('otu_2', 'otu_3'))
    data = [.7, .01, .35]
    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index), columns=['r'])


@pytest.fixture()
def cor():
    data = [(1.0, .70, .01),
            (.70, 1.0, .35),
            (.01, .35, 1.0)]
    return squareform(np.array(data), checks=False)


@pytest.fixture()
def dist():
    data = [(0, .15, .495),
            (.15, 0, .325),
            (.495, .325, 0)]
    return squareform(np.array(data), checks=False)


@pytest.fixture()
def biom_table1():
    arr = np.array([[250,   0, 100, 446,   75],
                    [  0,   0,   1,   1,    2],
                    [  2,   2,   2,   2,    2],
                    [100, 100, 500,   1, 1000],
                    [500,   5,   0,  50,  100]])
    obs_ids = ["otu_%s" % i for i in range(5)]
    samp_ids = ["samp_%s" % i for i in range(5)]
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
    obs_ids = ["otu_%s" % i for i in range(5)]
    return obs_ids


@pytest.fixture()
def modules1(dist1, obs_ids1):
    return make_modules(dist1, min_dist=.21, obs_ids=obs_ids1)


@pytest.fixture()
def metadata():
    meta = {
        'otu_0': {'taxonomy': 'k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; f__Staphylococcaceae; g__Staphylococcus; s__'},
        'otu_1': {'taxonomy': 'k__Bacteria; p__Firmicutes; c__Bacilli; o__Bacillales; f__Paenibacillaceae; g__Paenibacillus; s__'},
        'otu_2': {'taxonomy': 'k__Bacteria; p__Proteobacteria; c__Betaproteobacteria; o__Methylophilales; f__Methylophilaceae; g__; s__'},
        'otu_3': {'taxonomy': 'k__Bacteria; p__Firmicutes; c__Clostridia; o__Clostridiales; f__Lachnospiraceae; g__[Ruminococcus]; s__'},
        'otu_4': {'taxonomy': 'k__Bacteria; p__Actinobacteria; c__Actinobacteria; o__Actinomycetales; f__Microbacteriaceae; g__; s__'}
    }
    return meta


def test_correls_to_cor(correls, cor):
    test_cor, test_labels = correls_to_cor(correls)
    assert set(test_labels) == set([j for i in correls.index for j in i])
    assert np.array_equal(test_cor, cor)


def test_cor_to_dist(cor, dist):
    test_dist = cor_to_dist(cor)
    assert np.allclose(test_dist, dist)


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


def test_write_modules_to_file(modules1, tmpdir):
    path = tmpdir.join('modules.txt')
    write_modules_to_file(modules1, path_str=str(path))
    data = open(str(path)).readlines()
    assert len(data) == 1
    assert len(data[0].strip().split()) == 4


def test_add_module_to_metadata(modules1, metadata):
    test_metadata = add_modules_to_metadata(modules1, metadata)
    assert len(metadata) == len(test_metadata)
    assert len(test_metadata['otu_0']) == 2
    assert test_metadata['otu_0']['module'] == 0
    assert len(test_metadata['otu_1']) == 2
    assert test_metadata['otu_1']['module'] == 0
    assert len(test_metadata['otu_2']) == 2
    assert test_metadata['otu_2']['module'] == 0
    assert len(test_metadata['otu_3']) == 1
    assert len(test_metadata['otu_4']) == 1
