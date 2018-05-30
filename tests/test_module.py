import pytest
import os
from SCNIC.module import module_maker
import pandas as pd
from biom.table import Table
from biom.util import biom_open
import numpy as np


@pytest.fixture()
def args1():
    class Arguments(object):
        def __init__(self):
            self.input = 'correls.txt'
            self.output = 'out_dir'
            self.min_r = .5
            self.min_p = None
            self.table = 'table1.biom'
            self.verbose = True
    return Arguments()


@pytest.fixture()
def table():
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
def correls():
    index = (('otu_1', 'otu_2'),
             ('otu_1', 'otu_3'),
             ('otu_2', 'otu_3'))
    data = [.7, .01, .35]
    return pd.DataFrame(data, index=pd.MultiIndex.from_tuples(index), columns=['r'])


# integration test
def test_within_correls_classic_correlation_min_r_min_sample(tmpdir, args1, correls, table):
    loc = tmpdir.mkdir("with_correls_test")
    with biom_open(str(loc.join("table1.biom")), 'w') as f:
        table.to_hdf5(f, "madebyme")
    correls.to_csv(str(loc.join('correls.txt')), sep='\t')
    os.chdir(str(loc))
    module_maker(args1)
    files = os.listdir(str(loc)+'/out_dir')
    assert "modules.txt" in files
