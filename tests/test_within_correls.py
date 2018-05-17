import pytest
import os
from SCNIC.general import simulate_correls
from SCNIC.within_correls import within_correls


@pytest.fixture()
def args1():
    class Arguments(object):
        def __init__(self):
            self.input = "table1.biom"
            self.output = "out_dir"
            self.correl_method = "spearman"
            self.p_adjust = "bh"
            self.outlier_removal = False
            self.verbose = False
            self.force = False
            self.min_sample = 2
            self.sparcc_filter = False
            self.procs = 1
            self.sparcc_p = None

    return Arguments()


@pytest.fixture()
def args2():
    class Arguments(object):
        def __init__(self):
            self.input = "table1.biom"
            self.output = "out_dir"
            self.correl_method = "spearman"
            self.p_adjust = None
            self.outlier_removal = False
            self.verbose = True
            self.force = False
            self.min_sample = None
            self.sparcc_filter = True
            self.procs = 1
            self.sparcc_p = None

    return Arguments()


# integration test
def test_within_correls_classic_correlation_min_r_min_sample(args1, tmpdir):
    table = simulate_correls()
    loc = tmpdir.mkdir("with_correls_test")
    table.to_json("madebyme", open(str(loc)+"/table1.biom", 'w'))
    os.chdir(str(loc))
    within_correls(args1)
    files = os.listdir(str(loc)+'/out_dir')
    assert "correls.txt" in files


# integration test
def test_within_correls_classic_correlation_min_r_sparcc_filter(args2, tmpdir):
    table = simulate_correls()
    loc = tmpdir.mkdir("with_correls_test")
    table.to_json("madebyme", open(str(loc)+"/table1.biom", 'w'))
    os.chdir(str(loc))
    within_correls(args2)
    files = os.listdir(str(loc)+'/out_dir')
    assert "correls.txt" in files
