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
            self.min_sample = None
            self.sparcc_filter = False
            self.procs = 1
            self.min_p = None
            self.min_r = .35

    return Arguments()



# integration test
def test_within_correls_classic_correlation_min_r(args1, tmpdir):
    table = simulate_correls()
    loc = tmpdir.mkdir("with_correls_test")
    table.to_json("madebyme", open(str(loc)+"/table1.biom", 'w'))
    os.chdir(str(loc))
    within_correls(args1)
    files = os.listdir(str(loc)+'/out_dir')
    assert "collapsed.biom" in files
    assert "modules.txt" in files
    assert "modules" in files
    assert "conetwork.gml" in files
    assert "correls.txt" in files
