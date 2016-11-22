import pytest
import os
from SCNIC.general import simulate_correls
from SCNIC.within_correls import within_correls

@pytest.fixture()
def args():
    class Arguments(object):
        def __init__(self):
            self.input = "table1.biom"
            self.output = "out_dir"
            self.correl_method = "pearson"
            self.p_adjust = "bh"
            self.outlier_removal = False
            self.k_size = 3

        def __getattr__(self, name):
            return None
    return Arguments()


def test_within_correls(args, tmpdir):
    table = simulate_correls()
    loc = tmpdir.mkdir("with_correls_test")
    table.to_json("madebyme", open(str(loc)+"/table1.biom", 'w'))
    os.chdir(str(loc))
    within_correls(args)
    files = os.listdir(str(loc)+'/out_dir')
    assert "collapsed.biom" in files
    assert "modules.txt" in files
    assert "modules" in files
    assert "conetwork.gml" in files
    assert "correls.txt" in files
