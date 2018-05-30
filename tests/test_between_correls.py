import pytest
import os
from SCNIC.general import simulate_correls
from SCNIC.between_correls import between_correls
from biom.util import biom_open


@pytest.fixture()
def args():
    class Arguments(object):
        def __init__(self):
            self.table1 = "table1.biom"
            self.table2 = "table2.biom"
            self.output = "out_dir"
            self.correl_method = "spearman"
            self.p_adjust = "bh"
            self.min_sample = None
            self.min_p = None
            self.min_r = None
            self.sparcc_filter = True
            self.force = False
            self.procs = 1

    return Arguments()


def test_between_correls(args, tmpdir):
    table1 = simulate_correls()
    table2 = simulate_correls()
    loc = tmpdir.mkdir("with_correls_test")
    with biom_open(str(loc.join("table1.biom")), 'w') as f:
        table1.to_hdf5(f, 'madebyme')
    with biom_open(str(loc.join("table2.biom")), 'w') as f:
        table2.to_hdf5(f, 'madebyme')
    os.chdir(str(loc))
    between_correls(args)
    files = os.listdir(str(loc)+'/out_dir')
    assert "correls.txt" in files
    assert "crossnet.gml" in files
