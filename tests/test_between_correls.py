import pytest
import os
from SCNIC.general import simulate_correls
from SCNIC.between_correls import between_correls


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

    return Arguments()


def test_between_correls(args, tmpdir):
    table1 = simulate_correls()
    table2 = simulate_correls()
    loc = tmpdir.mkdir("with_correls_test")
    table1.to_json("madebyme", open(str(loc)+"/table1.biom", 'w'))
    table2.to_json("madebyme", open(str(loc) + "/table2.biom", 'w'))
    os.chdir(str(loc))
    between_correls(args)
    files = os.listdir(str(loc)+'/out_dir')
    assert "correls.txt" in files
    assert "crossnet.gml" in files
