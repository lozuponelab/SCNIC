import os
from SCNIC.general import simulate_correls
from SCNIC.between_correls import between_correls
from biom.util import biom_open
import pytest


def test_between_correls(tmpdir):
    table1 = simulate_correls()
    table2 = simulate_correls()
    loc = tmpdir.mkdir("with_correls_test")
    with biom_open(str(loc.join("table1.biom")), 'w') as f:
        table1.to_hdf5(f, 'madebyme')
    with biom_open(str(loc.join("table2.biom")), 'w') as f:
        table2.to_hdf5(f, 'madebyme')
    os.chdir(str(loc))

    # P value based module making is not yet implemented, check this
    with pytest.raises(Exception) as exc_info:   
        between_correls('table1.biom', 'table2.biom', 'out_dir', correl_method='pearson', max_p=.1)
    assert exc_info.value.args[0] == "SCNIC does not currently support module making based on p-values."
    
    # Run with min r threshold and check that files exist
    between_correls('table1.biom', 'table2.biom', 'out_dir_2', correl_method='pearson', min_r=.2)
    files = os.listdir(str(loc)+'/out_dir_2')
    assert "correls.txt" in files
    assert "crossnet.gml" in files
