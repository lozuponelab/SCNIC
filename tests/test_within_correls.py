import os
from SCNIC.general import simulate_correls
from SCNIC.within_correls import within_correls
from biom.util import biom_open


# integration test
def test_within_correls_classic_correlation_min_r_min_sample(tmpdir):
    table = simulate_correls()
    loc = tmpdir.mkdir("with_correls_test")
    with biom_open(str(loc.join("table1.biom")), 'w') as f:
        table.to_hdf5(f, 'madebyme')
    os.chdir(str(loc))
    within_correls('table1.biom', 'out_dir', correl_method='pearson')
    files = os.listdir(str(loc)+'/out_dir')
    assert "correls.txt" in files


# integration test
def test_within_correls_classic_correlation_min_r_sparcc_filter(tmpdir):
    table = simulate_correls()
    loc = tmpdir.mkdir("with_correls_test")
    with biom_open(str(loc.join("table1.biom")), 'w') as f:
        table.to_hdf5(f, 'madebyme')
    os.chdir(str(loc))
    within_correls('table1.biom', 'out_dir', correl_method='pearson')
    files = os.listdir(str(loc)+'/out_dir')
    assert "correls.txt" in files
