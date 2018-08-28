[![Codacy Badge](https://api.codacy.com/project/badge/Grade/9013f85974f84a06b544598aa934e032)](https://app.codacy.com/app/shafferm/SCNIC?utm_source=github.com&utm_medium=referral&utm_content=shafferm/SCNIC&utm_campaign=Badge_Grade_Dashboard)
[![PyPI](https://img.shields.io/pypi/v/SCNIC.svg)](https://pypi.python.org/pypi/SCNIC/0.5) [![Travis](https://img.shields.io/travis/shafferm/SCNIC.svg)](https://travis-ci.org/shafferm/SCNIC) [![Codacy grade](https://img.shields.io/codacy/grade/44d7474307bf4c62a271a9264c0c213a.svg)](https://www.codacy.com/app/shafferm/SCNIC/dashboard) [![Coveralls](https://img.shields.io/coveralls/shafferm/SCNIC.svg)](https://coveralls.io/github/shafferm/SCNIC)

# SCNIC
Sparse Cooccurnce Network Investigation for Compositional data
Pronounced 'scenic'.

*NOTE: SCNIC was recently updated to be python 3 only, old installations in python 2 only environments will not be
functional*

SCNIC is a package for the generation and analysis of cooccurence networks with compositional data. Data generated by
many next gen sequencing experiments is compositional (is a subsampling of the total community) which violates
assumptions of typical cooccurence network analysis techniques. 16S sequencing data is often very compositional in
nature so methods such as SparCC (https://bitbucket.org/yonatanf/sparcc) have been developed for studying correlations
microbiome data. SCNIC is designed with compositional data in mind and so provides multiple correlation measures
including SparCC.

Running SCNIC is possible via two different methods. SCNIC is packaged with scripts to allow running it on the command
line but also is avaliable as a Qiime2 plugin (https://www.github.com/shafferm/q2-SCNIC). Either method is valid but
usage of the Qiime2 plugin provides easier access when working within the Qiime2 ecosystem.

## Overview
### Within
The 'within' method takes as input BIOM formatted files (http://biom-format.org/) and forms cooccurence networks using a
 user specified correlation metric.

### Modules
From the correlation network generated as part of the within step, SCNIC  finds modules of cooccuring observations
by finding groups of observations which all have a minimum pairwise correlation value. Modules are summarized and a new
biom table with observations contained in modules collapsed into single observations are returned. This biom table along
with a list of modules and their contents are output.  A gml file of the network that can be opened using network
visualization tools such as cytoscape (http://www.cytoscape.org/) is created which contains all observation metadata
provided in the input biom file as well as module information. Please be aware that the networks output by this analysis
will only include positive correlations as only positive correlations are used in module finding and summarization.

### Between
The 'between' method takes two biom tables as input and calculates all pairwise correlations between the tables using a
selection of correlation metrics. A gml correlation network is output as well as a file containing statistics and
p-values of all correlations.

## Installation
SCNIC depends on a variety of software all of which can be install via conda and most of which can be installed by pip. The recommended installation method is to use conda but if you do not want to use conda and instead would like to install via pip then you must install [fastspar](https://github.com/scwatts/fastspar) and have it in your path.

### conda installation
It is recommended to install all of SCNIC's dependencies via conda in a new conda environment. To do this you first need to create a new environment:

```
conda create -n SCNIC python=3 pandas scipy numpy statsmodels h5py biom-format biom-format networkx >2 scikit-bio fastspar
```

Then enter the environment and install SCNIC via pip using these commands:
```
source activate SCNIC
pip install SCNIC
```

### Pip installation
To download the latest release from PyPI install using this command:
```
pip install SCNIC
```

### Install the latest version from github
To download the lastest changes to the repository use the following commands:
```
git clone https://github.com/shafferm/SCNIC.git
cd SCNIC/
python setup.py install
```
NOTE: This latest code may not be functional and should only be used if you want to play around with the code this is
based on.

## Example usage:

### 'within' mode:
```
SCNIC_analysis.py within -i example_table.biom -o within_output/ -m sparcc
```

### 'modules' mode:
```
SCNIC_analysis.py modules -i within_output/correls.txt -o modules_output/ --min_r .35 --table example_table.biom
```
NOTE: We use a minimum R value of .3 when running SparCC with 16S data as a computationally demanding bootstrapping
procedure must be run to determine p-values. We have run SparCC with 1 million bootstraps on a variety of datasets and
found that a R value of between .3 and .35 will always return FDR adjusted p-values less than .05 and .1 respectively.

### 'between' mode:
```
SCNIC_analysis.py between -1 example_table1.biom -2 example_table2.biom -o output_folder/ -m spearman --min_p .05
```
