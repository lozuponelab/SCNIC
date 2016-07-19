__author__ = 'shafferm'

from setuptools import setup, find_packages

setup(
    name="SCNIC",
    version="0.1",
    setup_requires=['numpy'],
    # need to get rid of pysurvey it needs pandas and matplotlib and I don't
    install_requires=["scipy", "networkx", "biom-format", "pysurvey", "pandas", "matplotlib"],
    scripts=["scripts/SCNIC_analysis.py", "scripts/sparcc_correls.py"],
    packages=find_packages()
)
