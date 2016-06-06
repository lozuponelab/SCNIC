__author__ = 'shafferm'

from setuptools import setup, find_packages

setup(
    name="SCNIC",
    version="0.1",
    install_requires=["scipy", "numpy", "networkx", "biom-format", "pysurvey"],
    scripts=["scripts/SCNIC.py", "scripts/sparcc_correls.py"],
    packages=find_packages()
)
