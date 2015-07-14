__author__ = 'shafferm'

from setuptools import setup, find_packages

setup(
    name="correl_nets",
    version="0.1",
    install_requires=["scipy", "numpy", "networkx", "biom-format", "matplotlib", "pysurvey"],
    scripts=["scripts/correl_networks.py"],
    packages=find_packages()
)
