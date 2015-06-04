__author__ = 'shafferm'

from setuptools import setup, find_packages

setup(
    name="correl_nets",
    packages=find_packages(),
    version="0.1",
    install_requires=["scipy", "numpy", "networkx", "biom-format", "matplotlib"],
    scripts=["scripts/correl_nets.py"]
)
