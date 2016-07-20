__author__ = 'shafferm'

from setuptools import setup, find_packages

setup(
    name="SCNIC",
    version="0.1",
    install_requires=["numpy", "scipy", "networkx", "biom-format", "pandas", "fast_sparCC"],
    scripts=["scripts/SCNIC_analysis.py"],
    packages=find_packages()
)
