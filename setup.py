from setuptools import setup, find_packages

__author__ = 'shafferm'

setup(
    name="SCNIC",
    version="0.2.1",
    setup_requires=['pytest-runner'],
    test_require=['pytest'],
    install_requires=["numpy", "scipy", "networkx", "biom-format", "pandas", "fast_sparCC"],
    scripts=["scripts/SCNIC_analysis.py"],
    packages=find_packages(),
    description="A tool for finding and summarizing modules of highly correlated observations in compositional data",
    author="Michael Shaffer",
    author_email='michael.shaffer@ucdenver.edu',
    url="https://github.com/shafferm/SCNIC/",
    download_url="https://github.com/shafferm/SCNIC/tarball/0.2.1"
)
