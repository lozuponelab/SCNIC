from setuptools import setup, find_packages

__author__ = 'shafferm'
__version__ = '0.3.2'

setup(
      name="SCNIC",
      version=__version__,
      setup_requires=['pytest-runner'],
      test_require=['pytest'],
      install_requires=["numpy", "scipy", "networkx", "biom-format", "pandas", "fast_sparCC <= 0.1.6",
                        "scikit-bio == 0.4.2"],
      scripts=["scripts/SCNIC_analysis.py"],
      packages=find_packages(),
      description="A tool for finding and summarizing modules of highly correlated observations in compositional data",
      author="Michael Shaffer",
      author_email='michael.shaffer@ucdenver.edu',
      url="https://github.com/shafferm/SCNIC/",
      download_url="https://github.com/shafferm/SCNIC/tarball/%s" % __version__
)
