from setuptools import setup, find_packages

__author__ = 'lozuponelab'
__version__ = '0.6.4'

setup(
      name="SCNIC",
      version=__version__,
      setup_requires=['pytest-runner'],
      test_require=['pytest'],
      install_requires=["numpy", "scipy>=1.9.0,<=1.10.1", "networkx>=2", "biom-format", "pandas>=1", "scikit-bio", "statsmodels", "tqdm", "seaborn",
                        "h5py"],
      scripts=['scripts/SCNIC_analysis.py', 'scripts/module_enrichment.py'],
      packages=find_packages(),
      description="A tool for finding and summarizing modules of highly correlated observations in compositional data",
      author="Lozupone Lab",
      author_email='lozuponelab.dev@olucdenver.onmicrosoft.com',
      url="https://github.com/lozuponelab/SCNIC/",
      download_url="https://github.com/lozuponelab/SCNIC/tarball/%s" % __version__
)
