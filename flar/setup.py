# setup.py
from setuptools import setup, find_packages

setup(
    name="flar",                 # The name of your package
    version="0.1.0",                 # Package version
    description="FLAR Causal Discovery methods",
    author="Anonymous",
    packages=["flar"],        # Automatically find sub-packages (the 'dagboost/' folder)
    python_requires=">=3.9",         # or whichever Python version(s) you support
    install_requires=[
        "numpy",
        "pandas",
        "torch>=1.10",
        "networkx",
        "scikit-learn",
        "statsmodels",
        # add any other dependencies you need
    ],
)
