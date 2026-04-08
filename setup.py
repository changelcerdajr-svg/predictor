# setup.py
from setuptools import setup, find_packages

setup(
    name="baseball-betting-pipeline",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "lightgbm>=3.3.0",
        "scikit-learn>=0.24.0",
        "scipy>=1.7.0",
        "pybaseball>=0.1.0",
        "requests>=2.26.0",
        "joblib>=1.0.0",
    ],
)