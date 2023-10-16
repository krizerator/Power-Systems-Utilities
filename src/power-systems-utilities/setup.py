"""Setup docstring"""
from setuptools import setup, find_packages

setup(
    name='power-systems-utilities',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'sympy'
    ]
)
