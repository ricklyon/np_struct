from setuptools import setup, find_packages
import os


setup(
    name='structures',
    description='dev package',
    author='Rick Lyon',
    version='0.1.1',
    url="https://github.com/rlyon14/structures",
    packages=['structures',],
    install_requires=(
        'numpy',
        'pyserial >= 3.0'
    ),
)