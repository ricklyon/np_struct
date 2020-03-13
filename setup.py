from setuptools import setup, find_packages
import os


setup(
    name='pktcomm',
    description='dev package',
    author='Rick Lyon',
    author_email='rlyon@yahoo.com',
    version='0.1.1',
    url="https://github.com/rlyon14/comm",
    packages=['packets34',],
    install_requires=(
        'numpy',
        'pyserial >= 3.0'
    ),
)