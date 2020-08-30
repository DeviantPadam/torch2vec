#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 20:25:50 2020

@author: deviantpadam
"""


from setuptools import setup, find_packages

setup_args = dict(
    name='torch2vec',
    version='0.1.2',
    description='A PyTorch implementation of Doc2Vec (distributed memory) with similarity measure.',
    license='GPL3',
    packages=find_packages(),
    author='DeviantPadam(Padam Gupta)',
    author_email='padamgupta1999@gmail.com',
    url='https://github.com/DeviantPadam/torch2vec'
)

install_requires = [
    'numpy',
    'tqdm',
    'pandas',
    'torch',
    'scikit-learn',
    'nltk',
    'pytorch-lightning'
]

if __name__ == '__main__':
    setup(**setup_args, install_requires=install_requires)