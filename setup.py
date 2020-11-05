from setuptools import setup, find_packages

requirements = [
    's3fs',
    'autogluon',
    'mxnet',
    'pyyaml',
]

setup(
    name='autogluon-benchmark',
    version='0.0.1',
    packages=find_packages(exclude=('docs', 'tests', 'scripts')),
    url='https://github.com/Innixma/autogluon-benchmark',
    license='Apache',
    author='AutoGluon Community',
    install_requires=requirements,
    description='utilities for AutoGluon'
)
