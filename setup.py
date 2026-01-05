from setuptools import setup, find_packages

setup(
    name='admet_gnn',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'torch',
        'torch_geometric',
        'scikit-learn',
        'matplotlib',
        'numpy',
        'rdkit-pypi',
    ],
    entry_points={
        'console_scripts': [
            'admet_predict=admet_gnn.inference:main',
        ],
    },
)
