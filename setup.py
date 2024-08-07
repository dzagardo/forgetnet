# setup.py
from setuptools import setup, find_packages

setup(
    name='forgetnet',
    version='0.1.12',
    packages=find_packages(exclude=['tests']),
    install_requires=[
        'numpy',
        'torch',
        'trl',
    ],
    author='David Zagardo',
    author_email='dave@greenwillowstudios.com',
    description='A package for applying differential privacy to model training using block-level gradient shuffling',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dzagardo/forgetnet',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    entry_points={
        'forgetnet.privacy_mechanisms': [
            'dpshuffle = forgetnet.dp.dp_shuffle:DPShuffleGenerator',
        ],
    },
)