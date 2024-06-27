# setup.py

from setuptools import setup, find_packages

setup(
    name='my_dp_model_package',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'torch',
    ],
    author='David Zagardo',
    author_email='dave@greenwillowstudios.com',
    description='A package for applying differential privacy to model weights',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/dzagardo/forgetnet',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
