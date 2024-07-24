# setup.py

from setuptools import setup, find_packages

setup(
    name='RSPAnalyzer',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'pandas',
        'matplotlib',
        'scipy',
        'emcee',
        'corner',
        'tqdm'
    ],
    author='Noa Levy',
    author_email='noa.levy13@mail.huji.ac.il',
    description='This library is designed for analyzing the physical parameters of \
    light curves and spectra from various radio supernovae. It uses Chevaliers 1998 \
    Equation 1 model and the MCMC algorithm.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/my_library',
    license='MIT',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
)
