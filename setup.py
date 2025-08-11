from setuptools import setup, find_packages

with open('README.md', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='quantfeat',
    version='0.1.0',
    author='Srisha KS',
    author_email='srishaks6@gmail.com',
    description='Data analytics and feature engineering library for time series price data',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/srisha6505/quantfeat',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Topic :: Scientific/Engineering :: Information Analysis',
    ],
    python_requires='>=3.8',
    install_requires=[
        'pandas',
        'numpy',
        'seaborn',
        'matplotlib',
        'scipy',
    ],
)
