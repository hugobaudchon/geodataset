from setuptools import setup, find_packages

setup(
    name='geodataset',
    version='0.0.1',
    packages=find_packages(),
    install_requires=[
        'einops==0.7.0',
        'geopandas==0.13.2',
        'laspy==2.5.3',
        'lazrs==0.5.3',
        'matplotlib==3.8.2',
        'numpy==1.25.0',
        'pandas==2.2.0',
        'Pillow==10.2.0',
        'rasterio==1.3.7',
        'shapely==2.0.1',
        'tqdm==4.65.0',
        'xmltodict==0.13.0',
    ],
    description='This package provide essential tools for cutting raster and their labels into smaller tiles,'
                ' useful for machine learning tasks. Also provides datasets compatible with pytorch.',
    author='Hugo Baudchon',
    author_email='hugo@baudchon.com',
)
