import versioneer
from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    requirements = f.read().splitlines()

setup(
    name='geodataset',
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    packages=find_packages(),
    package_data={
        '': ['utils/categories/*/*.json', 'utils/aois/*/*.geojson', 'utils/aois/*/*.gpkg']
    },
    install_requires=requirements,
    description='This package provide essential tools for cutting raster and their labels into smaller tiles,'
                ' useful for machine learning tasks. Also provides datasets compatible with pytorch.',
    author='Hugo Baudchon',
    author_email='hugo@baudchon.com',
    license='MIT'
)
