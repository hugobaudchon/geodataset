# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'GeoDataset'
copyright = '2024, Hugo Baudchon'
author = 'Hugo Baudchon'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

sys.path.insert(0, os.path.abspath("../.."))

print('Source code path:', os.path.abspath("../.."))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx_autodoc_typehints',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

autodoc_inherit_docstrings = True

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'pydata_sphinx_theme'
html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/hugobaudchon/geodataset",
            "icon": "fab fa-github-square fa-xl",
        },
    ]
}
