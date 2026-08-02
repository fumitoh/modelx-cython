# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Make the modelx_cython package importable for autodoc.
sys.path.insert(
    0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

import modelx_cython

# -- Project information -----------------------------------------------------

project = "modelx-cython"
copyright = "2023-2026, Fumito Hamamura"
author = "Fumito Hamamura"
version = modelx_cython.__version__
release = modelx_cython.__version__

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

exclude_patterns = []

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
]
myst_heading_anchors = 3

# -- Autodoc / Napoleon configuration ----------------------------------------

# Docstrings follow the NumPy style.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_preprocess_types = True
# Render Attributes sections as :ivar: fields so they do not collide with
# the attribute entries that autodoc generates for the same names.
napoleon_use_ivar = True

autodoc_member_order = "bysource"
autodoc_typehints = "description"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

try:
    import pydata_sphinx_theme  # noqa: F401

    html_theme = "pydata_sphinx_theme"
    html_theme_options = {
        "icon_links": [
            {
                "name": "GitHub",
                "url": "https://github.com/fumitoh/modelx-cython",
                "icon": "fa-brands fa-github",
            },
        ],
    }
except ImportError:
    html_theme = "alabaster"
