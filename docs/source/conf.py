# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import re

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import sys

sys.path.insert(0, "../../deepof")
sys.path.insert(0, "../../")
sys.path.insert(0, "../")
sys.path.insert(0, "./")

# -- Project information -----------------------------------------------------

project = "deepof"
copyright = "2024, Lucas Miranda"
author = "Lucas Miranda"

# The full version, including alpha/beta/rc tags
release = "0.7.1"


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx_rtd_theme",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "jupyter_sphinx",
    "nbsphinx",
]
autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "logo_only": True,
    "display_version": False,
}


def setup(app):
    app.add_css_file("custom.css")


# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# Change the icon displayed in the sidebar and the web browser tab for the docs
html_favicon = "deepof.ico"
html_logo = "deepof_sidebar.ico"

# Autodoc default configuration
autodoc_default_options = {"member-order": "bysource", "special-members": "__init__"}
