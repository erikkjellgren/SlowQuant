#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# python setup.py build_ext --inplace
# sphinx-build documentation docs

import sphinx_rtd_theme

# -- General configuration ------------------------------------------------
# If your documentation needs a minimal Sphinx version, state it here.
# needs_sphinx = '1.0'
extensions = ['sphinx.ext.imgmath']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = 'SLOWQUANT'
copyright = '2017, Erik Kjellgren'
author = 'Erik Kjellgren'

# -- Options for HTML output ----------------------------------------------
html_theme = "sphinx_rtd_theme"
html_theme_path = ["_themes", ]
html_static_path = ['_static']
html_logo = "_static/slowquant.png"


# -- Options for LaTeX output ---------------------------------------------
latex_elements = {'preamble': r'\pdfimageresolution=144'}

latex_documents = [
    (master_doc, 'sphinx-example.tex', 'sphinx-example Documentation',
     'Firstname Lastname', 'manual'),]


# -- Options for manual page output ---------------------------------------
man_pages = [
    (master_doc, 'sphinx-example', 'sphinx-example Documentation',
     [author], 1)]


# -- Options for Texinfo output -------------------------------------------
texinfo_documents = [
    (master_doc, 'sphinx-example', 'sphinx-example Documentation',
     author, 'sphinx-example', 'One line description of project.',
     'Miscellaneous'),]


