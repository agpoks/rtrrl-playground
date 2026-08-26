import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

project = "rtrrl-playground"
copyright = "2026, agpoks"
author = "agpoks"

extensions = [
    "myst_parser",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "matplotlib.sphinxext.plot_directive",
    "sphinxcontrib.bibtex",
]

source_suffix = {".rst": "restructuredtext", ".md": "markdown"}
myst_enable_extensions = ["colon_fence", "dollarmath", "amsmath"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_logo = "_static/logo.svg"
html_favicon = "_static/logo.svg"
html_theme_options = {"logo_only": True}

bibtex_bibfiles = ["../../papers/references.bib"]
bibtex_default_style = "alpha"
bibtex_reference_style = "author_year"

plot_include_source = True
plot_html_show_source_link = False
plot_html_show_formats = False
plot_formats = [("png", 110)]
plot_rcparams = {"figure.autolayout": True}
