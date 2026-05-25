import sys
import os

sys.path.insert(0, os.path.abspath("../../py_package"))

project = "robustrolling"
author = "Igor Ptak"
release = "0.2.0"
copyright = "2026, Igor Ptak"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

autodoc_mock_imports = ["robust_rolling_core"]

napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}

html_theme = "furo"
html_title = "robustrolling"
html_logo = "../_static/logo.svg"
html_favicon = "../_static/logo.svg"
html_static_path = ["../_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "light_css_variables": {
        "color-brand-primary": "#2E86AB",
        "color-brand-content": "#2E86AB",
        "color-highlight-on-target": "#e8f4f8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#5aafe0",
        "color-brand-content": "#5aafe0",
    },
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
}
