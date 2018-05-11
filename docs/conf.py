# -*- coding: utf-8 -*-
#
# PMDA documentation build configuration file
#
# This file is execfile()d with the current directory set to its
# containing dir.
#
# Note that not all possible configuration values are present in this
# autogenerated file.
#
# All configuration values have a default; values that are commented out
# serve to show the default.

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- General configuration ------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.intersphinx',
              'sphinx.ext.mathjax', 'sphinx.ext.viewcode',
              'sphinx.ext.napoleon', 'sphinx.ext.todo',
              'sphinx_sitemap',
              'alabaster']

mathjax_path = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.0/MathJax.js?config=TeX-AMS-MML_HTMLorMML'

# for sitemap with https://github.com/jdillard/sphinx-sitemap
site_url = "https://www.mdanalysis.org/pmda/"

# https://stackoverflow.com/questions/5599254/how-to-use-sphinxs-autodoc-to-document-a-classs-init-self-method
autoclass_content = 'both'

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# The master toctree document.
master_doc = 'index'

# General information about the project.
project = u'PMDA'
author = u'Max Linke, Oliver Beckstein'
copyright = u'2018, ' + author


# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The full version, including alpha/beta/rc tags.
release = __import__('pmda').__version__
# The short X.Y version.
version = ".".join(release.split('.')[:2])

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = None

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This patterns also effect to html_static_path and html_extra_path
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False


# -- Options for HTML output ----------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# styles/fonts to match http://mdanalysis.org (see public/css)
#
# /* MDAnalysis orange: #FF9200 */
# /* MDAnalysis gray: #808080 */
# /* MDAnalysis white: #FFFFFF */
# /* MDAnalysis black: #000000 */

color = {'orange': '#FF9200',
         'gray': '#808080',
         'white': '#FFFFFF',
         'black': '#000000',}

html_theme_options = {
    'logo' : "logos/pmda-logo.png",
    'logo_name': True,
    'logo_text_align': "left",
    'description': "Parallel Molecular Dynamics Analysis",
    'github_user': "MDAnalysis",
    'github_repo': "pmda",
    'github_button': False,
    'github_banner': True,
    'extra_nav_links': {
        "MDAnalysis": "https://www.mdanalysis.org",
        "Dask": "https://dask.pydata.org",
        "distributed": "https://distributed.readthedocs.io/",
#        "mpi4py": "https://mpi4py.readthedocs.io",
    },
    'show_related': True,
    'fixed_sidebar': False,
    'sidebar_includehidden': True,
    'sidebar_collapse': True,
    # style
    'link': color['orange'],
    'link_hover': color['orange'],
    'gray_1': color['gray'],
    'narrow_sidebar_bg': color['gray'],
    'narrow_sidebar_fg': color['white'],
    # typography
    #'font_size': 17,
    'font_family': "'PT Sans', Helvetica, Arial, 'sans-serif'",
    'head_font_family': "",
    'code_font_size': "smaller",
    'code_font_family': "Menlo, Monaco, 'Courier New', monospace",
    'caption_font_size': "smaller",
}

# Add any paths that contain custom themes here, relative to this directory.
#html_theme_path = [alabaster.get_path()]

# The name for this set of Sphinx documents.  If None, it defaults to
# "<project> v<release> documentation".
#html_title = None

# A shorter title for the navigation bar.  Default is the same as html_title.
#html_short_title = None

# The name of an image file (relative to this directory) to place at the top
# of the sidebar. -- use theme (only enable if NOT using html_sidebars)
##html_logo = "_static/logos/pmda-logo.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "_static/logos/mdanalysis-logo.ico"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# Custom sidebar templates, maps document names to template names.
# alabaster sidebars
html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
    ]
}

# If true, an OpenSearch description file will be output, and all pages will
# contain a <link> tag referring to it.  The value of this option must be the
# base URL from which the finished HTML is served.
html_use_opensearch = 'https://www.mdanalysis.org/pmda'


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = 'pmdadoc'


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',

    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',

    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',

    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, 'pmda.tex', u'pmda Documentation',
     author, 'manual'),
]


# -- Options for manual page output ---------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, 'pmda', u'pmda Documentation',
     [author], 1)
]


# -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (master_doc, 'pmda', u'pmda Documentation',
     author, 'pmda', 'Parallel Molecular Dynamics Analysis',
     'Miscellaneous'),
]


# Configuration for intersphinx: refer to the Python standard library
# and other packages used by MDAnalysis
intersphinx_mapping = {'https://docs.python.org/': None,
                       'https://www.mdanalysis.org/docs/': None,
                       'https://dask.pydata.org/en/latest/': None,
                       'https://distributed.readthedocs.io/en/latest/': None,
                       }

