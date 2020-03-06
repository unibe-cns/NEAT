from datetime import date
from sphinx_gallery.sorting import ExplicitOrder
import sphinx_rtd_theme
from warnings import filterwarnings

filterwarnings(
    "ignore", message="Matplotlib is currently using agg", category=UserWarning
)

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))

# General configuration
# ---------------------


# Trick to get the jupyter nbs to the documentation
import shutil
project_root = '..'
print("Copy example notebooks into docs/_examples")


def all_but_ipynb(dir, contents):
    result = []
    for c in contents:
        if os.path.isfile(os.path.join(dir, c)) and (not c.endswith(".ipynb")):
            result += [c]
    return result


shutil.rmtree(os.path.join(project_root, "docs/_examples"), ignore_errors=True)
shutil.copytree(os.path.join(project_root, "examples"),
                os.path.join(project_root, "docs/_examples"),
                ignore=all_but_ipynb)

# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx_gallery.gen_gallery",
    "nb2plots",
    "texext",
    "nbsphinx"
]

# https://github.com/sphinx-gallery/sphinx-gallery
# sphinx_gallery_conf = {
#     # path to your examples scripts
#     "examples_dirs": "../examples",
#     "subsection_order": ExplicitOrder(
#         [
#             "../examples/basic",
#             "../examples/drawing",
#             "../examples/graph",
#             "../examples/algorithms",
#             "../examples/advanced",
#             "../examples/3d_drawing",
#             "../examples/pygraphviz",
#             "../examples/javascript",
#             "../examples/jit",
#             "../examples/applications",
#             "../examples/subclass",
#         ]
#     ),
#     # path where to save gallery generated examples
#     "gallery_dirs": "auto_examples",
#     "backreferences_dir": "modules/generated",
# }

# generate autosummary pages
autosummary_generate = True

# Napoleon settings
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = None


# Add any paths that contain templates here, relative to this directory.
# templates_path = ['']

suppress_warnings = ["ref.citation", "ref.footnote"]

# The suffix of source filenames.
source_suffix = ".rst"

# The encoding of source files.
source_encoding = "utf-8"

# The master toctree document.
master_doc = "index"

# Do not include release announcement template
exclude_patterns = ["release/release_template.rst", "neat/fittools", "simtools/neuron", '_build', '**.ipynb_checkpoints']

# General substitutions.
project = "NEAT"
copyright = f"2020-{date.today().year}, NEAT Developers"

# The default replacements for |version| and |release|, also used in various
# other places throughout the built documents.
#
# The short X.Y version.
#import neat

#version = neat.__version__
version = '0.1'
# The full version, including dev info
#release = neat.__version__.replace("_", "")
release = "0.1"

# There are two options for replacing |today|: either, you set today to some
# non-false value, then it is used:
# today = ''
# Else, today_fmt is used as the format for a strftime call.
# today_fmt = '%B %d, %Y'

# List of documents that shouldn't be included in the build.
# unused_docs = ['']

# If true, '()' will be appended to :func: etc. cross-reference text.
# add_function_parentheses = True

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = False

# show_authors = True

# The name of the Pygments (syntax highlighting) style to use.
# pygments_style = 'friendly'
pygments_style = "sphinx"

# A list of prefixs that are ignored when creating the module index. (new in Sphinx 0.6)
modindex_common_prefix = ["neat."]

#doctest_global_setup = "import neat"

# Options for HTML output
# -----------------------


html_theme = "sphinx_rtd_theme"
html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

html_theme_options = {
    "canonical_url": "https://neat.github.io/documentation/stable/",
    "navigation_depth": 3,
}


# The style sheet to use for HTML and HTML Help pages. A file of that name
# must exist either in Sphinx' static/ path, or in one of the custom paths
# given in html_static_path.
# html_style = ''

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

# If not '', a 'Last updated on:' timestamp is inserted at every page bottom,
# using the given strftime format.
html_last_updated_fmt = "%b %d, %Y"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
# html_use_smartypants = True

# Content template for the index page.
# html_index = 'index.html'

# Custom sidebar templates, maps page names to templates.
# html_sidebars = {}

# Additional templates that should be rendered to pages, maps page names to
# templates.
# html_additional_pages = {'': ''}

# If true, the reST sources are included in the HTML build as _sources/<name>.
html_copy_source = False

html_use_opensearch = "http://neat.github.io"

# Output file base name for HTML help builder.
htmlhelp_basename = "NEAT"

# Options for LaTeX output
# ------------------------

# The paper size ('letter' or 'a4').
latex_paper_size = "letter"

# The font size ('10pt', '11pt' or '12pt').
# latex_font_size = '10pt'

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title, author, document class [howto/manual]).
latex_documents = [
    (
        "reference/index",
        "neat_reference.tex",
        "NEAT Reference",
        "Willem Wybo",
        "manual",
        1,
    )
]

latex_appendices = ["tutorial"]

# Intersphinx mapping
intersphinx_mapping = {
    "https://docs.python.org/2/": None,
    "https://docs.scipy.org/doc/numpy/": None,
}

# The reST default role (used for this markup: `text`) to use for all
# documents.
default_role = "obj"

numpydoc_show_class_members = False


# Add the 'copybutton' javascript, to hide/show the prompt in code examples
def setup(app):
    app.add_javascript("copybutton.js")
