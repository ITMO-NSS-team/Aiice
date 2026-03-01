"""
.. include:: ../../README.md
   :start-line: 1
<!-- The comment bellow is required as a marker for pdoc. See .github/doc/module.html.jinja2 -->
<!-- MAIN_README_PDOC -->
.. include:: ../../CONTRIBUTE.md
"""

from aiice import core, loader, metrics, preprocess, runner

# visible modules to pdoc
__all__ = ["core", "loader", "metrics", "preprocess", "runner"]
