"""
pyossim Python package entry point

This re-exports symbols from the compiled extension module so that 'import pyossim' works the same as 'from pyossim import pyossim'.
"""

from .pyossim import *

__all__ = [
    *[n for n in dir() if not n.startswith("__")],
]
