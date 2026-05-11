"""
Dataset preparation scripts and preprocessing helpers.
"""

from .arithmetic import generate_equation
from .arithmetic_error import inject_error
from .sudoku import preprocess_sudoku

__all__ = [
    "generate_equation",
    "inject_error",
    "preprocess_sudoku",
]
