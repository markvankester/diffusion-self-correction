"""
Compatibility entry point for MDLM inference.

The implementation lives in ``MDLM.inference`` so CLI parsing, model loading,
sampling, metrics, and Sudoku visualization stay in separate modules.
"""

from __future__ import annotations

import sys
from pathlib import Path


if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from MDLM.inference.cli import main
from MDLM.inference.model_loading import load_model
from MDLM.inference.runner import run_prompts


__all__ = ["load_model", "run_prompts", "main"]


if __name__ == "__main__":
    main()
