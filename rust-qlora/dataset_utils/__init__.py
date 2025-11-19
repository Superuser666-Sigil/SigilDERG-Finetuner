"""
Dataset utilities for SigilDERG-Finetuner.

Currently exposes the DatasetLoader abstraction that encapsulates cached vs
streaming dataset logic.
"""

from .loader import DatasetLoader

__all__ = ["DatasetLoader"]

