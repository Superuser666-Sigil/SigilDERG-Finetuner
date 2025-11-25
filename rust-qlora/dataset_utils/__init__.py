"""
Dataset utilities for SigilDERG-Finetuner.

Currently exposes the DatasetLoader abstraction that encapsulates cached vs
streaming dataset logic.

Copyright (c) 2025 Dave Tofflemire, SigilDERG Project
Version: 2.7.2
"""

from .loader import DatasetLoader

__all__ = ["DatasetLoader"]

