"""
HybridAdapter
=============

At the moment triton-anchor uses a single triton-shared-based lowering path.
The Hybrid adapter is therefore a thin alias over TritonSharedAdapter so call
sites can keep using the symbolic "hybrid" mode.
"""

from __future__ import annotations

from typing import Any, List

from .base import ILinalgOptAdapter
from .triton_shared_adapter import TritonSharedAdapter


class HybridAdapter(ILinalgOptAdapter):
    def name(self) -> str:
        return "hybrid"

    def convert(self, ttir_module: Any, metadata: dict, context: Any = None) -> Any:
        return TritonSharedAdapter(mode="unstructured").convert(ttir_module, metadata, context)

    def get_output_dialects(self) -> List[str]:
        return TritonSharedAdapter(mode="unstructured").get_output_dialects()
