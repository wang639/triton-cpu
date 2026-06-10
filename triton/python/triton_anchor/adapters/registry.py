"""
Adapter Registry
=================

Manages discovery and selection of TTIR → Linalg adapters.
Selection is driven by ``HWCapability.ptr_model`` and optional user override.
"""

from __future__ import annotations

import importlib.metadata
import logging
from typing import Dict, Optional, TYPE_CHECKING

from .base import ITritonToLinalgAdapter

if TYPE_CHECKING:
    from ..hw_capability import HWCapability

logger = logging.getLogger(__name__)


class AdapterRegistry:
    _adapters: Dict[str, ITritonToLinalgAdapter] = {}
    _discovered: bool = False

    @classmethod
    def register(cls, adapter: ITritonToLinalgAdapter) -> None:
        name = adapter.name()
        if name in cls._adapters:
            logger.warning(f"Adapter '{name}' already registered, overwriting")
        cls._adapters[name] = adapter

    @classmethod
    def discover(cls) -> None:
        if cls._discovered:
            return
        cls._discovered = True
        try:
            eps = importlib.metadata.entry_points(group="triton.adapters")
        except TypeError:
            eps = importlib.metadata.entry_points().get("triton.adapters", [])
        for ep in eps:
            try:
                adapter_cls = ep.load()
                cls.register(adapter_cls())
            except Exception as e:
                logger.warning(f"Failed to load adapter entry_point '{ep.name}': {e}")

    @classmethod
    def get(cls, name: str) -> Optional[ITritonToLinalgAdapter]:
        cls.discover()
        return cls._adapters.get(name)

    @classmethod
    def get_adapter(cls, hw: HWCapability) -> ITritonToLinalgAdapter:
        cls.discover()
        if hw.preferred_adapter:
            adapter = cls._adapters.get(hw.preferred_adapter)
            if adapter:
                return adapter
            raise AdapterNotFoundError(
                f"Preferred adapter '{hw.preferred_adapter}' not found. "
                f"Available: {list(cls._adapters.keys())}"
            )

        model_to_adapter = {
            "structured": "triton-shared",
            "axis_info": "triton-shared",
            "hybrid": "triton-shared",
        }
        adapter_name = model_to_adapter.get(hw.ptr_model)
        if adapter_name and adapter_name in cls._adapters:
            return cls._adapters[adapter_name]

        if cls._adapters:
            return next(iter(cls._adapters.values()))

        raise AdapterNotFoundError(
            f"No adapters available for ptr_model='{hw.ptr_model}'. "
            f"Install the triton-anchor wheel with triton-shared support."
        )

    @classmethod
    def list_adapters(cls) -> Dict[str, str]:
        cls.discover()
        return {name: type(adapter).__name__ for name, adapter in cls._adapters.items()}

    @classmethod
    def reset(cls) -> None:
        cls._adapters.clear()
        cls._discovered = False


class AdapterNotFoundError(Exception):
    pass


def get_adapter(hw: HWCapability) -> ITritonToLinalgAdapter:
    return AdapterRegistry.get_adapter(hw)
