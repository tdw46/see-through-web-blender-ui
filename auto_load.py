from __future__ import annotations

import importlib
from typing import Dict, List, Sequence

_ORDERED_MODULES: List[str] = []
_MODULES: Dict[str, object] = {}


def set_modules(module_names: Sequence[str]) -> None:
    global _ORDERED_MODULES
    _ORDERED_MODULES = list(module_names)


def _package_name() -> str:
    package = __package__
    if not package:
        package = __name__.rpartition(".")[0]
    return package


def register() -> None:
    package = _package_name()
    for module_name in _ORDERED_MODULES:
        module = importlib.import_module(f".{module_name}", package)
        module = importlib.reload(module)
        _MODULES[module_name] = module
        if hasattr(module, "register"):
            module.register()


def unregister() -> None:
    for module_name in reversed(_ORDERED_MODULES):
        module = _MODULES.get(module_name)
        if module and hasattr(module, "unregister"):
            module.unregister()
    _MODULES.clear()


def get_module(name: str):
    module = _MODULES.get(name)
    if module is not None:
        return module

    for key, value in _MODULES.items():
        if key.rsplit(".", 1)[-1] == name:
            return value
    return None
