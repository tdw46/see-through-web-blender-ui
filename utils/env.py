from __future__ import annotations

import importlib
import sys
import zipfile
import importlib.util
import importlib.machinery
from pathlib import Path
from shutil import rmtree
from typing import Any

from . import paths

PSD_REQUIRED_MODULES = ("psd_tools", "PIL")
TRACE_REQUIRED_MODULES = ("vtracer",)


def bootstrap(configured_cache_dir: str = "") -> str:
    return str(paths.dependency_site_dir(configured_cache_dir))


def addon_package_id(package_name: str) -> str:
    parts = package_name.split(".")
    if len(parts) > 1:
        return ".".join(parts[:-1])
    return package_name


def _search_roots() -> tuple[Path, ...]:
    return (paths.dependency_site_dir(), paths.vendor_dir())


def _find_local_module_entry(module_name: str) -> Path | None:
    for root in _search_roots():
        package_dir = root / module_name
        init_file = package_dir / "__init__.py"
        if init_file.exists():
            return init_file

        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            ext_file = package_dir / f"{module_name}{suffix}"
            if ext_file.exists():
                return ext_file

        module_file = root / f"{module_name}.py"
        if module_file.exists():
            return module_file

        for suffix in importlib.machinery.EXTENSION_SUFFIXES:
            ext_file = root / f"{module_name}{suffix}"
            if ext_file.exists():
                return ext_file
    return None


def _load_local_module(module_name: str):
    if module_name in sys.modules:
        return sys.modules[module_name]

    entry = _find_local_module_entry(module_name)
    if entry is None:
        raise ModuleNotFoundError(module_name)

    is_package = entry.name == "__init__.py"
    kwargs = {"submodule_search_locations": [str(entry.parent)]} if is_package else {}
    spec = importlib.util.spec_from_file_location(module_name, entry, **kwargs)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load local module spec for {module_name} from {entry}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _preload_local_dependencies(module_name: str) -> None:
    dependency_map = {
        "psd_tools": ("typing_extensions", "attr", "attrs", "numpy", "PIL"),
        "vtracer": (),
    }
    for dependency in dependency_map.get(module_name, ()):
        try:
            importlib.import_module(dependency)
        except Exception:
            _load_local_module(dependency)


def can_import(module_name: str, configured_cache_dir: str = "") -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        try:
            _preload_local_dependencies(module_name)
            _load_local_module(module_name)
            return True
        except Exception:
            return False


def import_optional(module_name: str, configured_cache_dir: str = "") -> Any:
    try:
        return importlib.import_module(module_name)
    except Exception:
        _preload_local_dependencies(module_name)
        return _load_local_module(module_name)


def _bundled_wheel_candidates() -> list[Path]:
    wheels_root = paths.wheels_dir()
    if not wheels_root.exists():
        return []
    return sorted(path for path in wheels_root.glob("*.whl") if path.is_file())


def _matching_wheels(prefixes: tuple[str, ...]) -> list[Path]:
    matches: list[Path] = []
    for wheel in _bundled_wheel_candidates():
        normalized = wheel.name.lower().replace("-", "_")
        if any(normalized.startswith(prefix.lower().replace("-", "_")) for prefix in prefixes):
            matches.append(wheel)
    return matches


def psd_backend_assets() -> dict[str, object]:
    install_root = paths.dependency_site_dir()
    vendor_root = paths.vendor_dir()
    vendored = all(
        (install_root / module).exists() or (vendor_root / module).exists()
        for module in PSD_REQUIRED_MODULES
    ) and all(
        (install_root / module).exists() or (vendor_root / module).exists()
        for module in TRACE_REQUIRED_MODULES
    )
    psd_wheels = _matching_wheels(("psd_tools",))
    pillow_wheels = _matching_wheels(("pillow",))
    vtracer_wheels = _matching_wheels(("vtracer",))
    return {
        "install_root": install_root,
        "vendor_root": vendor_root,
        "wheels_root": paths.wheels_dir(),
        "all_wheels": _bundled_wheel_candidates(),
        "vendored": vendored,
        "psd_wheels": psd_wheels,
        "pillow_wheels": pillow_wheels,
        "vtracer_wheels": vtracer_wheels,
    }


def install_bundled_psd_backend() -> str:
    assets = psd_backend_assets()
    install_root = assets["install_root"]
    vendor_root = assets["vendor_root"]
    all_wheels = assets["all_wheels"]
    psd_wheels = assets["psd_wheels"]
    pillow_wheels = assets["pillow_wheels"]
    vtracer_wheels = assets["vtracer_wheels"]

    if assets["vendored"]:
        if all((vendor_root / module).exists() for module in PSD_REQUIRED_MODULES):
            return f"PSD backend already available from {vendor_root}"
        return f"PSD backend already available from {install_root}"

    if not psd_wheels or not pillow_wheels or not vtracer_wheels:
        raise RuntimeError(
            "Missing bundled import dependency wheels. Add compatible `psd_tools`, `Pillow`, and `vtracer` wheels under "
            f"{assets['wheels_root']} or vendor extracted packages under {assets['vendor_root']}."
        )

    install_root.mkdir(parents=True, exist_ok=True)

    for child in list(install_root.iterdir()):
        if child.is_dir():
            rmtree(child)
        else:
            child.unlink()

    for wheel in all_wheels:
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(install_root)

    bootstrap()
    for module_name in PSD_REQUIRED_MODULES:
        importlib.invalidate_caches()
        import_optional(module_name)
    return f"Installed PSD backend from bundled wheels into {install_root}"


def ensure_psd_backend(configured_cache_dir: str = "") -> None:
    if not can_import("psd_tools", configured_cache_dir) or not can_import("vtracer", configured_cache_dir):
        assets = psd_backend_assets()
        raise RuntimeError(
            "PSD import backend is not available. Put extracted `psd_tools`/`PIL`/`vtracer` packages under "
            f"{assets['vendor_root']} or compatible `psd_tools`, `Pillow`, and `vtracer` wheels under {assets['wheels_root']}, "
            "then use the bundled PSD install action in the add-on preferences."
        )


def psd_backend_status(configured_cache_dir: str = "") -> str:
    if can_import("psd_tools", configured_cache_dir) and can_import("vtracer", configured_cache_dir):
        return "ready"
    assets = psd_backend_assets()
    if assets["vendored"]:
        return "vendored-but-unloaded"
    if assets["psd_wheels"] and assets["pillow_wheels"] and assets["vtracer_wheels"]:
        return "bundled wheels available"
    return "missing"
