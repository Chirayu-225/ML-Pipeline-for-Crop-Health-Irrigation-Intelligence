"""YAML config loading with attribute access, `_base_` inheritance and CLI overrides."""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Iterable

import yaml


class Config(dict):
    """A dict whose keys can also be read as attributes (recursively)."""

    def __getattr__(self, key: str) -> Any:
        try:
            return self[key]
        except KeyError as exc:  # pragma: no cover - trivial
            raise AttributeError(key) from exc

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(obj: Any) -> Any:
    if isinstance(obj, dict) and not isinstance(obj, Config):
        return Config({k: _wrap(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_wrap(v) for v in obj]
    return obj


def _unwrap(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _unwrap(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_unwrap(v) for v in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, val in override.items():
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], val)
        else:
            out[key] = copy.deepcopy(val)
    return out


def _load_raw(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    base_ref = raw.pop("_base_", None)
    if base_ref:
        base = _load_raw((path.parent / base_ref).resolve())
        raw = _deep_merge(base, raw)
    return raw


def set_by_path(cfg: dict, dotted: str, value: Any) -> None:
    """Set cfg['a']['b'] = value from the string 'a.b'."""
    parts = dotted.split(".")
    node = cfg
    for part in parts[:-1]:
        node = node.setdefault(part, Config())
    node[parts[-1]] = value


def load_config(path: str | Path, overrides: Iterable[str] | None = None) -> Config:
    """Load a YAML config. `overrides` are strings like 'classifier.epochs=3'."""
    cfg = _wrap(_load_raw(Path(path).resolve()))
    for item in overrides or []:
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"Override must look like key=value, got: {item!r}")
        set_by_path(cfg, key.strip(), yaml.safe_load(val))
    return cfg


def base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml", help="Path to YAML config")
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides, e.g. classifier.epochs=3 data.image_size=96",
    )
    return parser
