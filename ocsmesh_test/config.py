"""YAML configuration loading.

A config file describes one benchmark: domain, DEM sources, mesh recipe,
test matrix, execution modes and the cluster (SLURM) settings. String
values may contain ``${ENV_VAR}`` and ``~``; relative paths are resolved
against ``base_dir`` (default: the directory that holds the config file).

Expansion rules:

* ``cluster.env`` and ``cluster.modules`` are **not** expanded: they are
  written verbatim into the SLURM scripts and expanded on the compute
  node, after ``module load`` (e.g. ``${CRAY_LD_LIBRARY_PATH}``).
* A value that still references an undefined variable after expansion is
  replaced by ``None`` and recorded in :attr:`Config.unresolved`, so
  optional settings (mail address) are simply dropped and required ones
  (GEBCO file) fail with a message naming the variable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

MPI_MODE_PREFIX = "mpi"
_VERBATIM_KEYS = ("cluster.env", "cluster.modules")
_UNRESOLVED_RE = re.compile(r"\$\{?([A-Za-z_][A-Za-z0-9_]*)\}?")


def _expand(value: Any, key: str, unresolved: Dict[str, str]) -> Any:
    if key in _VERBATIM_KEYS:
        return value
    if isinstance(value, str):
        out = os.path.expanduser(os.path.expandvars(value))
        m = _UNRESOLVED_RE.search(out)
        if m:
            unresolved[key] = m.group(1)
            return None
        return out
    if isinstance(value, list):
        return [_expand(v, f"{key}[{i}]", unresolved) for i, v in enumerate(value)]
    if isinstance(value, dict):
        return {k: _expand(v, f"{key}.{k}" if key else str(k), unresolved) for k, v in value.items()}
    return value


@dataclass
class Config:
    """Thin wrapper around the YAML dict with path resolution helpers."""

    raw: Dict[str, Any]
    path: Path
    unresolved: Dict[str, str] = field(default_factory=dict)
    base_dir: Path = field(init=False)

    def require(self, dotted: str) -> Any:
        """Like ``self[dotted]`` but names the missing environment variable."""
        if dotted in self.unresolved:
            raise KeyError(f"config key {dotted} needs environment variable "
                           f"${self.unresolved[dotted]}, which is not set")
        return self[dotted]

    def __post_init__(self) -> None:
        # Relative paths resolve against the config's directory, or against
        # ``base_dir`` (itself relative to the config's directory) when given.
        cfg_dir = self.path.resolve().parent
        base = self.raw.get("base_dir")
        self.base_dir = (cfg_dir / base).resolve() if base else cfg_dir

    # -- generic access -----------------------------------------------------
    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for key in dotted.split("."):
            if not isinstance(node, dict) or key not in node:
                return default
            node = node[key]
        return node

    def __getitem__(self, dotted: str) -> Any:
        value = self.get(dotted, _MISSING)
        if value is _MISSING:
            raise KeyError(f"missing config key: {dotted}")
        return value

    def resolve(self, dotted: str, default: Optional[str] = None) -> Path:
        """Return a config path resolved against the config directory."""
        value = self.get(dotted, default)
        if value is None:
            raise KeyError(f"missing config path: {dotted}")
        p = Path(value)
        return p if p.is_absolute() else (self.base_dir / p).resolve()

    # -- commonly used sections --------------------------------------------
    @property
    def name(self) -> str:
        return self.raw.get("name", self.path.stem)

    @property
    def results_root(self) -> Path:
        return self.resolve("results_root", f"results/{self.name}")

    @property
    def tests(self) -> List[Dict[str, Any]]:
        return list(self.raw.get("tests", []))

    def test(self, test_id: int) -> Dict[str, Any]:
        for t in self.tests:
            if int(t["id"]) == int(test_id):
                return t
        raise KeyError(f"unknown test id {test_id}")

    @property
    def methods(self) -> List[str]:
        return list(self.raw.get("methods", ["exact"]))

    @property
    def modes(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.raw.get("modes", {"serial": {"nprocs": 1}}))

    def mode(self, name: str) -> Dict[str, Any]:
        if name not in self.modes:
            raise KeyError(f"unknown mode {name!r}; known: {sorted(self.modes)}")
        return self.modes[name]

    @staticmethod
    def is_mpi_mode(name: str) -> bool:
        return name.startswith(MPI_MODE_PREFIX)

    def ocsmesh_mode(self, name: str) -> str:
        """Map a benchmark mode name to OCSMesh's execution_mode."""
        if self.is_mpi_mode(name):
            return "mpi"
        if name == "parallel":
            return "parallel"
        return "serial"

    def nprocs(self, name: str) -> int:
        m = self.mode(name)
        if self.is_mpi_mode(name):
            return int(m.get("cpus_per_task", 1))
        return int(m.get("nprocs", 1))

    def case_dir(self, test_id: int, mode: str, method: str) -> Path:
        t = self.test(test_id)
        return self.results_root / f"test{t['id']}_{t['name']}" / mode / method


_MISSING = object()


def load_config(path: os.PathLike | str) -> Config:
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    unresolved: Dict[str, str] = {}
    expanded = _expand(raw, "", unresolved)
    return Config(raw=expanded, path=p, unresolved=unresolved)
