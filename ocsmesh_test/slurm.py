"""Generate and submit SLURM scripts.

* :func:`generate` writes one script per (test, mode, method) benchmark case.
* :func:`render_job` writes a script for any other ``ocsmesh-test`` command
  (regression checks under ``srun``, per-tile timing, ...).

Everything cluster-specific comes from the ``cluster`` and ``modes``
sections of the config, so the same generator serves Gaea, Hercules or a
new machine. ``ocsmesh_path`` (optional) puts an OCSMesh checkout first on
``PYTHONPATH`` so a run can use a specific branch without reinstalling the
conda environment; the script prints which OCSMesh it imported.
"""

from __future__ import annotations

import logging
import re
import shlex
import stat
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from .config import Config

_logger = logging.getLogger(__name__)


def script_dir(cfg: Config) -> Path:
    return cfg.resolve("cluster.script_dir", f"slurm/{cfg.name}")


def log_dir(cfg: Config) -> Path:
    return cfg.resolve("cluster.log_dir", f"logs/{cfg.name}")


def resolve_ntasks(cfg: Config, mode_cfg) -> int:
    """``ntasks: auto`` = one worker rank per usable tile + the manager rank."""
    ntasks = mode_cfg["ntasks"]
    if str(ntasks).lower() != "auto":
        return int(ntasks)
    from .data import manifest  # local: needs the manifest on disk
    try:
        return manifest.count_usable(cfg) + 1
    except FileNotFoundError as exc:
        raise RuntimeError("ntasks: auto needs the manifest; run `ocsmesh-test data` first") from exc


def _script(cfg: Config, *, job: str, header: str, nodes: int, ntasks: int, cpus: int,
            walltime: str, command: str, work_dir: Path, mpi: bool,
            ntasks_per_node: Optional[int] = None) -> str:
    """Common SLURM script: resources, environment, OCSMesh selection, launch."""
    cl = cfg.get("cluster", {}) or {}
    logs = log_dir(cfg)
    tmpdir = work_dir / "tmp"
    if ntasks_per_node is None and nodes > 1:
        ntasks_per_node = -(-ntasks // nodes)

    sbatch = [
        f"#SBATCH --job-name={job}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --time={walltime}",
        f"#SBATCH --output={logs}/{job}_%j.out",
        f"#SBATCH --error={logs}/{job}_%j.err",
    ]
    if ntasks_per_node:
        sbatch.append(f"#SBATCH --ntasks-per-node={ntasks_per_node}")
    for key in ("account", "partition", "qos", "clusters", "mail-user"):
        val = cl.get(key.replace("-", "_"))
        if val:
            sbatch.append(f"#SBATCH --{key}={val}")
    if cl.get("mail_user"):
        sbatch.append("#SBATCH --mail-type=END,FAIL")
    if cl.get("exclusive", True):
        sbatch.append("#SBATCH --exclusive")
    sbatch.extend(cl.get("extra_sbatch", []) or [])

    env_lines = [f"module load {mod}" for mod in (cl.get("modules") or [])]
    env_lines += [f'export {k}="{v}"' for k, v in (cl.get("env") or {}).items()]  # expanded on the compute node
    if cl.get("conda_sh"):
        env_lines.append(f'source "{cl["conda_sh"]}"')
    if cl.get("conda_env"):
        env_lines.append(f'conda activate "{cl["conda_env"]}"')
    ocsmesh_path = cfg.get("ocsmesh_path")
    if ocsmesh_path:
        env_lines.append(f'export PYTHONPATH="{ocsmesh_path}${{PYTHONPATH:+:$PYTHONPATH}}"')
    env_lines += [
        "export OMP_NUM_THREADS=1", "export MKL_NUM_THREADS=1", "export OPENBLAS_NUM_THREADS=1",
        f'export TMPDIR="{tmpdir}"', f'export OCSMESH_SHARED_TMPDIR="{tmpdir}"',
    ]

    launcher = cl.get("launcher", "srun")
    launch = (f"{launcher} --nodes={nodes} --ntasks={ntasks} --cpus-per-task={cpus} "
              f"{cl.get('launcher_args', '--cpu-bind=cores')} {command}")
    checks = ['python -c "import ocsmesh, sys; print(\'ocsmesh from:\', ocsmesh.__file__)"']
    if mpi:
        checks.append(f'{launcher} -n 1 python -c "from mpi4py import MPI; '
                      f'print(\'mpi4py OK:\', MPI.Get_version())"')

    return f"""#!/bin/bash
{header}
{chr(10).join(sbatch)}

set -euo pipefail
{chr(10).join(env_lines)}

mkdir -p "{work_dir}" "{tmpdir}" "{logs}"
cd "{cfg.base_dir}"   # relative paths in arguments resolve against the repo root
echo "=== {job} job=${{SLURM_JOB_ID:-local}} nodes=${{SLURM_JOB_NUM_NODES:-1}} $(date)"
{chr(10).join(checks)}
{launch}
echo "=== DONE {job} $(date)"
"""


def render(cfg: Config, test_id: int, mode: str, method: str, hfun_only: bool = False) -> str:
    """Script for one benchmark case."""
    test = cfg.test(test_id)
    m = cfg.mode(mode)
    cl = cfg.get("cluster", {}) or {}
    suffix = "_h" if hfun_only else ""
    job = f"{cfg.name}_t{test['id']}_{mode}_{method[:1]}{suffix}"
    if cfg.is_mpi_mode(mode):
        ntasks, cpus = resolve_ntasks(cfg, m), int(m.get("cpus_per_task", 1))
    else:
        ntasks, cpus = 1, int(m.get("nprocs", 1))
    command = (f"ocsmesh-test run -c '{cfg.path.resolve()}' --test {test['id']} "
               f"--mode {mode} --method {method}" + (" --hfun-only" if hfun_only else ""))
    header = (f"# {cfg.name}: test {test['id']} ({test['name']}) | mode={mode} | method={method}"
              f"{' | hfun only' if hfun_only else ''}\n# {test.get('description', '')}")
    return _script(cfg, job=job, header=header, nodes=int(m.get("nodes", 1)), ntasks=ntasks,
                   cpus=cpus, walltime=m.get("walltime", cl.get("walltime", "08:00:00")),
                   command=command, work_dir=cfg.case_dir(test_id, mode, method, hfun_only),
                   mpi=cfg.is_mpi_mode(mode), ntasks_per_node=m.get("ntasks_per_node"))


def render_job(cfg: Config, name: str, args: Sequence[str], ntasks: int = 1, cpus: int = 1,
               nodes: int = 1, walltime: Optional[str] = None) -> str:
    """Script running ``ocsmesh-test <args>`` with the cluster environment."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", name):
        raise ValueError(f"job name must be [A-Za-z0-9_.-]+, got {name!r}")
    cl = cfg.get("cluster", {}) or {}
    command = "ocsmesh-test " + " ".join(shlex.quote(a) for a in args)
    return _script(cfg, job=f"{cfg.name}_{name}", header=f"# {cfg.name}: {command}",
                   nodes=nodes, ntasks=ntasks, cpus=cpus,
                   walltime=walltime or cl.get("walltime", "08:00:00"), command=command,
                   work_dir=cfg.results_root / "jobs" / name, mpi=ntasks > 1)


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


def generate(cfg: Config, tests: Optional[Iterable[int]] = None, modes: Optional[Iterable[str]] = None,
             methods: Optional[Iterable[str]] = None, hfun_only: bool = False) -> List[Path]:
    sdir = script_dir(cfg)
    log_dir(cfg).mkdir(parents=True, exist_ok=True)
    tests = [t["id"] for t in cfg.tests] if tests is None else list(tests)
    modes = list(cfg.modes) if modes is None else list(modes)
    methods = cfg.methods if methods is None else list(methods)
    suffix = "_hfun" if hfun_only else ""
    paths = []
    for tid in tests:
        for mode in modes:
            for method in methods:
                name = f"{cfg.name}_t{tid}_{cfg.test(tid)['name']}_{mode}_{method}{suffix}.sh"
                paths.append(_write(sdir / name, render(cfg, tid, mode, method, hfun_only)))
    _logger.info("%d scripts written to %s", len(paths), sdir)
    return paths


def generate_job(cfg: Config, name: str, args: Sequence[str], **resources) -> Path:
    log_dir(cfg).mkdir(parents=True, exist_ok=True)
    return _write(script_dir(cfg) / f"{cfg.name}_job_{name}.sh", render_job(cfg, name, args, **resources))


def submit(paths: Iterable[Path]) -> List[str]:
    job_ids = []
    for p in paths:
        res = subprocess.run(["sbatch", str(p)], capture_output=True, text=True, check=False)
        if res.returncode == 0:
            parts = res.stdout.split()
            jid = parts[3] if len(parts) >= 4 else parts[-1]
            job_ids.append(jid)
            _logger.info("submitted %s -> %s", p.name, jid)
        else:
            _logger.error("sbatch failed for %s: %s", p.name, res.stderr.strip())
    return job_ids
