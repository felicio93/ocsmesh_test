"""Generate and submit SLURM scripts, one per (test, mode, method).

Everything cluster-specific comes from the ``cluster`` and ``modes``
sections of the config, so the same generator serves Gaea, Hercules or a
new machine.
"""

from __future__ import annotations

import logging
import stat
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

from .config import Config

_logger = logging.getLogger(__name__)


def script_dir(cfg: Config) -> Path:
    return cfg.resolve("cluster.script_dir", f"slurm/{cfg.name}")


def log_dir(cfg: Config) -> Path:
    return cfg.resolve("cluster.log_dir", f"logs/{cfg.name}")


def render(cfg: Config, test_id: int, mode: str, method: str) -> str:
    test = cfg.test(test_id)
    m = cfg.mode(mode)
    cl = cfg.get("cluster", {}) or {}
    out_dir = cfg.case_dir(test_id, mode, method)
    tmpdir = out_dir / "tmp"
    logs = log_dir(cfg)
    job = f"{cfg.name}_t{test['id']}_{mode}_{method[:1]}"

    nodes = int(m.get("nodes", 1))
    if cfg.is_mpi_mode(mode):
        ntasks = int(m["ntasks"])
        cpus = int(m.get("cpus_per_task", 1))
    else:
        ntasks = 1
        cpus = int(m.get("nprocs", 1))
    per_node = m.get("ntasks_per_node")
    if per_node is None and nodes > 1:
        per_node = -(-ntasks // nodes)

    sbatch = [
        f"#SBATCH --job-name={job}",
        f"#SBATCH --nodes={nodes}",
        f"#SBATCH --ntasks={ntasks}",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --time={m.get('walltime', cl.get('walltime', '08:00:00'))}",
        f"#SBATCH --output={logs}/{job}_%j.out",
        f"#SBATCH --error={logs}/{job}_%j.err",
    ]
    if per_node:
        sbatch.append(f"#SBATCH --ntasks-per-node={per_node}")
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
    env_lines += [f"export {k}={v}" for k, v in (cl.get("env") or {}).items()]
    if cl.get("conda_sh"):
        env_lines.append(f'source "{cl["conda_sh"]}"')
    if cl.get("conda_env"):
        env_lines.append(f'conda activate "{cl["conda_env"]}"')
    env_lines += [
        "export OMP_NUM_THREADS=1", "export MKL_NUM_THREADS=1", "export OPENBLAS_NUM_THREADS=1",
        f'export TMPDIR="{tmpdir}"', f'export OCSMESH_SHARED_TMPDIR="{tmpdir}"',
    ]

    launcher = cl.get("launcher", "srun")
    run_cmd = (f"ocsmesh-test run -c '{cfg.path.resolve()}' --test {test['id']} "
               f"--mode {mode} --method {method}")
    srun = (f"{launcher} --nodes={nodes} --ntasks={ntasks} --cpus-per-task={cpus} "
            f"{cl.get('launcher_args', '--cpu-bind=cores')} {run_cmd}")
    mpi_check = ""
    if cfg.is_mpi_mode(mode):
        mpi_check = (f'{launcher} -n 1 python -c "from mpi4py import MPI; '
                     f'print(\'mpi4py OK:\', MPI.Get_version())"\n')

    return f"""#!/bin/bash
# {cfg.name}: test {test['id']} ({test['name']}) | mode={mode} | method={method}
# {test.get('description', '')}
{chr(10).join(sbatch)}

set -euo pipefail
{chr(10).join(env_lines)}

mkdir -p "{out_dir}" "{tmpdir}" "{logs}"
echo "=== {job} job=${{SLURM_JOB_ID:-local}} nodes=${{SLURM_JOB_NUM_NODES:-1}} $(date)"
{mpi_check}{srun}
echo "=== DONE {job} $(date)"
"""


def generate(cfg: Config, tests: Optional[Iterable[int]] = None, modes: Optional[Iterable[str]] = None,
             methods: Optional[Iterable[str]] = None) -> List[Path]:
    sdir = script_dir(cfg)
    sdir.mkdir(parents=True, exist_ok=True)
    log_dir(cfg).mkdir(parents=True, exist_ok=True)
    tests = [t["id"] for t in cfg.tests] if tests is None else list(tests)
    modes = list(cfg.modes) if modes is None else list(modes)
    methods = cfg.methods if methods is None else list(methods)
    paths = []
    for tid in tests:
        for mode in modes:
            for method in methods:
                p = sdir / f"{cfg.name}_t{tid}_{cfg.test(tid)['name']}_{mode}_{method}.sh"
                p.write_text(render(cfg, tid, mode, method))
                p.chmod(p.stat().st_mode | stat.S_IEXEC)
                paths.append(p)
    _logger.info("%d scripts written to %s", len(paths), sdir)
    return paths


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
