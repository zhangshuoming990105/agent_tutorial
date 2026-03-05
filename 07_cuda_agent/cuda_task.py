"""
CUDA task workspace lifecycle manager.

Handles setting up an isolated working directory for each CUDA optimization task
by copying the fixed infrastructure (binding system, compile/verify/profile utils)
and the task-specific model.py into a fresh workspace.

Also provides resolve_task_path() to accept multiple task specifier formats:
  - Direct directory path:  task/example_axpby
  - Dataset level/id:       level1/3  or  level1/003
  - Global numeric ID:      42
"""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
import shutil
from pathlib import Path


_SELF_DIR = Path(__file__).resolve().parent
_TEMPLATE_DIR = _SELF_DIR / "template"
_DATASET_DIR = _SELF_DIR / "dataset"

_active_workspace: Path | None = None

# Loaded lazily on first use
_dataset_index: dict | None = None
_level_counts: list[tuple[str, int]] | None = None


def _load_index() -> dict:
    global _dataset_index, _level_counts
    if _dataset_index is not None:
        return _dataset_index

    index_file = _DATASET_DIR / "index.json"
    if not index_file.is_file():
        _dataset_index = {}
        _level_counts = []
        return _dataset_index

    with open(index_file, encoding="utf-8") as f:
        _dataset_index = json.load(f)

    counts: dict[str, int] = {}
    for key in _dataset_index:
        level = key.split("/")[0]
        counts[level] = counts.get(level, 0) + 1
    _level_counts = [(lvl, counts[lvl]) for lvl in sorted(counts)]
    return _dataset_index


def _get_level_counts() -> list[tuple[str, int]]:
    _load_index()
    return _level_counts or []


def resolve_task_path(spec: str) -> Path:
    """Resolve a task specifier to an absolute directory containing model.py.

    Accepted formats:
      - Existing directory path (absolute or relative) with model.py
      - "level{N}/{id}" e.g. "level1/3" or "level1/003"
      - Plain integer for global numeric ID (1-based across levels)
    """
    spec = spec.strip()

    # 1) Direct directory path
    candidate = Path(spec)
    if not candidate.is_absolute():
        candidate = (_SELF_DIR / spec).resolve()
    if candidate.is_dir() and (candidate / "model.py").is_file():
        return candidate

    # 2) level{N}/{id} format
    m = re.match(r"^(level\d+)/(\d+)$", spec)
    if m:
        level, num = m.group(1), int(m.group(2))
        task_id = f"{level}/{num:03d}"
        task_dir = _DATASET_DIR / level / f"{num:03d}"
        if task_dir.is_dir() and (task_dir / "model.py").is_file():
            return task_dir
        raise FileNotFoundError(
            f"Dataset task not found: {task_id} (looked in {task_dir})"
        )

    # 3) Global numeric ID
    if spec.isdigit():
        global_id = int(spec)
        if global_id < 1:
            raise ValueError(f"Global task ID must be >= 1, got {global_id}")
        offset = global_id
        for level, count in _get_level_counts():
            if offset <= count:
                task_dir = _DATASET_DIR / level / f"{offset:03d}"
                if task_dir.is_dir() and (task_dir / "model.py").is_file():
                    return task_dir
                raise FileNotFoundError(
                    f"Dataset task not found: {level}/{offset:03d}"
                )
            offset -= count
        total = sum(c for _, c in _get_level_counts())
        raise ValueError(
            f"Global task ID {global_id} out of range (max {total})"
        )

    raise FileNotFoundError(
        f"Cannot resolve task spec '{spec}': not a directory, not a level/id, not a number"
    )


def list_tasks(level_filter: str | None = None) -> str:
    """Return a formatted listing of available dataset tasks."""
    index = _load_index()
    if not index:
        return "No dataset found. Run build_dataset.py first."

    lines = []
    current_level = ""
    count = 0
    for task_id in sorted(index.keys()):
        level = task_id.split("/")[0]
        if level_filter and level != level_filter:
            continue
        if level != current_level:
            if current_level:
                lines.append("")
            current_level = level
            level_total = sum(1 for k in index if k.startswith(level + "/"))
            lines.append(f"=== {level} ({level_total} tasks) ===")
        entry = index[task_id]
        desc = entry.get("description", "")[:60]
        score = entry.get("score", "?")
        lines.append(f"  {task_id}  score={score}  {desc}")
        count += 1

    global_info = []
    running = 0
    for lvl, cnt in _get_level_counts():
        global_info.append(f"{lvl}: global {running+1}-{running+cnt}")
        running += cnt

    lines.append("")
    lines.append(f"Total: {count} tasks  ({', '.join(global_info)})")
    return "\n".join(lines)


def setup_workspace(task_dir: str | Path, workdir: str | Path) -> Path:
    """Initialise a task workspace.

    1. Copy template infrastructure (binding.cpp, binding_registry.h, utils/).
    2. Copy model.py from *task_dir*.
    3. Create empty kernels/ directory.

    Returns the resolved workspace path.
    """
    global _active_workspace

    task_dir = Path(task_dir).resolve()
    workdir = Path(workdir).resolve()

    model_file = task_dir / "model.py"
    if not model_file.is_file():
        raise FileNotFoundError(f"Task directory missing model.py: {task_dir}")

    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    # Copy fixed infrastructure from template
    for item in _TEMPLATE_DIR.iterdir():
        dest = workdir / item.name
        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # Copy task model
    shutil.copy2(model_file, workdir / "model.py")

    # Create empty kernels directory
    (workdir / "kernels").mkdir(exist_ok=True)

    _active_workspace = workdir
    return workdir


def get_workspace_path() -> Path | None:
    """Return the active workspace path, or None if not initialised."""
    return _active_workspace


def workspace_summary(workdir: str | Path | None = None) -> str:
    """Return a human-readable summary of the workspace contents."""
    root = Path(workdir) if workdir else _active_workspace
    if root is None or not root.exists():
        return "(no active workspace)"

    lines = [f"Workspace: {root}"]
    for item in sorted(root.iterdir()):
        if item.name.startswith(".") or item.name == "__pycache__":
            continue
        if item.is_dir():
            children = [p.name for p in sorted(item.iterdir()) if not p.name.startswith(".")]
            lines.append(f"  {item.name}/  ({len(children)} items: {', '.join(children[:8])})")
        else:
            lines.append(f"  {item.name}  ({item.stat().st_size} bytes)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# History library
# ---------------------------------------------------------------------------

def save_to_history(
    task_dir: str | Path,
    workdir: str | Path,
    profile_result: dict | None = None,
) -> Path:
    """Save a successful implementation to the task's history library.

    Copies model_new.py and kernels/ from *workdir* into
    ``<task_dir>/history/<timestamp>/`` along with a result.json.
    Returns the history entry path.
    """
    task_dir = Path(task_dir).resolve()
    workdir = Path(workdir).resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    entry = task_dir / "history" / timestamp
    entry.mkdir(parents=True, exist_ok=True)

    model_new = workdir / "model_new.py"
    if model_new.is_file():
        shutil.copy2(model_new, entry / "model_new.py")

    kernels_src = workdir / "kernels"
    if kernels_src.is_dir():
        kernels_dst = entry / "kernels"
        if kernels_dst.exists():
            shutil.rmtree(kernels_dst)
        shutil.copytree(
            kernels_src, kernels_dst,
            ignore=shutil.ignore_patterns("*.hip", "*_hip.cpp", "*.o"),
        )

    result = {
        "verify": "pass",
        "timestamp": timestamp,
        "profile": profile_result or {},
    }
    (entry / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return entry


def load_history_prompt(task_dir: str | Path) -> str:
    """Load the latest history entry and format it as LLM context.

    Returns an empty string if no history exists.
    """
    task_dir = Path(task_dir).resolve()
    history_dir = task_dir / "history"
    if not history_dir.is_dir():
        return ""

    entries = sorted(
        [d for d in history_dir.iterdir() if d.is_dir()],
        key=lambda p: p.name,
        reverse=True,
    )
    if not entries:
        return ""

    latest = entries[0]
    result_file = latest / "result.json"
    model_new_file = latest / "model_new.py"

    lines = ["## Previous Successful Implementation (for reference)",
             "A prior run on this task produced a working implementation. "
             "Use it as a starting point or reference — do NOT copy it blindly "
             "if you have a better approach."]

    if result_file.is_file():
        try:
            result = json.loads(result_file.read_text(encoding="utf-8"))
            profile = result.get("profile", {})
            if profile:
                lines.append(f"\nPrior profiling: "
                             f"Baseline={profile.get('baseline_us', '?')}us, "
                             f"Compile={profile.get('compile_us', '?')}us, "
                             f"CUDA={profile.get('cuda_us', '?')}us")
        except (json.JSONDecodeError, OSError):
            pass

    if model_new_file.is_file():
        code = model_new_file.read_text(encoding="utf-8", errors="replace")
        lines.append(f"\nPrior model_new.py:\n```python\n{code.strip()}\n```")

    kernel_files = []
    kernels_dir = latest / "kernels"
    if kernels_dir.is_dir():
        for f in sorted(kernels_dir.iterdir()):
            if f.suffix in (".cu", ".cpp") and f.is_file():
                kernel_files.append(f)

    for kf in kernel_files[:3]:
        code = kf.read_text(encoding="utf-8", errors="replace")
        if len(code) > 4000:
            code = code[:4000] + "\n... (truncated)"
        lines.append(f"\nPrior {kf.name}:\n```cpp\n{code.strip()}\n```")

    return "\n".join(lines)
