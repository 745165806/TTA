"""Git provenance helpers for P2 validation and confirmatory runs."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def git_commit(root=ROOT):
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True,
        check=False)
    value = completed.stdout.strip()
    if completed.returncode != 0 or len(value) != 40:
        raise ValueError("cannot resolve current git commit")
    return value


def require_clean_tree(root=ROOT):
    completed = subprocess.run(
        ["git", "status", "--porcelain"], cwd=root, capture_output=True, text=True,
        check=False)
    if completed.returncode != 0 or completed.stdout.strip():
        raise ValueError("validation evidence requires a clean git worktree")


def path_introduction_commit(path, root=ROOT):
    relative = str(Path(path).resolve().relative_to(Path(root).resolve()))
    completed = subprocess.run(
        ["git", "log", "--diff-filter=A", "--format=%H", "--", relative],
        cwd=root, capture_output=True, text=True, check=False)
    commits = [line for line in completed.stdout.splitlines() if line]
    if completed.returncode != 0 or not commits:
        raise ValueError("cannot resolve historical result commit: %s" % path)
    return commits[-1]
