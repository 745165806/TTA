from pathlib import Path


def list_unpublished_partial_directories(parent, output_name):
    """Read-only recovery inventory; callers decide whether to retain/remove."""
    root = Path(parent)
    return sorted(str(path) for path in root.glob(".%s.*" % output_name) if path.is_dir())
