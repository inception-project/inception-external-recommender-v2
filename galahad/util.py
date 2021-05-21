from pathlib import Path


def path_is_subfolder(parent: Path, child: Path) -> bool:
    """
    Checks whether `parent` is really a parent of `child`.
    This is mainly used to prevent clients sending path names like ../../bar that can
    potentially escape the data folder and cause harm.
    """
    return parent.resolve() in child.resolve().parents
