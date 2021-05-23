from pathlib import Path


def get_datasets_folder(data_dir: Path, dataset_id: str) -> Path:
    result = data_dir / "datasets" / dataset_id
    assert path_is_parent(data_dir, result)
    return result


def get_document_path(data_dir: Path, dataset_id: str, document_name: str) -> Path:
    result = get_datasets_folder(data_dir, dataset_id) / document_name
    assert path_is_parent(data_dir, result)
    return result


def path_is_parent(parent: Path, child: Path) -> bool:
    """
    Checks whether `parent` is really a parent of `child`.
    This is mainly used to prevent clients sending path names like ../../bar that can
    potentially escape the data folder and cause harm.
    """
    return parent.resolve() in child.resolve().parents
