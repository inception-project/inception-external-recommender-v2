from pathlib import Path


class ServerError(Exception):
    pass


# Wrong status code is returned
class StatusCodeError(ServerError):
    pass


# Response is not correct
class ResponseError(ServerError):
    pass


# Data does not exist but should
class DataNonExistentError(ServerError):
    pass


# Invalid name for file, folder or classifier
class NamingError(ServerError):
    pass


def get_datasets_folder(data_dir: Path) -> Path:
    result = data_dir / "datasets"
    if not path_is_parent(data_dir, result):
        raise NamingError("Naming is invalid. Please look at the documentation for correct naming.")
    return result


def get_dataset_folder(data_dir: Path, dataset_id: str) -> Path:
    result = get_datasets_folder(data_dir) / dataset_id
    if not path_is_parent(data_dir, result):
        raise NamingError(
            'Naming for the dataset "' + dataset_id + '" is invalid. '
            "Please look at the documentation for correct naming."
        )
    return result


def get_document_path(data_dir: Path, dataset_id: str, document_name: str) -> Path:
    result = get_dataset_folder(data_dir, dataset_id) / document_name
    if not path_is_parent(data_dir, result):
        raise NamingError(
            'Naming for the dataset "' + dataset_id + '" or for the document "' + document_name + '" is invalid. '
            "Please look at the documentation for correct naming."
        )
    return result


def path_is_parent(parent: Path, child: Path) -> bool:
    """
    Checks whether `parent` is really a parent of `child`.
    This is mainly used to prevent clients sending path names like ../../bar that can
    potentially escape the data folder and cause harm.
    """
    return parent.resolve() in child.resolve().parents
