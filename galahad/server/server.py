import asyncio
import pathlib
import re
import shutil
from concurrent.futures import ProcessPoolExecutor
from typing import Callable

from fastapi import FastAPI, HTTPException, Path, Response, status
from starlette.background import BackgroundTasks

from galahad.server.classifier import (Classifier, ClassifierStore,
                                       train_classifier)
from galahad.server.dataclasses import *
from galahad.server.util import (get_dataset_folder, get_datasets_folder,
                                 get_document_path)

# This regex forbids two consecutive dots so that ../foo does not work
# to discovery files outside of the document folder
PATH_REGEX = r"^[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*$"


def check_naming_is_ok_regex(name: str):
    if not re.match(PATH_REGEX, name):
        raise ValueError(f'The name "{name}" is invalid. ' "Please look at the documentation for correct naming.")


class GalahadServer(FastAPI):
    """Creates a Galahad server instance."""

    def __init__(self, title: str = "Galahad Server", data_dir: pathlib.Path = None) -> None:
        """Creates a Galahad server instance."""
        super().__init__(title=title)

        if data_dir is None:
            data_dir = pathlib.Path.cwd() / "galahad_data"

        data_dir.mkdir(exist_ok=True, parents=True)
        get_datasets_folder(data_dir).mkdir(exist_ok=True, parents=True)
        get_datasets_folder(data_dir).mkdir(exist_ok=True, parents=True)

        self._classifier_store = ClassifierStore(data_dir / "models")

        self.state.data_dir = data_dir
        self.state.lock_dir = data_dir / "locks"
        self.state.classifier_store = self._classifier_store

        _register_routes(self)

    def add_classifier(self, name: str, classifier: Classifier):
        check_naming_is_ok_regex(name)

        document_path = get_document_path(self.state.data_dir, "classifier", name)
        document_path.unlink(missing_ok=True)
        self._classifier_store.add_classifier(name, classifier)


def _register_routes(app: FastAPI):
    data_dir: pathlib.Path = app.state.data_dir
    lock_directory = app.state.lock_dir
    classifier_store: ClassifierStore = app.state.classifier_store

    # Scheduling
    # https://stackoverflow.com/questions/63169865/how-to-do-multiprocessing-in-fastapi

    @app.on_event("startup")
    async def startup_event():
        app.state.executor = ProcessPoolExecutor()

    @app.on_event("shutdown")
    async def on_shutdown():
        app.state.executor.shutdown()

    async def run_in_different_process(fn: Callable, *args):
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(app.state.executor, fn, *args)

    # Meta

    @app.get("/ping")
    def ping():
        return {"ping": "pong"}

    # Dataset

    @app.get(
        "/dataset",
        response_model=DatasetList,
        responses={status.HTTP_200_OK: {"description": "Returns list of all available datasets."}},
        status_code=status.HTTP_200_OK,
    )
    def list_datasets():
        """Lists dataset names managed by this server."""
        dataset_names = []

        for p in sorted(get_datasets_folder(data_dir).iterdir()):
            dataset_names.append(p.name)

        return DatasetList(names=dataset_names)

    @app.put(
        "/dataset/{dataset_id}",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Dataset created."},
            status.HTTP_409_CONFLICT: {"description": "Dataset already exists."},
        },
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def create_dataset(
        dataset_id: str = Path(..., title="Identifier of the dataset that should be created", regex=PATH_REGEX),
    ):
        """Creates a dataset with the given `dataset_id`. Does nothing and returns `409` if it already existed."""
        dataset_folder = get_dataset_folder(data_dir, dataset_id)

        if dataset_folder.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=f"Dataset with id [{dataset_id}] already exists."
            )

        dataset_folder.mkdir(parents=True)
        return Response(content="", status_code=status.HTTP_204_NO_CONTENT)

    @app.delete(
        "/dataset/{dataset_id}",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Dataset deleted."},
            status.HTTP_404_NOT_FOUND: {"description": "Dataset not found."},
        },
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_dataset(
        dataset_id: str = Path(..., title="Identifier of the dataset that should be deleted", regex=PATH_REGEX),
    ):
        """Deletes the dataset with the given `dataset_id` and its documents."""
        dataset_folder = get_dataset_folder(data_dir, dataset_id)

        if not dataset_folder.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset with id [{dataset_id}] not found."
            )

        shutil.rmtree(dataset_folder)

        return Response(content="", status_code=status.HTTP_204_NO_CONTENT)

    @app.get(
        "/dataset/{dataset_id}",
        response_model=DocumentList,
        responses={
            status.HTTP_200_OK: {"description": "Returns list of documents in dataset."},
            status.HTTP_404_NOT_FOUND: {"description": "Dataset not found."},
        },
        status_code=status.HTTP_200_OK,
    )
    def list_documents_in_dataset(
        dataset_id: str = Path(
            ..., title="Identifier of the dataset whose documents should be listed", regex=PATH_REGEX
        ),
    ):
        """Lists documents in the dataset with the given `dataset_id`."""
        dataset_folder = get_dataset_folder(data_dir, dataset_id)

        if not dataset_folder.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset with id [{dataset_id}] not found."
            )

        names = []
        versions = []

        for p in sorted(dataset_folder.iterdir()):
            document: Document = Document.parse_file(p)
            names.append(p.name)
            versions.append(document.version)

        return DocumentList(names=names, versions=versions)

    @app.put(
        "/dataset/{dataset_id}/{document_id}",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Document added."},
            status.HTTP_404_NOT_FOUND: {"description": "Dataset not found."},
        },
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def add_document_to_dataset(
        request: Document,
        dataset_id: str = Path(..., title="Identifier of the dataset to add to", regex=PATH_REGEX),
        document_id: str = Path(..., title="Identifier of the document to add", regex=PATH_REGEX),
    ):
        """Adds a document to an already existing dataset. Overwrites a document if it already existed."""
        dataset_folder = get_dataset_folder(data_dir, dataset_id)

        if not dataset_folder.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset with id [{dataset_id}] not found."
            )

        document_path = get_document_path(data_dir, dataset_id, document_id)
        with document_path.open("w", encoding="utf-8") as f:
            f.write(request.json(skip_defaults=True))

        return Response(content="", status_code=status.HTTP_204_NO_CONTENT)

    @app.delete(
        "/dataset/{dataset_id}/{document_id}",
        responses={
            status.HTTP_204_NO_CONTENT: {"description": "Document deleted."},
            status.HTTP_404_NOT_FOUND: {"description": "Dataset not found."},
        },
        status_code=status.HTTP_204_NO_CONTENT,
    )
    def delete_document_from_dataset(
        dataset_id: str = Path(..., title="Identifier of the dataset to delete from", regex=PATH_REGEX),
        document_id: str = Path(..., title="Identifier of the document to delete", regex=PATH_REGEX),
    ):
        """Deletes a document from a dataset. Does nothing if the document did not exist."""
        dataset_folder = get_dataset_folder(data_dir, dataset_id)

        if not dataset_folder.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset with id [{dataset_id}] not found."
            )

        document_path = get_document_path(data_dir, dataset_id, document_id)
        document_path.unlink(missing_ok=True)

        return Response(content="", status_code=status.HTTP_204_NO_CONTENT)

    # Model

    @app.get(
        "/classifier",
        response_model=List[ClassifierInfo],
        responses={
            status.HTTP_200_OK: {"description": "Returns list of documents in dataset."},
        },
        status_code=status.HTTP_200_OK,
    )
    def get_all_classifier_infos():
        """Gets the classifier info for all classifiers managed by this server."""
        return classifier_store.get_classifier_infos()

    @app.get(
        "/classifier/{classifier_id}",
        response_model=ClassifierInfo,
        responses={
            status.HTTP_200_OK: {"description": "Returns the classifier info of the requested classifier."},
            status.HTTP_404_NOT_FOUND: {"description": "Classifier not found."},
        },
        status_code=status.HTTP_200_OK,
    )
    def get_classifier_info(
        classifier_id: str = Path(..., title="Identifier of the classifier whose info to query", regex=PATH_REGEX)
    ):
        """Gets the classifier info for the requested classifier id if it exists."""
        classifier_info = classifier_store.get_classifier_info(classifier_id)

        if classifier_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Classifier with id [{classifier_id}] not found."
            )

        return classifier_info

    # Train

    @app.post(
        "/classifier/{classifier_id}/{model_id}/train/{dataset_id}",
        responses={
            status.HTTP_202_ACCEPTED: {"description": "Training started."},
            status.HTTP_404_NOT_FOUND: {"description": "Classifier or dataset not found."},
            status.HTTP_429_TOO_MANY_REQUESTS: {"description": "Already training."},
        },
    )
    def train_on_dataset(
        background_tasks: BackgroundTasks,
        classifier_id: str = Path(..., title="Name of the classifier that should be trained.", regex=PATH_REGEX),
        model_id: str = Path(..., title="Name of the model that should be trained.", regex=PATH_REGEX),
        dataset_id: str = Path(
            ..., title="Identifier of the dataset that should be used for training", regex=PATH_REGEX
        ),
    ):
        classifier = classifier_store.get_classifier(classifier_id)
        if classifier is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Classifier with id [{classifier_id}] not found."
            )

        dataset_folder = get_dataset_folder(data_dir, dataset_id)
        if not dataset_folder.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset with id [{dataset_id}] not found."
            )

        background_tasks.add_task(
            run_in_different_process, train_classifier, classifier, dataset_folder, model_id, lock_directory
        )

        return Response(content="", status_code=status.HTTP_202_ACCEPTED)

    # Prediction

    @app.post(
        "/classifier/{classifier_id}/{model_id}/predict",
        response_model=Document,
        responses={
            status.HTTP_200_OK: {"description": "Prediction successful."},
            status.HTTP_404_NOT_FOUND: {"description": "Classifier or model not found."},
        },
    )
    def predict_for_document(
        request: Document,
        classifier_id: str = Path(
            ..., title="Name of the classifier that should be used for prediction", regex=PATH_REGEX
        ),
        model_id: str = Path(..., title="Identifier of the model that should be used for prediction", regex=PATH_REGEX),
    ):
        classifier = classifier_store.get_classifier(classifier_id)
        if classifier is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Classifier with id [{classifier_id}] not found."
            )

        result = classifier.predict(model_id, request)
        if result is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=f"Model with id [{model_id}] not found.")

        return result

    @app.post("/classifier/{classifier_id}/{model_id}/predict/{dataset_id}/{document_id}")
    def predict_on_dataset(
        classifier_id: str = Path(..., title="Identifier of the model that should be used for prediction"),
        dataset_id: str = Path(..., title="Identifier of the dataset that should be used for prediction"),
        document_id: str = Path(
            ..., title="Identifier of the document in the given dataset that should be used for prediction"
        ),
    ):
        return {"classifier_id": classifier_id, "dataset_id": dataset_id, "document_id": document_id}
