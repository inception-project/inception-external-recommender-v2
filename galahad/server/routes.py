import pathlib
import shutil
from concurrent.futures import ProcessPoolExecutor

from fastapi import FastAPI, HTTPException, Path, Response, status

from galahad.server.classifier import ClassifierStore
from galahad.server.dataclasses import *
from galahad.server.util import get_datasets_folder, get_document_path

# This regex forbids two consecutive dots so that ../foo does not work
# to discovery files outside of the document folder
PATH_REGEX = r"^[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*$"


def register_routes(app: FastAPI):
    data_dir: pathlib.Path = app.state.data_dir
    classifier_store: ClassifierStore = app.state.classifier_store

    # Meta

    @app.get("/ping")
    def ping():
        return {"ping": "pong"}

    # Dataset

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
        """ Creates a dataset with the given `dataset_id`. Does nothing and returns `409` if it already existed. """
        dataset_folder = get_datasets_folder(data_dir, dataset_id)

        if dataset_folder.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT, detail=f"Dataset with id [{dataset_id}] already exists."
            )

        dataset_folder.mkdir(parents=True)
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
        """ Lists documents in the dataset with the given `dataset_id`. """
        dataset_folder = get_datasets_folder(data_dir, dataset_id)

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
        """ Deletes the dataset with the given `dataset_id`  and its documents. """
        dataset_folder = get_datasets_folder(data_dir, dataset_id)

        if not dataset_folder.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset with id [{dataset_id}] not found."
            )

        shutil.rmtree(dataset_folder)

        return Response(content="", status_code=status.HTTP_204_NO_CONTENT)

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
        """ Adds a document to an already existing dataset. Overwrites a document if it already existed. """
        dataset_folder = get_datasets_folder(data_dir, dataset_id)

        if not dataset_folder.is_dir():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Dataset with id [{dataset_id}] not found."
            )

        document_path = get_document_path(data_dir, dataset_id, document_id)
        with document_path.open("w", encoding="utf-8") as f:
            f.write(request.json(skip_defaults=True))

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
        """ Gets the classifier info for all classifiers managed by this server. """
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
    def get_classifier_info(classifier_id: str = Path(..., title="Identifier of the classifier whose info to query")):
        """ Gets the classifier info for the requested classifier id if it exists. """
        classifier_info = classifier_store.get_classifier_info(classifier_id)

        if classifier_info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail=f"Classifier with id [{classifier_id}] not found."
            )

        return classifier_info

    # Train

    @app.post("/classifier/{classifier_id}/train/{dataset_id}")
    def train_on_dataset(
        classifier_id: str = Path(..., title="Identifier of the model that should be trained"),
        dataset_id: str = Path(
            ..., title="Identifier of the dataset that should be used for training", regex=PATH_REGEX
        ),
    ):
        pass

    # Prediction

    @app.post("/model/{classifier_id}/predict", response_model=PredictionResponse)
    def predict_for_document(
        request: PredictionRequest,
        classifier_id: str = Path(..., title="Identifier of the model that should be used for prediction"),
    ):
        return {"classifier_id": classifier_id}

    @app.post("/model/{classifier_id}/predict/{dataset_id}/{document_id}")
    def predict_on_dataset(
        classifier_id: str = Path(..., title="Identifier of the model that should be used for prediction"),
        dataset_id: str = Path(..., title="Identifier of the dataset that should be used for prediction"),
        document_id: str = Path(
            ..., title="Identifier of the document in the given dataset that should be used for prediction"
        ),
    ):
        return {"classifier_id": classifier_id, "dataset_id": dataset_id, "document_id": document_id}

    # https://stackoverflow.com/questions/63169865/how-to-do-multiprocessing-in-fastapi

    @app.on_event("startup")
    async def startup_event():
        app.state.executor = ProcessPoolExecutor()

    @app.on_event("shutdown")
    async def on_shutdown():
        app.state.executor.shutdown()
