import json
import shutil
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException, Path, Response, status

from galahad.config import Settings
from galahad.model import *
from galahad.util import path_is_subfolder

app = FastAPI()

# This regex forbids two consecutive dots so that ../foo does not work
# to discovery files outside of the document folder
PATH_REGEX = r"^[a-zA-Z0-9_]+(?:\.[a-zA-Z0-9_]+)*$"


@lru_cache()
def get_settings():
    return Settings()


# Meta


@app.get("/ping")
def ping():
    return {"ping": "pong"}


# Dataset


@app.put(
    "/dataset/{dataset_id}",
    responses={204: {"description": "Dataset created."}, 409: {"description": "Dataset already exists."}},
    status_code=204,
)
def create_dataset(
    dataset_id: str = Path(..., title="Identifier of the dataset that should be created", regex=PATH_REGEX),
    settings: Settings = Depends(get_settings),
):
    """ Creates a dataset with the given `dataset_id`. Does nothing and returns `409` if it already existed. """
    dataset_folder = settings.data_dir / dataset_id
    assert path_is_subfolder(settings.data_dir, dataset_folder)

    if dataset_folder.exists():
        raise HTTPException(status_code=409, detail=f"Dataset with id [{dataset_id}] already exists")

    dataset_folder.mkdir(parents=True)
    return Response(content="", status_code=status.HTTP_204_NO_CONTENT)


@app.get(
    "/dataset/{dataset_id}",
    response_model=DocumentList,
    responses={
        200: {"description": "Returns list of documents in dataset."},
        404: {"description": "Dataset not found."},
    },
    status_code=200,
)
def list_documents_in_dataset(
    dataset_id: str = Path(..., title="Identifier of the dataset whose documents should be listed", regex=PATH_REGEX),
    settings: Settings = Depends(get_settings),
):
    """ Lists documents in the dataset with the given `dataset_id`. """
    dataset_folder = settings.data_dir / dataset_id
    assert path_is_subfolder(settings.data_dir, dataset_folder)

    if not dataset_folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset with id [{dataset_id}] not found")

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
        204: {"description": "Dataset deleted."},
        404: {"description": "Dataset not found."},
    },
    status_code=204,
)
def delete_dataset(
    dataset_id: str = Path(..., title="Identifier of the dataset that should be deleted", regex=PATH_REGEX),
    settings: Settings = Depends(get_settings),
):
    """ Deletes the dataset with the given `dataset_id`  and its documents. """
    dataset_folder = settings.data_dir / dataset_id
    assert path_is_subfolder(settings.data_dir, dataset_folder)

    if not dataset_folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset with id [{dataset_id}] not found")

    shutil.rmtree(dataset_folder)

    return Response(content="", status_code=status.HTTP_204_NO_CONTENT)


@app.put(
    "/dataset/{dataset_id}/{document_id}",
    responses={204: {"description": "Document added."}, 404: {"description": "Dataset not found."}},
    status_code=204,
)
def add_document_to_dataset(
    request: Document,
    dataset_id: str = Path(..., title="Identifier of the dataset to add to", regex=PATH_REGEX),
    document_id: str = Path(..., title="Identifier of the document to add", regex=PATH_REGEX),
    settings: Settings = Depends(get_settings),
):
    """ Adds a document to an already existing dataset. Overwrites a document if it already existed. """
    dataset_folder = settings.data_dir / dataset_id
    assert path_is_subfolder(settings.data_dir, dataset_folder)

    if not dataset_folder.is_dir():
        raise HTTPException(status_code=404, detail=f"Dataset with id [{dataset_id}] not found")

    document_path = dataset_folder / document_id
    with document_path.open("w", encoding="utf-8") as f:
        f.write(request.json())

    return Response(content="", status_code=status.HTTP_204_NO_CONTENT)


# Model


@app.get("/model", response_model=List[ModelMetaData])
def get_all_model_infos():
    pass


@app.get("/model/{model_id}", response_model=ModelMetaData)
def get_model_info(model_id: str = Path(..., title="Identifier of the model whose info to query")):
    pass


@app.delete("/model/{model_id}")
def delete_model(model_id: str = Path(..., title="Identifier of the model to delete")):
    pass


# Train


@app.post("/model/{model_id}/train/{dataset_id}")
def train_on_dataset(
    model_id: str = Path(..., title="Identifier of the model that should be trained"),
    dataset_id: str = Path(..., title="Identifier of the dataset that should be used for training", regex=PATH_REGEX),
):
    return {"model_id": model_id, "dataset_id": dataset_id, "document_id": document_id}


# Prediction


@app.post("/model/{model_id}/predict", response_model=PredictionResponse)
def predict_for_document(
    request: PredictionRequest,
    model_id: str = Path(..., title="Identifier of the model that should be used for prediction"),
):
    return {"model_id": model_id}


@app.post("/model/{model_id}/predict/{dataset_id}/{document_id}")
def predict_on_dataset(
    model_id: str = Path(..., title="Identifier of the model that should be used for prediction"),
    dataset_id: str = Path(..., title="Identifier of the dataset that should be used for prediction"),
    document_id: str = Path(
        ..., title="Identifier of the document in the given dataset that should be used for prediction"
    ),
):
    return {"model_id": model_id, "dataset_id": dataset_id, "document_id": document_id}
