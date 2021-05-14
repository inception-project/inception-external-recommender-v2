from fastapi import FastAPI, Path

from galahad.model import *

app = FastAPI()


# Meta


@app.get("/ping")
def ping():
    return {"ping": "pong!"}


# Dataset


@app.put("/dataset/{dataset_id}")
def create_dataset(dataset_id: str = Path(..., title="Identifier of the dataset that should be created")):
    pass


@app.put("/dataset/{dataset_id}/{document_id}")
def add_document_to_dataset(
    request: DocumentAddRequest,
    dataset_id: str = Path(..., title="Identifier of the dataset to add to"),
    document_id: str = Path(..., title="Identifier of the document to add"),
):
    pass


@app.get("/dataset/{dataset_id}", response_model=DocumentListResponse)
def list_documents_in_dataset(
    dataset_id: str = Path(..., title="Identifier of the dataset whose documents should be listed")
):
    pass


@app.delete("/dataset/{dataset_id}")
def delete_dataset(dataset_id: str = Path(..., title="Identifier of the dataset that should be deleted")):
    pass


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
    dataset_id: str = Path(..., title="Identifier of the dataset that should be used for training"),
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
