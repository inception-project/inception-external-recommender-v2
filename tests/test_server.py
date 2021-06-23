import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from galahad.server import GalahadServer
from galahad.server.classifier import Classifier
from galahad.server.dataclasses import Document, DocumentList
from galahad.server.util import get_dataset_folder, get_document_path
from tests.fixtures import TestClassifier

tmpdir: Optional[Path] = None


@pytest.fixture
def server():
    tmp = TemporaryDirectory()

    global tmpdir
    tmpdir = Path(tmp.name)

    server = GalahadServer(data_dir=tmpdir)

    yield server
    tmp.cleanup()


@pytest.fixture
def client(server):
    yield TestClient(server)


@pytest.fixture
def classifier():
    yield TestClassifier()


# Test


def test_ping(client: TestClient):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong"}


# PUT list_datasets


def test_list_datasets(client: TestClient):
    expected_dataset_names = [f"test_dataset{i+1}" for i in range(3)]

    for name in expected_dataset_names:
        response = client.put(f"/dataset/{name}")
        assert response.status_code == 204
        assert response.text == ""
        assert get_dataset_folder(tmpdir, name).is_dir()

    response = client.get("/dataset")
    assert response.status_code == 200
    assert response.json() == {"names": expected_dataset_names}


# PUT create_dataset


def test_create_dataset_when_dataset_does_not_already_exist(client: TestClient):
    response = client.put("/dataset/test_dataset")
    assert response.status_code == 204
    assert response.text == ""
    assert get_dataset_folder(tmpdir, "test_dataset").is_dir()


def test_create_dataset_when_dataset_exist_already(client: TestClient):
    response = client.put("/dataset/test_dataset")
    assert response.status_code == 204
    assert response.text == ""
    assert get_dataset_folder(tmpdir, "test_dataset").is_dir()

    response = client.put("/dataset/test_dataset")
    assert response.status_code == 409
    assert response.json() == {"detail": "Dataset with id [test_dataset] already exists."}


# GET list_documents_in_dataset


def test_list_documents_in_dataset_when_dataset_does_not_already_exist(client: TestClient):
    response = client.get("/dataset/test_dataset")

    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset with id [test_dataset] not found."}


def test_list_documents_in_dataset(client: TestClient):
    client.put("/dataset/test_dataset")

    expected_names = []
    expected_versions = []

    for i in range(3):
        name = f"test_document{i}"
        version = i

        expected_names.append(name)
        expected_versions.append(i)

        request = Document.Config.schema_extra["example"]
        request["version"] = version

        response = client.put(f"/dataset/test_dataset/{name}", json=request)

        assert response.status_code == 204
        assert response.text == ""

        p = get_document_path(tmpdir, "test_dataset", name)
        assert p.is_file()

    response = client.get("/dataset/test_dataset")
    document_list = DocumentList(**response.json())

    assert document_list.names == expected_names
    assert document_list.versions == expected_versions


# DELETE delete_dataset


def test_delete_dataset_when_dataset_does_not_already_exist(client: TestClient):
    response = client.delete("/dataset/test_dataset")
    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset with id [test_dataset] not found."}


def test_delete_dataset_when_dataset_exist_already(client: TestClient):
    p = get_dataset_folder(tmpdir, "test_dataset")
    client.put("/dataset/test_dataset")
    assert p.is_dir()

    for i in range(3):
        request = Document.Config.schema_extra["example"]
        response = client.put(f"/dataset/test_dataset/test_document{i}", json=request)
        assert response.status_code == 204

    response = client.delete("/dataset/test_dataset")
    assert response.status_code == 204
    assert response.text == ""
    assert not p.exists()


# PUT add_document_to_dataset


def test_add_document_to_dataset_when_dataset_does_not_already_exist(client: TestClient):
    request = Document.Config.schema_extra["example"]

    response = client.put("/dataset/test_dataset/test_document", json=request)
    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset with id [test_dataset] not found."}


def test_add_document_to_dataset_when_document_does_not_already_exist(client: TestClient):
    client.put("/dataset/test_dataset")

    request = Document.Config.schema_extra["example"]

    response = client.put("/dataset/test_dataset/test_document", json=request)
    assert response.status_code == 204
    assert response.text == ""

    p = get_document_path(tmpdir, "test_dataset", "test_document")

    assert p.is_file()

    with p.open() as f:
        document = json.load(f)

    assert document == request


# GET get_all_classifier_infos


def test_get_all_classifier_infos(server: GalahadServer, client: TestClient, classifier: Classifier):
    expected_infos = []

    for i in range(3):
        name = f"test_classifier_{i}"
        server.add_classifier(name, classifier)

        info = {"name": name}
        expected_infos.append(info)

    response = client.get("/classifier")

    assert response.status_code == 200
    assert response.json() == expected_infos


# GET get_classifier_info


def test_get_classifier(server: GalahadServer, client: TestClient, classifier: Classifier):
    expected_info = {"name": "test_classifier"}
    server.add_classifier("test_classifier", classifier)

    response = client.get("/classifier/test_classifier")

    assert response.status_code == 200
    assert response.json() == expected_info


def test_get_classifier_when_classifier_does_not_exist(
    server: GalahadServer, client: TestClient, classifier: Classifier
):
    expected_infos = []

    for i in range(3):
        name = f"test_classifier_{i}"
        server.add_classifier(name, classifier)

        info = {"name": name}
        expected_infos.append(info)

    response = client.get("/classifier/test_classifier")

    assert response.status_code == 404
    assert response.json() == {"detail": "Classifier with id [test_classifier] not found."}


# POST train_on_dataset


def test_train_on_dataset(server: GalahadServer, client: TestClient, classifier: Classifier):
    with client:
        # Add classifier
        server.add_classifier("test_classifier", classifier)

        # Add dataset
        client.put("/dataset/test_dataset")
        request = Document.Config.schema_extra["example"]
        response = client.put("/dataset/test_dataset/test_document", json=request)
        assert response.status_code == 204

        response = client.post("/classifier/test_classifier/test_model/train/test_dataset")
        assert response.status_code == 202
        assert response.text == ""

        model_path = classifier._get_model_path("test_model")

        # Wait for training to finish
        server.state.executor.shutdown()

        assert model_path.is_file()


def test_train_on_dataset_when_classifier_does_not_exist(client: TestClient):
    response = client.post("/classifier/test_classifier/test_model/train/test_dataset")

    assert response.status_code == 404
    assert response.json() == {"detail": "Classifier with id [test_classifier] not found."}


def test_train_on_dataset_when_dataset_does_not_exist(
    server: GalahadServer, client: TestClient, classifier: Classifier
):
    server.add_classifier("test_classifier", classifier)

    response = client.post("/classifier/test_classifier/test_model/train/test_dataset")

    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset with id [test_dataset] not found."}


# POST predict_on_document


def test_predict_on_document(server: GalahadServer, client: TestClient, classifier: Classifier):
    test_train_on_dataset(server, client, classifier)

    request = Document(**Document.Config.schema_extra["example"])
    response = client.post("/classifier/test_classifier/test_model/predict", json=request.dict())

    assert response.status_code == 200
    assert response.json() == request.dict()


def test_predict_on_document_when_classifier_does_not_exist(client: TestClient):
    request = Document.Config.schema_extra["example"]
    response = client.post("/classifier/test_classifier/test_model/predict", json=request)

    assert response.status_code == 404
    assert response.json() == {"detail": "Classifier with id [test_classifier] not found."}


def test_predict_on_document_when_model_does_not_exist(
    server: GalahadServer, client: TestClient, classifier: Classifier
):
    server.add_classifier("test_classifier", classifier)

    request = Document.Config.schema_extra["example"]
    response = client.post("/classifier/test_classifier/test_model/predict", json=request)

    assert response.status_code == 404
    assert response.json() == {"detail": "Model with id [test_model] not found."}
