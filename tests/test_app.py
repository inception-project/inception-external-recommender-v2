import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pytest
from fastapi.testclient import TestClient

from galahad.classifier import Classifier
from galahad.dataclasses import Document, DocumentList
from galahad.server import GalahadServer
from galahad.util import get_datasets_folder, get_document_path

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
    class TestClassifier(Classifier):
        pass

    yield TestClassifier()


# Test


def test_ping(client: TestClient):
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong"}


# PUT create_dataset


def test_create_dataset_when_dataset_does_not_already_exist(client: TestClient):
    response = client.put("/dataset/test_dataset")
    assert response.status_code == 204
    assert response.text == ""
    assert get_datasets_folder(tmpdir, "test_dataset").is_dir()


def test_create_dataset_when_dataset_exist_already(client: TestClient):
    response = client.put("/dataset/test_dataset")
    assert response.status_code == 204
    assert response.text == ""
    assert get_datasets_folder(tmpdir, "test_dataset").is_dir()

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
    p = get_datasets_folder(tmpdir, "test_dataset")
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
