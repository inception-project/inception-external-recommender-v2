import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pytest
from fastapi.testclient import TestClient

import galahad.app as main
from galahad import config
from galahad.app import app
from galahad.model import Document, DocumentList
from galahad.util import get_datasets_folder, get_document_path

tmpdir: Optional[Path] = None


@pytest.fixture(autouse=True)
def build_tmp_dir():
    tmp = TemporaryDirectory()
    global tmpdir
    tmpdir = Path(tmp.name)
    yield
    tmp.cleanup()


def get_settings_override():
    return config.Settings(data_dir=tmpdir)


client = TestClient(app)
app.dependency_overrides[main.get_settings] = get_settings_override

# Test


def test_ping():
    response = client.get("/ping")
    assert response.status_code == 200
    assert response.json() == {"ping": "pong"}


# PUT create_dataset


def test_create_dataset_dataset_does_not_exist_already():
    response = client.put("/dataset/test_dataset")
    assert response.status_code == 204
    assert response.text == ""
    assert get_datasets_folder(tmpdir, "test_dataset").is_dir()


def test_create_dataset_dataset_exist_already():
    response = client.put("/dataset/test_dataset")
    assert response.status_code == 204
    assert response.text == ""
    assert get_datasets_folder(tmpdir, "test_dataset").is_dir()

    response = client.put("/dataset/test_dataset")
    assert response.status_code == 409
    assert response.json() == {"detail": "Dataset with id [test_dataset] already exists"}


# GET list_documents_in_dataset


def test_list_documents_in_dataset_dataset_does_not_exist_already():
    response = client.get("/dataset/test_dataset")

    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset with id [test_dataset] not found"}


def test_list_documents_in_dataset():
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


def test_delete_dataset_dataset_does_not_exist_already():
    response = client.delete("/dataset/test_dataset")
    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset with id [test_dataset] not found"}


def test_delete_dataset_dataset_exist_already():
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


def test_add_document_to_dataset_dataset_does_not_exist_already():
    request = Document.Config.schema_extra["example"]

    response = client.put("/dataset/test_dataset/test_document", json=request)
    assert response.status_code == 404
    assert response.json() == {"detail": "Dataset with id [test_dataset] not found"}


def test_add_document_to_dataset_document_does_not_exist_already():
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
