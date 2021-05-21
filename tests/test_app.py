import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pytest
from fastapi.testclient import TestClient

import galahad.app as main
from galahad import config
from galahad.app import app
from galahad.model import DocumentAddRequest

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
    response = client.put("/dataset/test_data")
    assert response.status_code == 204
    assert response.text == ""
    assert (tmpdir / "test_data").is_dir()


def test_create_dataset_dataset_exist_already():
    response = client.put("/dataset/test_data")
    assert response.status_code == 204
    assert response.text == ""
    assert (tmpdir / "test_data").is_dir()

    response = client.put("/dataset/test_data")
    assert response.status_code == 409
    assert response.text == ""


# PUT add_document_to_dataset


def test_add_document_to_dataset_dataset_does_not_exist_already():
    request = DocumentAddRequest.Config.schema_extra["example"]

    response = client.put("/dataset/test_data/test_document", json=request)
    assert response.status_code == 404
    assert response.text == ""


def test_add_document_to_dataset_document_does_not_exist_already():
    client.put("/dataset/test_data")

    request = DocumentAddRequest.Config.schema_extra["example"]

    response = client.put("/dataset/test_data/test_document", json=request)
    assert response.status_code == 204
    assert response.text == ""

    p = tmpdir / "test_data" / "test_document"

    assert p.is_file()

    with p.open() as f:
        document = json.load(f)

    assert document == request
