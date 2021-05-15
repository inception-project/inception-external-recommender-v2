from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pytest
from fastapi.testclient import TestClient

import galahad.app as main
from galahad import config
from galahad.app import app

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


def test_create_dataset_dataset_does_not_exist_already():
    response = client.put("/dataset/test_data")
    assert response.status_code == 204
    assert response.text == ""
    assert tmpdir.is_dir()


def test_create_dataset_dataset_exist_already():
    response = client.put("/dataset/test_data")
    assert response.status_code == 204
    assert response.text == ""
    assert tmpdir.is_dir()

    response = client.put("/dataset/test_data")
    assert response.status_code == 409
    assert response.text == ""
