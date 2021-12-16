from galahad.client.api_client import GalahadClient
from galahad.server import GalahadServer
from galahad.server.dataclasses import Document, DocumentList
import logging
import pytest

server = GalahadServer()
host = "127.0.0.1"
port = 8000
global address
address = "http://" + host + ":" + str(port)
client = GalahadClient(address)


def test_creation():
    assert client.endpoint_url == address


def test_is_connected():
    assert client.is_connected()


def test_create_dataset(caplog):
    client.delete_all_datasets()

    client.create_dataset("dataset1", {"docA": "lorem ipsum"})
    assert len(caplog.messages) == 0

    client.create_dataset("dataset2", {"docB": "foo bar"})
    assert len(caplog.messages) == 0

    client.create_dataset("dataset2", {"docB": "foo bar"})
    assert len(caplog.messages) == 1 and caplog.messages[0] == "Dataset with id: dataset2 already exists"


def test_list_datasets():
    client.delete_all_datasets()

    client.create_dataset("dataset1", {"docA": "lorem ipsum"})
    client.create_dataset("dataset2", {"docB": "foo bar"})

    assert ["dataset1", "dataset2"] == client.list_datasets()
    assert client.contains_dataset("dataset1")
    assert client.contains_dataset("dataset2")


def test_delete_dataset(caplog):
    client.delete_all_datasets()

    client.create_dataset("dataset1", {"docA": "lorem ipsum"})

    client.delete_dataset("dataset1")
    assert len(caplog.messages) == 0

    client.delete_dataset("dataset1")
    assert len(caplog.messages) == 1 and caplog.messages[0] == "Dataset with id: dataset1 does not exist"


def test_list_documents_in_dataset():
    client.delete_all_datasets()

    client.create_dataset("test_dataset")
    expected_names = []
    expected_versions = []

    for i in range(3):
        name = f"test_document{i}"
        version = i

        expected_names.append(name)
        expected_versions.append(i)

        request = Document.Config.schema_extra["example"]
        request["version"] = version

        response = client.create_document(f"test_dataset/{name}")

    response = client.list_documents_in_dataset("test_dataset")
    print("asdf")


    assert client.list_documents_in_dataset("dataset1") == input_docs
