from galahad.client.api_client import GalahadClient, NamingError, DataNonExistentError
from galahad.server import GalahadServer
from galahad.server.dataclasses import Document
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

    client.create_dataset("dataset1")
    assert len(caplog.messages) == 0
    assert len(client.list_datasets()) == 1

    client.create_dataset("dataset2")
    assert len(caplog.messages) == 0

    client.create_dataset("dataset2")
    assert len(caplog.messages) == 1 and caplog.messages[0] == "Dataset with id: dataset2 already exists"

    with pytest.raises(NamingError):
        client.create_dataset("-")


def test_list_datasets():
    client.delete_all_datasets()

    client.create_dataset("dataset2")
    client.create_dataset("dataset1")

    # sorted by name
    assert ["dataset1", "dataset2"] == client.list_datasets()
    assert client.contains_dataset("dataset1")
    assert client.contains_dataset("dataset2")


def test_delete_dataset(caplog):
    client.delete_all_datasets()

    client.create_dataset("dataset1")

    client.delete_dataset("dataset1")
    assert len(caplog.messages) == 0
    assert len(client.list_datasets()) == 0

    client.delete_dataset("dataset1")
    assert len(caplog.messages) == 1 and caplog.messages[0] == "Dataset with id: dataset1 does not exist"

    with pytest.raises(NamingError):
        client.delete_dataset("-")


def test_create_document_in_dataset():
    doc = Document.Config.schema_extra["example"]
    client.delete_all_datasets()

    with pytest.raises(DataNonExistentError):
        client.create_document_in_dataset("dataset1", "doc1", doc)

    client.create_document_in_dataset("dataset1", "doc1", doc, True)

    client.create_dataset("dataset2")
    client.create_document_in_dataset("dataset2", "doc1", doc)

    with pytest.raises(NamingError):
        client.create_document_in_dataset("-", "doc1", doc)

    with pytest.raises(NamingError):
        client.create_document_in_dataset("dataset1", "-", doc)


def test_list_documents_in_dataset():
    client.delete_all_datasets()
    doc = Document.Config.schema_extra["example"]

    client.create_dataset("dataset1")

    doc["version"] = 7
    client.create_document_in_dataset("dataset1", "doc3", doc)
    doc["version"] = 2
    client.create_document_in_dataset("dataset1", "doc1", doc)
    doc["version"] = 8
    client.create_document_in_dataset("dataset1", "doc2", doc)

    response = client.list_documents_in_dataset("dataset1")

    # sorted by doc id
    assert list(response.keys()) == ["doc1", "doc2", "doc3"]
    assert list(response.values()) == [2, 8, 7]

    with pytest.raises(DataNonExistentError):
        client.list_documents_in_dataset("dataset2")

    with pytest.raises(NamingError):
        client.list_documents_in_dataset("-")


def test_delete_document_in_dataset():
    client.delete_all_datasets()
    doc = Document.Config.schema_extra["example"]

    client.create_dataset("dataset1")

    client.create_document_in_dataset("dataset1", "doc1", doc)

    assert len(client.list_documents_in_dataset("dataset1")) == 1

    client.delete_document_in_dataset("dataset1", "doc1")
    assert len(client.list_documents_in_dataset("dataset1")) == 0

    client.delete_document_in_dataset("dataset1", "doc2")

    with pytest.raises(DataNonExistentError):
        client.delete_document_in_dataset("dataset2", "doc1")

    with pytest.raises(NamingError):
        client.delete_document_in_dataset("-", "doc1")

    with pytest.raises(NamingError):
        client.delete_document_in_dataset("dataset1", "-")
