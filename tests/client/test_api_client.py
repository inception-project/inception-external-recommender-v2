import pytest

from galahad.client.api_client import GalahadClient
from galahad.server import GalahadServer
from galahad.server.dataclasses import Document
from galahad.server.util import DataNonExistentError, NamingError
from tests.fixtures import TestClassifier

server = GalahadServer()
classifier = TestClassifier()
server.add_classifier("classifier1", classifier)
server.add_classifier("classifier2", classifier)
server.add_classifier("classifier3", classifier)
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


def test_list_all_classifiers():
    # for initialisation of servers see top of document

    expected_infos = [{"name": "classifier1"}, {"name": "classifier2"}, {"name": "classifier3"}]

    assert client.list_all_classifiers() == expected_infos


def test_list_classifier():
    assert client.list_classifier("classifier1") == {"name": "classifier1"}
    with pytest.raises(DataNonExistentError):
        client.list_classifier("classifier4")

    with pytest.raises(NamingError):
        client.list_classifier("-")


def test_train_on_dataset():
    client.delete_all_datasets()
    request = Document.Config.schema_extra["example"]
    client.create_document_in_dataset("dataset1", "document1", request, True)
    assert client.train_on_dataset("classifier1", "model1", "dataset1")

    # TODO: train for long time such that client.train_on_dataset("classifier1", "model1", "dataset1") is false

    with pytest.raises(DataNonExistentError):
        client.train_on_dataset("classifier4", "model1", "dataset1")

    with pytest.raises(DataNonExistentError):
        client.train_on_dataset("classifier1", "model1", "dataset3")

    with pytest.raises(NamingError):
        client.train_on_dataset("-", "model1", "dataset3")

    with pytest.raises(NamingError):
        client.train_on_dataset("classifier1", "model1", "-")


def test_predict_on_document():
    client.delete_all_datasets()
    request = Document.Config.schema_extra["example"]
    client.create_document_in_dataset("dataset1", "document1", request, True)
    client.train_on_dataset("classifier1", "model1", "dataset1")
    predicted_doc = client.predict_on_document("classifier1", "model1", request)
    assert Document(**request) == predicted_doc

    with pytest.raises(DataNonExistentError):
        client.predict_on_document("classifier4", "model1", request)

    with pytest.raises(DataNonExistentError):
        client.predict_on_document("classifier1", "model4", request)

    with pytest.raises(NamingError):
        client.predict_on_document("-", "model1", request)

    with pytest.raises(NamingError):
        client.predict_on_document("classifier1", "-", request)
