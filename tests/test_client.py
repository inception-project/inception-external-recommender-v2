import logging
import threading
from time import sleep

import pytest
import requests
import uvicorn
from uvicorn import Config

from galahad.client import GalahadClient, HTTPError
from galahad.server import GalahadServer
from galahad.server.dataclasses import ClassifierInfo, Document
from tests.fixtures import TestClassifier

HOST = "127.0.0.1"
PORT = 8000
URL = f"http://{HOST}:{PORT}"

EXAMPLE_DOCUMENT = Document(**Document.Config.schema_extra["example"])


class UvicornTestServer(uvicorn.Server):
    def __init__(self, config: Config):
        super().__init__(config)
        self._thread = threading.Thread(target=self.run)

    def install_signal_handlers(self):
        pass

    def run_in_thread(self):
        self._thread.start()

    def wait_for_ready(self):
        while not self.started:
            sleep(2)

    def terminate(self):
        self.should_exit = True
        self._thread.join()


@pytest.fixture(scope="session")
def server():
    galahad = GalahadServer()
    classifier = TestClassifier()
    galahad.add_classifier("classifier1", classifier)
    galahad.add_classifier("classifier2", classifier)
    galahad.add_classifier("classifier3", classifier)

    config = Config(galahad, host=HOST, port=PORT, log_level="debug")
    server = UvicornTestServer(config)
    server.run_in_thread()
    # Wait for server to start
    server.wait_for_ready()

    yield server
    server.terminate()


@pytest.fixture
def client(server) -> GalahadClient:
    my_client = GalahadClient(URL)
    my_client.delete_all_datasets()
    return my_client


def test_client_creation(client: GalahadClient):
    assert client.endpoint_url == URL


def test_is_connected(client: GalahadClient):
    assert client.is_connected()


def test_create_dataset(client: GalahadClient):
    client.create_dataset("dataset1")
    assert len(client.list_datasets()) == 1


def test_create_dataset_which_already_exists(caplog, client: GalahadClient):
    client.create_dataset("dataset1")

    with caplog.at_level(logging.INFO):
        client.create_dataset("dataset1")
    assert len(caplog.messages) == 1
    assert caplog.messages[0] == 'Dataset with id "dataset1" already exists'
    assert len(client.list_datasets()) == 1


def test_list_datasets(client: GalahadClient):
    client.create_dataset("dataset2")
    client.create_dataset("dataset1")

    # sorted by name
    assert ["dataset1", "dataset2"] == client.list_datasets()


def contains_dataset_if_dataset_exists(client: GalahadClient):
    client.create_dataset("dataset1")
    assert client.contains_dataset("dataset1") is True


def contains_dataset_if_dataset_does_not_exist(client: GalahadClient):
    assert client.contains_dataset("dataset1") is False


def contains_dataset_naming(client: GalahadClient):
    with pytest.raises(ValueError):
        client.contains_dataset("-")


def test_delete_dataset(client: GalahadClient):
    client.create_dataset("dataset1")

    client.delete_dataset("dataset1")
    assert len(client.list_datasets()) == 0


def test_delete_dataset_which_does_not_exist(caplog, client: GalahadClient):
    with caplog.at_level(logging.INFO):
        client.delete_dataset("dataset1")
    assert len(caplog.messages) == 1
    assert caplog.messages[0] == 'Dataset with id "dataset1" does not exist'


def test_delete_dataset_naming(client: GalahadClient):
    with pytest.raises(ValueError):
        client.delete_dataset("-")


# TODO: No assert here -> how to check the creation of a document? Integrate test from test_server?
def test_create_document_in_dataset(client: GalahadClient):
    doc = EXAMPLE_DOCUMENT

    client.create_dataset("dataset1")
    client.create_document_in_dataset("dataset1", "doc1", doc)


def test_create_document_in_dataset_if_dataset_does_not_exist(client: GalahadClient):
    doc = EXAMPLE_DOCUMENT

    with pytest.raises(ValueError):
        client.create_document_in_dataset("dataset1", "doc1", doc)

    client.create_document_in_dataset("dataset1", "doc1", doc, True)


@pytest.mark.parametrize("dataset_id, document_id", [("-", "doc1"), ("dataset1", "-")])
def test_create_document_in_dataset_naming(client: GalahadClient, dataset_id, document_id):
    doc = EXAMPLE_DOCUMENT

    with pytest.raises(ValueError):
        client.create_document_in_dataset(dataset_id, document_id, doc)


def test_list_documents_in_dataset(client: GalahadClient):
    doc = Document.Config.schema_extra["example"]

    client.create_dataset("dataset1")

    doc["version"] = 7
    client.create_document_in_dataset("dataset1", "doc3", Document(**doc))
    doc["version"] = 2
    client.create_document_in_dataset("dataset1", "doc1", Document(**doc))
    doc["version"] = 8
    client.create_document_in_dataset("dataset1", "doc2", Document(**doc))

    response = client.list_documents_in_dataset("dataset1")

    # sorted by doc id
    assert list(response.keys()) == ["doc1", "doc2", "doc3"]
    assert list(response.values()) == [2, 8, 7]


def test_list_documents_in_dataset_if_dataset_does_not_exist(client: GalahadClient):
    with pytest.raises(HTTPError):
        client.list_documents_in_dataset("dataset1")


def test_list_documents_in_dataset_naming(client: GalahadClient):
    with pytest.raises(ValueError):
        client.list_documents_in_dataset("-")


def test_dataset_contains_document(client: GalahadClient):
    doc = EXAMPLE_DOCUMENT
    client.create_dataset("dataset1")
    client.create_document_in_dataset("dataset1", "doc1", doc)

    assert client.dataset_contains_document("dataset1", "doc1") is True


def test_dataset_contains_document_if_document_does_not_exist(client: GalahadClient):
    client.create_dataset("dataset1")
    assert client.dataset_contains_document("dataset1", "doc1") is False


def test_dataset_contains_document_if_dataset_does_not_exist(client: GalahadClient):
    with pytest.raises(HTTPError):
        client.dataset_contains_document("dataset1", "doc1")


@pytest.mark.parametrize("dataset_id, document_id", [("-", "doc1"), ("dataset1", "-")])
def test_dataset_contains_document_naming(client: GalahadClient, dataset_id, document_id):
    with pytest.raises(ValueError):
        client.dataset_contains_document(dataset_id, document_id)


def test_delete_document_in_dataset(client: GalahadClient):
    doc = EXAMPLE_DOCUMENT

    client.create_dataset("dataset1")

    client.create_document_in_dataset("dataset1", "doc1", doc)

    assert len(client.list_documents_in_dataset("dataset1")) == 1

    client.delete_document_in_dataset("dataset1", "doc1")
    assert len(client.list_documents_in_dataset("dataset1")) == 0


def test_delete_document_in_dataset_if_document_does_not_exist(caplog, client: GalahadClient):
    client.create_dataset("dataset1")
    with caplog.at_level(logging.INFO):
        client.delete_document_in_dataset("dataset1", "doc1")
    assert len(caplog.messages) == 1
    assert caplog.messages[0] == 'Document with id "doc1" does not exist in dataset with id "dataset1"'


def test_delete_document_in_dataset_if_dataset_does_not_exist(client: GalahadClient):
    with pytest.raises(HTTPError):
        client.delete_document_in_dataset("dataset2", "doc1")


@pytest.mark.parametrize("dataset_id, document_id", [("-", "doc1"), ("dataset1", "-")])
def test_delete_document_in_dataset_naming(client: GalahadClient, dataset_id, document_id):
    client.create_dataset("dataset1")
    with pytest.raises(ValueError):
        client.delete_document_in_dataset(dataset_id, document_id)


def test_list_all_classifiers(client: GalahadClient):
    expected_infos = [
        ClassifierInfo.parse_obj({"name": "classifier1"}),
        ClassifierInfo.parse_obj({"name": "classifier2"}),
        ClassifierInfo.parse_obj({"name": "classifier3"}),
    ]

    assert client.list_all_classifiers() == expected_infos


def test_get_classifier_info(client: GalahadClient):
    assert client.get_classifier_info("classifier1") == ClassifierInfo.parse_obj({"name": "classifier1"})


def test_get_classifier_info_if_classifier_does_not_exist(client: GalahadClient):
    with pytest.raises(HTTPError):
        client.get_classifier_info("classifier4")


def test_get_classifier_info_naming(client: GalahadClient):
    with pytest.raises(ValueError):
        client.get_classifier_info("-")


# TODO: train for long time such that client.train_on_dataset("classifier1", "model1", "dataset1") is false
def test_train_on_dataset(client: GalahadClient):
    doc = EXAMPLE_DOCUMENT
    client.create_document_in_dataset("dataset1", "document1", doc, True)
    assert client.train_on_dataset("classifier1", "model1", "dataset1")


def test_train_on_dataset_if_classifier_does_not_exist(client: GalahadClient):
    doc = EXAMPLE_DOCUMENT
    client.create_document_in_dataset("dataset1", "document1", doc, True)
    with pytest.raises(HTTPError):
        client.train_on_dataset("classifier4", "model1", "dataset1")


def test_train_on_dataset_if_dataset_does_not_exist(client: GalahadClient):
    with pytest.raises(HTTPError):
        client.train_on_dataset("classifier1", "model1", "dataset3")


@pytest.mark.parametrize("classifier_id, dataset_id", [("-", "doc1"), ("dataset1", "-")])
def test_train_on_dataset_naming(client: GalahadClient, classifier_id, dataset_id):
    with pytest.raises(ValueError):
        client.train_on_dataset(classifier_id, "model1", dataset_id)


def test_predict_on_document(client: GalahadClient):
    doc = EXAMPLE_DOCUMENT

    client.create_dataset("dataset1")
    client.create_document_in_dataset("dataset1", "doc1", doc)
    client.train_on_dataset("classifier1", "model1", "dataset1")
    client.delete_dataset("dataset1")

    predicted_doc = client.predict_on_document("classifier1", "model1", doc)
    assert doc == predicted_doc


def test_predict_on_document_if_classifier_does_not_exist(client: GalahadClient):
    doc = EXAMPLE_DOCUMENT
    with pytest.raises(HTTPError):
        client.predict_on_document("classifier4", "model1", doc)


def test_predict_on_document_if_model_does_not_exist(client: GalahadClient):
    doc = EXAMPLE_DOCUMENT
    with pytest.raises(HTTPError):
        client.predict_on_document("classifier1", "model4", doc)


@pytest.mark.parametrize("classifier_id, model_id", [("-", "model1"), ("classifier1", "-")])
def test_predict_on_document_naming(client: GalahadClient, classifier_id, model_id):
    doc = EXAMPLE_DOCUMENT
    with pytest.raises(ValueError):
        client.predict_on_document(classifier_id, model_id, doc)
