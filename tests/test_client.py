import json
import logging
import threading
from pathlib import Path
from time import sleep

import pytest
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
CAPTURING_FOLDER = Path(__file__).parent.parent.resolve() / "captures"
CAPTURING = True


def start_capturing_session(client: GalahadClient, test_name: str):
    if not CAPTURING:
        return

    CAPTURING_FOLDER.mkdir(exist_ok=True, parents=True)
    p = CAPTURING_FOLDER / test_name

    if p.exists():
        p.unlink()

    def logging_hook(response, *args, **kwargs):
        request = response.request

        s = f"> {request.method} {request.url}"

        if request.body:
            body = json.loads(request.body)
            s += f"\n{json.dumps(body, indent=2)}"

        s += f"\n\n< {response.status_code}"
        if response.content:
            content = json.loads(response.content)
            s += f"\n{json.dumps(content, indent=2)}"

        with p.open("a", newline="\n") as f:
            f.write(s)
            f.write("\n\n")

    session = client.start_session()
    session.hooks["response"] = [logging_hook]


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
    start_capturing_session(client, "test_is_connected")

    assert client.is_connected()


def test_create_dataset(client: GalahadClient):
    start_capturing_session(client, "test_create_dataset")

    client.create_dataset("dataset1")
    assert len(client.list_datasets()) == 1


def test_create_dataset_which_already_exists(caplog, client: GalahadClient):
    start_capturing_session(client, "test_create_dataset_which_already_exists")

    client.create_dataset("dataset1")

    with caplog.at_level(logging.INFO):
        client.create_dataset("dataset1")
    assert len(caplog.messages) == 1
    assert caplog.messages[0] == 'Dataset with id "dataset1" already exists'
    assert len(client.list_datasets()) == 1


def test_list_datasets(client: GalahadClient):
    start_capturing_session(client, "test_list_datasets")

    client.create_dataset("dataset2")
    client.create_dataset("dataset1")

    # sorted by name
    assert ["dataset1", "dataset2"] == client.list_datasets()


def test_contains_dataset_if_dataset_exists(client: GalahadClient):
    start_capturing_session(client, "test_contains_dataset_if_dataset_exists")

    client.create_dataset("dataset1")
    assert client.contains_dataset("dataset1") is True


def test_contains_dataset_if_dataset_does_not_exist(client: GalahadClient):
    start_capturing_session(client, "test_contains_dataset_if_dataset_does_not_exist")

    assert client.contains_dataset("dataset1") is False


def test_dataset_naming_invalid(client: GalahadClient):
    start_capturing_session(client, "test_dataset_naming_invalid")

    with pytest.raises(ValueError):
        client.contains_dataset("-")


def test_delete_dataset(client: GalahadClient):
    start_capturing_session(client, "test_delete_dataset")

    client.create_dataset("dataset1")

    client.delete_dataset("dataset1")
    assert len(client.list_datasets()) == 0


def test_delete_dataset_which_does_not_exist(caplog, client: GalahadClient):
    start_capturing_session(client, "test_delete_dataset_which_does_not_exist")

    with caplog.at_level(logging.INFO):
        client.delete_dataset("dataset1")
    assert len(caplog.messages) == 1
    assert caplog.messages[0] == 'Dataset with id "dataset1" does not exist'


def test_delete_dataset_with_invalid_naming(client: GalahadClient):
    start_capturing_session(client, "test_delete_dataset_with_invalid_naming")

    with pytest.raises(ValueError):
        client.delete_dataset("-")


# TODO: No assert here -> how to check the creation of a document? Integrate test from test_server?
def test_create_document_in_dataset(client: GalahadClient):
    start_capturing_session(client, "test_create_document_in_dataset")

    doc = EXAMPLE_DOCUMENT

    client.create_dataset("dataset1")
    client.create_document_in_dataset("dataset1", "doc1", doc)


def test_create_document_in_dataset_if_dataset_does_not_exist(client: GalahadClient):
    start_capturing_session(client, "test_create_document_in_dataset_if_dataset_does_not_exist")

    doc = EXAMPLE_DOCUMENT

    with pytest.raises(ValueError):
        client.create_document_in_dataset("dataset1", "doc1", doc)

    client.create_document_in_dataset("dataset1", "doc1", doc, True)


@pytest.mark.parametrize("dataset_id, document_id", [("-", "doc1"), ("dataset1", "-")])
def test_create_document_in_dataset_naming(client: GalahadClient, dataset_id, document_id):
    start_capturing_session(client, f"test_create_document_in_dataset_naming_{dataset_id}_{document_id}")

    doc = EXAMPLE_DOCUMENT

    with pytest.raises(ValueError):
        client.create_document_in_dataset(dataset_id, document_id, doc)


def test_list_documents_in_dataset(client: GalahadClient):
    start_capturing_session(client, "test_list_documents_in_dataset")

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
    start_capturing_session(client, "test_list_documents_in_dataset_if_dataset_does_not_exist")

    with pytest.raises(HTTPError):
        client.list_documents_in_dataset("dataset1")


def test_list_documents_in_dataset_naming(client: GalahadClient):
    start_capturing_session(client, "test_list_documents_in_dataset_naming")

    with pytest.raises(ValueError):
        client.list_documents_in_dataset("-")


def test_dataset_contains_document(client: GalahadClient):
    start_capturing_session(client, "test_dataset_contains_document")

    doc = EXAMPLE_DOCUMENT
    client.create_dataset("dataset1")
    client.create_document_in_dataset("dataset1", "doc1", doc)

    assert client.dataset_contains_document("dataset1", "doc1") is True


def test_dataset_contains_document_if_document_does_not_exist(client: GalahadClient):
    start_capturing_session(client, "test_dataset_contains_document_if_document_does_not_exist")

    client.create_dataset("dataset1")
    assert client.dataset_contains_document("dataset1", "doc1") is False


def test_dataset_contains_document_if_dataset_does_not_exist(client: GalahadClient):
    start_capturing_session(client, "test_dataset_contains_document_if_dataset_does_not_exist")

    with pytest.raises(HTTPError):
        client.dataset_contains_document("dataset1", "doc1")


@pytest.mark.parametrize("dataset_id, document_id", [("-", "doc1"), ("dataset1", "-")])
def test_dataset_contains_document_naming(client: GalahadClient, dataset_id, document_id):
    start_capturing_session(client, f"test_dataset_contains_document_naming_{dataset_id}_{document_id}")

    with pytest.raises(ValueError):
        client.dataset_contains_document(dataset_id, document_id)


def test_delete_document_in_dataset(client: GalahadClient):
    start_capturing_session(client, "test_delete_document_in_dataset")

    doc = EXAMPLE_DOCUMENT

    client.create_dataset("dataset1")

    client.create_document_in_dataset("dataset1", "doc1", doc)

    assert len(client.list_documents_in_dataset("dataset1")) == 1

    client.delete_document_in_dataset("dataset1", "doc1")
    assert len(client.list_documents_in_dataset("dataset1")) == 0


def test_delete_document_in_dataset_if_document_does_not_exist(caplog, client: GalahadClient):
    start_capturing_session(client, "test_delete_document_in_dataset_if_document_does_not_exist")

    client.create_dataset("dataset1")
    with caplog.at_level(logging.INFO):
        client.delete_document_in_dataset("dataset1", "doc1")
    assert len(caplog.messages) == 1
    assert caplog.messages[0] == 'Document with id "doc1" does not exist in dataset with id "dataset1"'


def test_delete_document_in_dataset_if_dataset_does_not_exist(client: GalahadClient):
    start_capturing_session(client, "test_delete_document_in_dataset_if_dataset_does_not_exist")

    with pytest.raises(HTTPError):
        client.delete_document_in_dataset("dataset2", "doc1")


@pytest.mark.parametrize("dataset_id, document_id", [("-", "doc1"), ("dataset1", "-")])
def test_delete_document_in_dataset_with_invalid_naming(client: GalahadClient, dataset_id, document_id):
    start_capturing_session(client, "test_delete_document_in_dataset_with_invalid_naming")

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
    start_capturing_session(client, "test_get_classifier_info_if_classifier_does_not_exist")

    with pytest.raises(HTTPError):
        client.get_classifier_info("classifier4")


def test_get_classifier_info_naming(client: GalahadClient):
    start_capturing_session(client, "test_get_classifier_info_naming")

    with pytest.raises(ValueError):
        client.get_classifier_info("-")


# TODO: train for long time such that client.train_on_dataset("classifier1", "model1", "dataset1") is false
def test_train_on_dataset(client: GalahadClient):
    start_capturing_session(client, "test_train_on_dataset")

    doc = EXAMPLE_DOCUMENT
    client.create_document_in_dataset("dataset1", "document1", doc, True)
    assert client.train_on_dataset("classifier1", "model1", "dataset1")


def test_train_on_dataset_if_classifier_does_not_exist(client: GalahadClient):
    start_capturing_session(client, "test_train_on_dataset_if_classifier_does_not_exist")

    doc = EXAMPLE_DOCUMENT
    client.create_document_in_dataset("dataset1", "document1", doc, True)
    with pytest.raises(HTTPError):
        client.train_on_dataset("classifier4", "model1", "dataset1")


def test_train_on_dataset_if_dataset_does_not_exist(client: GalahadClient):
    start_capturing_session(client, "test_train_on_dataset_if_dataset_does_not_exist")

    with pytest.raises(HTTPError):
        client.train_on_dataset("classifier1", "model1", "dataset3")


@pytest.mark.parametrize("classifier_id, dataset_id", [("-", "doc1"), ("dataset1", "-")])
def test_train_on_dataset_naming(client: GalahadClient, classifier_id, dataset_id):
    start_capturing_session(client, f"test_train_on_dataset_naming_{classifier_id}_{dataset_id}")

    with pytest.raises(ValueError):
        client.train_on_dataset(classifier_id, "model1", dataset_id)


def test_predict_on_document(client: GalahadClient):
    start_capturing_session(client, "test_predict_on_document")

    doc = EXAMPLE_DOCUMENT

    client.create_dataset("dataset1")
    client.create_document_in_dataset("dataset1", "doc1", doc)
    client.train_on_dataset("classifier1", "model1", "dataset1")
    client.delete_dataset("dataset1")

    predicted_doc = client.predict_on_document("classifier1", "model1", doc)
    assert doc == predicted_doc


def test_predict_on_document_if_classifier_does_not_exist(client: GalahadClient):
    start_capturing_session(client, "test_predict_on_document_if_classifier_does_not_exist")

    doc = EXAMPLE_DOCUMENT
    with pytest.raises(HTTPError):
        client.predict_on_document("classifier4", "model1", doc)


def test_predict_on_document_if_model_does_not_exist(client: GalahadClient):
    start_capturing_session(client, "test_predict_on_document_if_model_does_not_exist")

    doc = EXAMPLE_DOCUMENT
    with pytest.raises(HTTPError):
        client.predict_on_document("classifier1", "model4", doc)


@pytest.mark.parametrize("classifier_id, model_id", [("-", "model1"), ("classifier1", "-")])
def test_predict_on_document_naming(client: GalahadClient, classifier_id, model_id):
    start_capturing_session(client, f"test_predict_on_document_naming_{classifier_id}_{model_id}")

    doc = EXAMPLE_DOCUMENT
    with pytest.raises(ValueError):
        client.predict_on_document(classifier_id, model_id, doc)
