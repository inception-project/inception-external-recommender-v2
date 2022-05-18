import logging
from typing import Dict, List

import requests
from requests_toolbelt import sessions

from galahad.server import server
from galahad.server.dataclasses import ClassifierInfo, Document

logger = logging.getLogger("galahad.client")


class HTTPError(Exception):
    def __init__(self, response: requests.Response):
        error_msg = ""

        if isinstance(response.reason, bytes):
            # We attempt to decode utf-8 first because some servers
            # choose to localize their reason strings. If the string
            # isn't utf-8, we fall back to iso-8859-1 for all other
            # encodings. (See PR #3538)
            try:
                response_reason = response.reason
                reason = response_reason.decode("utf-8")
            except UnicodeDecodeError:
                reason = response.reason.decode("iso-8859-1")
        else:
            reason = response.reason

        status_code = response.status_code
        if 400 <= status_code < 500:
            error_msg = f"{status_code} Client Error: {reason} for url: {response.url}"

        elif 500 <= status_code < 600:
            error_msg = f"{status_code} Server Error: {reason} for url: {response.url}"

        body = response.content
        if len(body):
            error_msg += "\n"
            error_msg += body.decode("utf-8")

        super().__init__(error_msg)


def create_error_message(**kwargs) -> str:
    variables = kwargs
    if len(variables.keys()) == 0:
        return ""
    else:
        error_message = ["The problem appeared with the variables "]
        for variable_name in variables.keys():
            # variable_list.append(variable_name)
            # variable_list.append(variables[variable_name])
            error_message.append(f'"{variable_name}"')
            error_message.append(":")
            error_message.append(f'"{variables[variable_name]}"')
            error_message.append("and")
        del error_message[-1]

    return " ".join(error_message)


def check_response(response: requests.Response):
    try:
        response.raise_for_status()
    except requests.exceptions.HTTPError as e:
        raise HTTPError(response) from e


# TODO: would it be better for the performance if we checked the naming before sending the request to the server?
def check_naming_is_ok(given_status: int, **kwargs):
    if given_status == 422:
        raise ValueError(
            f"Naming for one of the variables is invalid. "
            f"Please look at the documentation for correct naming. {create_error_message(**kwargs)}"
        )


class GalahadClient:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url.rstrip("/")
        self._session = self._build_session()

    def start_session(self) -> requests.Session:
        self._session = self._build_session()
        return self._session

    def _build_session(self) -> requests.Session:
        session = sessions.BaseUrlSession(self.endpoint_url)
        return session

    def is_connected(self) -> bool:
        response = self._session.get("/ping")
        if response.status_code != 200:
            logger.info("StatusCodeError")
            return False
        if response.json() != {"ping": "pong"}:
            logger.info("ResponseError")
            return False
        return True

    # output is sorted by dataset name
    def list_datasets(self) -> List[str]:
        response = self._session.get("/dataset")
        check_response(response)
        return response.json()["names"]

    def contains_dataset(self, dataset_id: str) -> bool:
        server.check_naming_is_ok_regex(dataset_id)
        return dataset_id in self.list_datasets()

    def create_dataset(self, dataset_id: str):
        response = self._session.put(f"/dataset/{dataset_id}", {})
        check_naming_is_ok(response.status_code, dataset_id=dataset_id)
        if response.status_code == 409:
            logger.info(f'Dataset with id "{dataset_id}" already exists')
            return None

        check_response(response)

    def delete_dataset(self, dataset_id: str):
        response = self._session.delete(f"/dataset/{dataset_id}")
        check_naming_is_ok(response.status_code, dataset_id=dataset_id)
        if response.status_code == 404:
            logger.info(f'Dataset with id "{dataset_id}" does not exist')
            return None

        check_response(response)

    def delete_datasets(self, dataset_ids: List[str]):
        for dataset_id in dataset_ids:
            self.delete_dataset(dataset_id)

    def delete_all_datasets(self):
        self.delete_datasets(self.list_datasets())

    # The new document of the same name will override an existing one!
    def create_document_in_dataset(
        self, dataset_id: str, document_id: str, document: Document, auto_create_dataset=False
    ):
        response = self._session.put(f"/dataset/{dataset_id}/{document_id}", json=document.dict())
        check_naming_is_ok(response.status_code, dataset_id=dataset_id, document_id=document_id)

        if response.status_code == 404:
            if auto_create_dataset:
                self.create_dataset(dataset_id)
                response = self._session.put(f"/dataset/{dataset_id}/{document_id}", json=document.dict())
            else:
                raise ValueError(
                    f'The dataset for the given id: "{dataset_id}" does not exist. To create it, '
                    'set the optional parameter "auto_create_dataset" to True'
                )

        check_response(response)

    # result is sorted by doc id
    def list_documents_in_dataset(self, dataset_id) -> Dict[str, int]:
        response = self._session.get(f"/dataset/{dataset_id}")
        check_naming_is_ok(response.status_code, dataset_id=dataset_id)
        check_response(response)

        return dict(zip(response.json()["names"], response.json()["versions"]))

    def dataset_contains_document(self, dataset_id: str, document_id: str) -> bool:
        server.check_naming_is_ok_regex(document_id)
        return document_id in list(self.list_documents_in_dataset(dataset_id).keys())

    def delete_document_in_dataset(self, dataset_id: str, document_id: str):
        if self.dataset_contains_document(dataset_id, document_id):
            response = self._session.delete(f"/dataset/{dataset_id}/{document_id}")
            check_naming_is_ok(response.status_code, dataset_id=dataset_id, document_id=document_id)
            check_response(response)
        else:
            logger.info(f'Document with id "{document_id}" does not exist in dataset with id "{dataset_id}"')

    def delete_all_documents_in_dataset(self, dataset_id: str):
        for document_id in list(self.list_documents_in_dataset(dataset_id).keys()):
            self.delete_document_in_dataset(dataset_id, document_id)

    def delete_all_documents(self):
        for dataset_id in self.list_datasets():
            self.delete_all_documents_in_dataset(dataset_id)

    def list_all_classifiers(self) -> List[ClassifierInfo]:
        response = self._session.get("/classifier")
        check_response(response)

        info_list = []
        for classifier in response.json():
            info_list.append(ClassifierInfo.parse_obj(classifier))

        return info_list

    def get_classifier_info(self, classifier_id: str) -> ClassifierInfo:
        response = self._session.get(f"/classifier/{classifier_id}")
        check_naming_is_ok(response.status_code, classifier_id=classifier_id)
        check_response(response)

        return ClassifierInfo.parse_obj(response.json())

    # True: training has started. False: training has started already and function call had no effect
    def train_on_dataset(self, classifier_id: str, model_id: str, dataset_id: str) -> bool:
        response = self._session.post(f"/classifier/{classifier_id}/{model_id}/train/{dataset_id}")
        check_naming_is_ok(response.status_code, classifier_id=classifier_id, model_id=model_id, dataset_id=dataset_id)
        if response.status_code == 429:
            # logger.info("Training has already started! {create_error_message(variables)}")
            return False

        check_response(response)

        return True

    def predict_on_document(self, classifier_id: str, model_id: str, document: Document) -> Document:
        response = self._session.post(
            f"{self.endpoint_url}/classifier/{classifier_id}/{model_id}/predict", json=document.dict()
        )

        check_naming_is_ok(response.status_code, classifier_id=classifier_id, model_id=model_id)
        check_response(response)
        return response.json()
