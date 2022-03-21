import logging
from typing import Dict, List

import requests
from requests import Response

from galahad.server import server
from galahad.server.dataclasses import ClassifierInfo, Document
from galahad.server.util import (
    DataNonExistentError,
    NamingError,
    ResponseError,
    StatusCodeError,
)

logger = logging.getLogger("galahad-client")


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


def check_data_is_there(given_status: int, **kwargs):
    if given_status == 404:
        raise DataNonExistentError(f"No data was found in this path. {create_error_message(**kwargs)}")


def check_status_is_ok(expected_status: int, given_status: int, **kwargs):
    if expected_status != given_status:
        raise StatusCodeError(
            f"Expected HTTP status is {expected_status}, but {given_status} "
            f"was received from server. {create_error_message(**kwargs)}"
        )


def check_response_is_ok(expected_response: str, given_response: str, **kwargs):
    if expected_response != given_response:
        raise ResponseError(
            f'Expected JSON response is "{expected_response}", but "{given_response}" '
            f"was received from server. {create_error_message(**kwargs)}"
        )


def check_response_is_empty(response: Response, **kwargs):
    check_status_is_ok(204, response.status_code, **kwargs)
    check_response_is_ok("", response.text, **kwargs)


# TODO: would it be better for the performance if we checked the naming before sending the request to the server?
def check_naming_is_ok(given_status: int, **kwargs):
    if given_status == 422:
        raise NamingError(
            f"Naming for one of the variables is invalid. "
            f"Please look at the documentation for correct naming. {create_error_message(**kwargs)}"
        )


class GalahadClient:
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url.rstrip("/")

    def is_connected(self) -> bool:
        response = requests.get(f"{self.endpoint_url}/ping")
        if response.status_code != 200:
            logger.info("StatusCodeError")
            return False
        if response.json() != {"ping": "pong"}:
            logger.info("ResponseError")
            return False
        return True

    # output is sorted by dataset name
    def list_datasets(self) -> List[str]:
        response = requests.get(f"{self.endpoint_url}/dataset")
        check_status_is_ok(200, response.status_code)
        return response.json()["names"]

    def contains_dataset(self, dataset_id: str) -> bool:
        server.check_naming_is_ok_regex(dataset_id)
        return dataset_id in self.list_datasets()

    def create_dataset(self, dataset_id: str):
        response = requests.put(f"{self.endpoint_url}/dataset/{dataset_id}", {})
        check_naming_is_ok(response.status_code, dataset_id=dataset_id)
        if response.status_code == 409:
            logger.info(f'Dataset with id "{dataset_id}" already exists')
            return None
        check_response_is_empty(response, dataset_id=dataset_id)

    def delete_dataset(self, dataset_id: str):
        response = requests.delete(f"{self.endpoint_url}/dataset/{dataset_id}")
        check_naming_is_ok(response.status_code, dataset_id=dataset_id)
        if response.status_code == 404:
            logger.info(f'Dataset with id "{dataset_id}" does not exist')
            return None
        check_response_is_empty(response, dataset_id=dataset_id)

    def delete_datasets(self, dataset_ids: List[str]):
        for dataset_id in dataset_ids:
            self.delete_dataset(dataset_id)

    def delete_all_datasets(self):
        self.delete_datasets(self.list_datasets())

    # The new document of the same name will override an existing one!
    def create_document_in_dataset(
        self, dataset_id: str, document_id: str, document: Document, auto_create_dataset=False
    ):
        response = requests.put(f"{self.endpoint_url}/dataset/{dataset_id}/{document_id}", json=document.dict())
        check_naming_is_ok(response.status_code, dataset_id=dataset_id, document_id=document_id)
        if response.status_code == 404:
            if auto_create_dataset:
                self.create_dataset(dataset_id)
                response = requests.put(f"{self.endpoint_url}/dataset/{dataset_id}/{document_id}", json=document.dict())
            else:
                raise DataNonExistentError(
                    f'The dataset for the given id: "{dataset_id}" does not exist. To create it, '
                    'set the optional parameter "auto_create_dataset" to True'
                )

        check_response_is_empty(response, dataset_id=dataset_id, document_id=document_id)

    # result is sorted by doc id
    def list_documents_in_dataset(self, dataset_id) -> Dict[str, int]:
        response = requests.get(f"{self.endpoint_url}/dataset/{dataset_id}")
        check_naming_is_ok(response.status_code, dataset_id=dataset_id)
        check_data_is_there(response.status_code, dataset_id=dataset_id)
        check_status_is_ok(200, response.status_code, dataset_id=dataset_id)

        return dict(zip(response.json()["names"], response.json()["versions"]))

    def dataset_contains_document(self, dataset_id: str, document_id: str) -> bool:
        server.check_naming_is_ok_regex(document_id)
        return document_id in list(self.list_documents_in_dataset(dataset_id).keys())

    def delete_document_in_dataset(self, dataset_id: str, document_id: str):
        if self.dataset_contains_document(dataset_id, document_id):
            response = requests.delete(f"{self.endpoint_url}/dataset/{dataset_id}/{document_id}")
            check_naming_is_ok(response.status_code, dataset_id=dataset_id, document_id=document_id)
            check_data_is_there(response.status_code, dataset_id=dataset_id, document_id=document_id)
            check_response_is_empty(response, dataset_id=dataset_id, document_id=document_id)
        else:
            logger.info(f'Document with id "{document_id}" does not exist in dataset with id "{dataset_id}"')

    def delete_all_documents_in_dataset(self, dataset_id: str):
        for document_id in list(self.list_documents_in_dataset(dataset_id).keys()):
            self.delete_document_in_dataset(dataset_id, document_id)

    def delete_all_documents(self):
        for dataset_id in self.list_datasets():
            self.delete_all_documents_in_dataset(dataset_id)

    def list_all_classifiers(self) -> List[ClassifierInfo]:
        response = requests.get(f"{self.endpoint_url}/classifier")
        check_status_is_ok(200, response.status_code)
        info_list = []
        for classifier in response.json():
            info_list.append(ClassifierInfo.parse_obj(classifier))

        return info_list

    def get_classifier_info(self, classifier_id: str) -> ClassifierInfo:
        response = requests.get(f"{self.endpoint_url}/classifier/{classifier_id}")
        check_naming_is_ok(response.status_code, classifier_id=classifier_id)
        check_data_is_there(response.status_code, classifier_id=classifier_id)
        check_status_is_ok(200, response.status_code, classifier_id=classifier_id)
        return ClassifierInfo.parse_obj(response.json())

    # True: training has started. False: training has started already and function call had no effect
    def train_on_dataset(self, classifier_id: str, model_id: str, dataset_id: str) -> bool:
        response = requests.post(f"{self.endpoint_url}/classifier/{classifier_id}/{model_id}/train/{dataset_id}")
        check_naming_is_ok(response.status_code, classifier_id=classifier_id, model_id=model_id, dataset_id=dataset_id)
        check_data_is_there(response.status_code, classifier_id=classifier_id, model_id=model_id, dataset_id=dataset_id)
        if response.status_code == 429:
            # logger.info("Training has already started! {create_error_message(variables)}")
            return False

        check_status_is_ok(
            202, response.status_code, classifier_id=classifier_id, model_id=model_id, dataset_id=dataset_id
        )
        check_response_is_ok("", response.text, classifier_id=classifier_id, model_id=model_id, dataset_id=dataset_id)
        return True

    def predict_on_document(self, classifier_id: str, model_id: str, document: Document) -> Document:
        response = requests.post(
            f"{self.endpoint_url}/classifier/{classifier_id}/{model_id}/predict", json=document.dict()
        )
        check_naming_is_ok(response.status_code, classifier_id=classifier_id, model_id=model_id)
        check_data_is_there(response.status_code, classifier_id=classifier_id, model_id=model_id)
        check_status_is_ok(200, response.status_code, classifier_id=classifier_id, model_id=model_id)
        return response.json()
