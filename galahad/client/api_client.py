from typing import List, Dict

import requests
from requests import Response
import logging
from galahad.server.dataclasses import Document


class ApiClientError(Exception):
    pass


# Wrong status code is returned
class StatusCodeError(ApiClientError):
    pass


# Response is not correct
class ResponseError(ApiClientError):
    pass


# Data does not exist but should
class DataNonExistentError(ApiClientError):
    pass


# Invalid name for file or folder
class NamingError(ApiClientError):
    pass


def create_error_message(variables: Dict[str, str]) -> str:
    if len(variables.keys()) == 0:
        error_message = ""
    else:
        error_message = "The problem appeared with the variables "
        for variable_name in variables.keys():
            error_message = error_message + variable_name + ": " + variables[variable_name] + " and "
        error_message = error_message.removesuffix(" and ")
        error_message = error_message + "."
    return error_message


def is_status_ok(expected_status: int, given_status: int, variables: Dict[str, str]):
    if expected_status != given_status:
        error_message = create_error_message(variables)
        raise StatusCodeError(
            "Expected HTTP status is " + str(expected_status) + ", but " +
            str(given_status) + " was received from server. " + error_message)


def is_response_ok(expected_response: str, given_response: str, variables: Dict[str, str]):
    if expected_response != given_response:
        error_message = create_error_message(variables)
        raise ResponseError("Expected JSON response is \"" + expected_response +
                            "\", but \"" + given_response + "\" was received from server. " + error_message)


def is_response_empty(response: Response, variables: Dict[str, str]):
    is_status_ok(204, response.status_code, variables)
    is_response_ok("", response.text, variables)


# TODO: would it be better for the performance if we checked the naming before sending the request to the server?
def is_naming_ok(given_status: int, variables: Dict[str, str]):
    if given_status == 422:
        error_message = create_error_message(variables)
        raise NamingError("Naming for one of the variables is invalid. "
                          "Please look at the documentation for correct naming. " + error_message)


class GalahadClient:

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url.removesuffix("/")

    def is_connected(self) -> bool:
        response = requests.get(self.endpoint_url + "/ping")
        if response.status_code != 200:
            logging.exception("StatusCodeError")
            return False
        if response.json() != {"ping": "pong"}:
            logging.exception("ResponseError")
            return False
        return True

    # output is sorted by dataset name
    def list_datasets(self) -> List[str]:
        response = requests.get(self.endpoint_url + "/dataset")
        is_status_ok(200, response.status_code, {})
        return response.json()["names"]

    def contains_dataset(self, dataset_id: str) -> bool:
        return dataset_id in self.list_datasets()

    def create_dataset(self, dataset_id: str):
        response = requests.put(self.endpoint_url + "/dataset/" + dataset_id, {})
        is_naming_ok(response.status_code, {"dataset_id": dataset_id})
        if response.status_code == 409:
            logging.exception("Dataset with id: " + dataset_id + " already exists")
            return None
        is_response_empty(response, {"dataset_id": dataset_id})

    def delete_dataset(self, dataset_id: str):
        response = requests.delete(self.endpoint_url + "/dataset/" + dataset_id)
        is_naming_ok(response.status_code, {"dataset_id": dataset_id})
        if response.status_code == 404:
            logging.exception("Dataset with id: " + dataset_id + " does not exist")
            return None
        is_response_empty(response, {"dataset_id": dataset_id})

    def delete_datasets(self, dataset_ids: List[str]):
        for dataset_id in dataset_ids:
            self.delete_dataset(dataset_id)

    def delete_all_datasets(self):
        self.delete_datasets(self.list_datasets())

    # The new document of the same name will override an existing one!
    def create_document_in_dataset(self, dataset_id: str, document_id: str, document: Document,
                                   auto_create_dataset=False):
        response = requests.put(self.endpoint_url + "/dataset/" + dataset_id + "/" + document_id, json=document)
        is_naming_ok(response.status_code, {"dataset_id": dataset_id, "document_id": document_id})
        if response.status_code == 404:
            if auto_create_dataset:
                self.create_dataset(dataset_id)
                response = requests.put(self.endpoint_url + "/dataset/" + dataset_id + "/" + document_id, json=document)
            else:
                raise DataNonExistentError("The dataset for the given id: " + dataset_id +
                                           " does not exist. To create it, "
                                           "set the optional parameter auto_create_dataset to True")

        is_response_empty(response, {"dataset_id": dataset_id, "document_id": document_id})

    # result is sorted by doc id
    def list_documents_in_dataset(self, dataset_id) -> Dict[str, int]:
        response = requests.get(self.endpoint_url + "/dataset/" + dataset_id)
        is_naming_ok(response.status_code, {"dataset_id": dataset_id})
        if response.status_code == 404:
            raise DataNonExistentError("Dataset with id: " + dataset_id + " does not exist")
        is_status_ok(200, response.status_code, {"dataset_id": dataset_id})

        return dict(zip(response.json()["names"], response.json()["versions"]))

    def dataset_contains_document(self, dataset_id: str, document_id: str) -> bool:
        return document_id in list(self.list_documents_in_dataset(dataset_id).keys())

    def delete_document_in_dataset(self, dataset_id: str, document_id: str):
        response = requests.delete(self.endpoint_url + "/dataset/" + dataset_id + "/" + document_id)
        is_naming_ok(response.status_code, {"dataset_id": dataset_id, "document_id": document_id})
        if response.status_code == 404:
            raise DataNonExistentError("Dataset with id: " + dataset_id + " does not exist")
        is_response_empty(response, {"dataset_id": dataset_id, "document_id": document_id})

    def delete_all_documents_in_dataset(self, dataset_id: str):
        for document_id in list(self.list_documents_in_dataset(dataset_id).keys()):
            self.delete_document_in_dataset(dataset_id, document_id)

    def delete_all_documents(self):
        for dataset_id in self.list_datasets():
            self.delete_all_documents_in_dataset(dataset_id)
