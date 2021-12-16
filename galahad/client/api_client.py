from typing import List, Dict, Optional
import requests
import logging


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

    def list_datasets(self) -> List[str]:
        response = requests.get(self.endpoint_url + "/dataset")
        if response.status_code != 200:
            raise StatusCodeError("Status Code should be 200 but is " + str(response.status_code))
        return response.json()["names"]

    def contains_dataset(self, dataset_id: str) -> bool:
        return dataset_id in self.list_datasets()

    def create_dataset(self, dataset_id: str, data=None):
        if data is None:
            data = {}
        response = requests.put(self.endpoint_url + "/dataset/" + dataset_id, data)
        if response.status_code == 409:
            logging.exception("Dataset with id: " + dataset_id + " already exists")
            return None
        if response.status_code != 204:
            raise StatusCodeError("Problem with dataset_id: " + dataset_id +
                                  ". Status Code should be 204 but is " + str(response.status_code))
        if response.text != "":
            raise ResponseError("Problem with dataset_id: " + dataset_id + ". JSON response was not empty")
        return None

    def delete_dataset(self, dataset_id: str):
        response = requests.delete(self.endpoint_url + "/dataset/" + dataset_id)
        if response.status_code == 404:
            logging.exception("Dataset with id: " + dataset_id + " does not exist")
            return None
        if response.status_code != 204:
            raise StatusCodeError("Problem with dataset_id: " + dataset_id +
                                  ". Status Code should be 204 but is " + str(response.status_code))
        if response.text != "":
            raise ResponseError("Problem with dataset_id: " + dataset_id + ". JSON response was not empty")
        return None

    def delete_datasets(self, dataset_ids: List[str]):
        for dataset_id in dataset_ids:
            self.delete_dataset(dataset_id)

    def delete_all_datasets(self):
        self.delete_datasets(self.list_datasets())

    def list_documents_in_dataset(self, dataset_id) -> Dict[str, int]:
        response = requests.get(self.endpoint_url + "/dataset/" + dataset_id)
        if response.status_code == 404:
            raise DataNonExistentError("Dataset with id: " + dataset_id + " does not exist")
        if response.status_code != 200:
            raise StatusCodeError("Problem with dataset_id: " + dataset_id +
                                  ". Status Code should be 200 but is " + str(response.status_code))

        return dict(zip(response.json()["names"], response.json()["versions"]))

    #TODO def create_document_in_dataset(self):
    #TODO /dataset/ besser loesen
