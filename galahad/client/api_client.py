import typing
import requests


class GalahadClient:

    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url.removesuffix("/")

    def is_connected(self) -> [bool, str]:
        response = requests.get(self.endpoint_url + "/ping")
        if response.status_code != 200:
            return [False, "Wrong status code"]
        if response.json() != {"ping": "pong"}:
            return [False, "Wrong response text"]
        return [True, "OK"]

    # def get_recommenders(self):
    #    return 0

    # def create_dataset(self, name: str):
    #    print(name)
