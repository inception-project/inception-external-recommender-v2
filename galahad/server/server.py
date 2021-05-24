from pathlib import Path

from fastapi import FastAPI

from galahad.server.classifier import Classifier, ClassifierStore
from galahad.server.routes import register_routes


class GalahadServer(FastAPI):
    def __init__(self, title: str = "Galahad Server", data_dir: Path = None) -> None:
        super().__init__(title=title)

        if data_dir is None:
            data_dir = Path(__file__).resolve().parents[1]

        self._data_dir = data_dir
        self._classifier_store = ClassifierStore(data_dir / "models")

        self.state.data_dir = self._data_dir
        self.state.classifier_store = self._classifier_store

        register_routes(self)

    def add_classifier(self, name: str, classifier: Classifier):
        self._classifier_store.add_classifier(name, classifier)
