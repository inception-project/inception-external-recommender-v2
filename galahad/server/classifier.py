import logging
import os
from enum import Enum
from multiprocessing import Condition, Value
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
from filelock import FileLock

from galahad.server.dataclasses import ClassifierInfo, Document

logger = logging.getLogger(__file__)


class AnnotationTypes(Enum):
    TOKEN = "t.token"
    SENTENCE = "t.sentence"
    ANNOTATION = "t.annotation"


class AnnotationFeatures(Enum):
    VALUE = "f.value"


class Remapper:
    def __init__(self, remaps: Dict[str, str] = None):
        if remaps is None:
            remaps = {}

        self._remaps = remaps

    def remap(self, name: str) -> str:
        return self._remaps.get(name, name)


class Classifier:
    def __init__(self):
        self._model_directory: Optional[Path] = None

    def train(self, model_id: str, documents: List[Document]):
        pass

    def predict(self, model_id: str, document: Document) -> Optional[Document]:
        raise NotImplementedError()

    def consumes(self) -> List[str]:
        return []

    def produces(self) -> List[str]:
        return []

    def _save_model(self, model_id: str, model: Any):
        assert self._model_directory is not None, "You need to set `self._model_directory` first before saving!"

        model_path = self._get_model_path(model_id)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_model_path = model_path.with_suffix(".joblib.tmp")
        joblib.dump(model, tmp_model_path)

        os.replace(tmp_model_path, model_path)

    def _load_model(self, model_id: str) -> Optional[Any]:
        model_path = self._get_model_path(model_id)
        if model_path.is_file():
            logger.debug("Model found for [%s]", model_path)
            return joblib.load(model_path)
        else:
            logger.debug("No model found for [%s]", model_path)
            return None

    def _get_model_path(self, model_id: str) -> Path:
        return self._model_directory / self.name / f"model_{model_id}.joblib"

    @property
    def name(self) -> str:
        return type(self).__name__


class ClassifierStore:
    def __init__(self, model_directory: Path):
        self._model_directory = model_directory
        self._classifiers: Dict[str, Classifier] = {}

    def add_classifier(self, name: str, classifier: Classifier):
        if name in self._classifiers:
            raise ValueError(f"Model [{name}] already in classifier store!")

        classifier._model_directory = self._model_directory
        self._classifiers[name] = classifier

    def get_classifier(self, name: str) -> Optional[Classifier]:
        return self._classifiers.get(name)

    def get_classifier_info(self, name: str) -> Optional[ClassifierInfo]:
        """Builds classifier info for the classifier given by `name` and returns it.

        Args:
            name: The name of the classifier whose info to get.

        Returns:
            The classifier info of the classifier named `name` if it was found, else `None`.
        """
        classifier = self._classifiers.get(name)
        if not classifier:
            return None

        return ClassifierInfo(name=name)

    def get_classifier_infos(self) -> List[ClassifierInfo]:
        """Builds classifier infos for all classifiers in this store and returns it.

        Returns:
            List of classifier infos for all stored classifiers.
        """
        return [self.get_classifier_info(name) for name in sorted(self._classifiers.keys())]


def train_classifier(classifier: Classifier, dataset_folder: Path, model_id: str, lock_directory: Path):
    lock = get_lock(lock_directory, model_id)

    try:
        lock.acquire()

        documents = [Document.parse_file(p) for p in sorted(dataset_folder.iterdir())]
        classifier.train(model_id, documents)
    except TimeoutError:
        logger.info("Already training [%s] with model id [%s], skipping!", classifier.name, model_id)
    finally:
        lock.release()


def get_lock(lock_directory: Path, lock_id: str) -> FileLock:
    lock_directory.mkdir(parents=True, exist_ok=True)
    lock_path = lock_directory / f"{lock_id}.lock"
    return FileLock(str(lock_path), timeout=1)
