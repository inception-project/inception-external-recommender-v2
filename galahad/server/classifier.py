from enum import Enum
from typing import Dict, List, Optional

from galahad.server.dataclasses import ClassifierInfo, Document


class AnnotationTypes(Enum):
    TOKEN = "g.token"
    SENTENCE = "g.sentence"
    SENTENCE_LABEL = "g.sentence_label"


class Remapper:
    def __init__(self, remaps: Dict[str, str]):
        self._remaps = remaps

    def remap(self, name: str) -> str:
        return self._remaps.get(name, name)


class Classifier:
    def train(self, documents: List[Document], remapper: Remapper):
        raise NotImplementedError()

    def predict(self, remapper: Remapper):
        raise NotImplementedError()

    def consumes(self) -> List[str]:
        raise NotImplementedError()

    def produces(self) -> List[str]:
        raise NotImplementedError()


class ClassifierStore:
    def __init__(self):
        self._classifiers: Dict[str, Classifier] = {}

    def add_classifier(self, name: str, classifier: Classifier):
        if name in self._classifiers:
            raise ValueError(f"Model [{name}] already in classifier store!")

        self._classifiers[name] = classifier

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
