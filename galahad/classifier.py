from typing import Dict, List, Optional

from galahad.dataclasses import ClassifierInfo


class Classifier:
    def train(self):
        raise NotImplementedError()

    def test(self):
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
