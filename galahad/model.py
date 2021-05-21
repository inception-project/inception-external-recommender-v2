from typing import Any, Dict, List, Optional

from galahad.dataclasses import ModelInfo


class Model:
    def train(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()


class ModelStore:
    def __init__(self):
        self._models: Dict[str, Model] = {}

    def add_model(self, name: str, model: Model):
        if name in self._models:
            raise ValueError(f"Model [{name}] already in model store!")

        self._models[name] = model

    def get_model_info(self, name: str) -> ModelInfo:
        """Builds model info for the model given by `name` and returns it.

        Args:
            name: The name of the model whose info to get.

        Returns:
            The model info of the model named `name`.
        """

    def get_model_infos(self) -> List[ModelInfo]:
        """Builds model infos for all models in this store and returns it.

        Returns:
            List of model infos for all stored models.
        """
        return [self.get_model_info(name) for name in sorted(self._models.keys())]
