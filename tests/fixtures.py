from typing import List, Optional

from galahad.server.classifier import Classifier
from galahad.server.dataclasses import Document


class TestClassifier(Classifier):
    def train(self, model_id: str, documents: List[Document]):
        self._save_model(model_id, [d.json() for d in documents])

    def predict(self, model_id: str, document: Document) -> Optional[Document]:
        model = self._load_model(model_id)
        return document if model else None
