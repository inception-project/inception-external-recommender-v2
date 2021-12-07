import tempfile
from pathlib import Path

import uvicorn

from galahad.server import GalahadServer
from galahad.server.contrib.ner.spacy_ner import SpacyNerClassifier
from galahad.server.contrib.sentence_classification.sklearn_sentence_classifier import SklearnSentenceClassifier

# data_dir = Path(tempfile.mkdtemp())
data_dir = Path("inception")


server = GalahadServer(data_dir=data_dir)
server.add_classifier("sklearn1", SklearnSentenceClassifier())
server.add_classifier("sklearn2", SklearnSentenceClassifier())
server.add_classifier("spacy_ner", SpacyNerClassifier("en_core_web_sm"))


if __name__ == "__main__":
    uvicorn.run(server)
