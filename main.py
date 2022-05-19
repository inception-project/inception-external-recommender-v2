import logging

import uvicorn

from galahad.server import GalahadServer
from galahad.server.contrib.ner.spacy_ner import SpacyNerTagger
from galahad.server.contrib.pos.spacy_pos import SpacyPosTagger
from galahad.server.contrib.sentence_classification.sklearn_sentence_classifier import SklearnSentenceClassifier

server = GalahadServer("my_data_folder")
server.add_classifier("SpacyPOS", SpacyPosTagger("en_core_web_sm"))
server.add_classifier("SpacyNER", SpacyNerTagger("en_core_web_sm"))
server.add_classifier("Sent", SklearnSentenceClassifier())


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.DEBUG)

    uvicorn.run(server, host="127.0.0.1", port=8000)
