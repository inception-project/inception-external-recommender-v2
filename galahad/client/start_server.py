from galahad.client import api_client
from galahad.server import GalahadServer
from galahad.server.contrib.ner.spacy_ner import SpacyNerClassifier

server = GalahadServer()
spacy_ner = SpacyNerClassifier("en_core_web_sm")
server.add_classifier("SpacyNER", spacy_ner)
