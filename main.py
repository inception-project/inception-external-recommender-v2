import uvicorn

from galahad.server import GalahadServer
from galahad.server.contrib.ner.spacy_ner import SpacyNerTagger
from galahad.server.contrib.pos.spacy_pos import SpacyPosTagger
from galahad.server.contrib.sentence_classification.sklearn_sentence_classifier import \
    SklearnSentenceClassifier

app = GalahadServer()
# app.add_classifier("SpacyNER", SpacyNerClassifier("en_core_web_sm"))
app.add_classifier("SpacyPOS", SpacyPosTagger("en_core_web_sm"))

ner_server = GalahadServer()
ner_server.add_classifier("SpacyNER", SpacyNerTagger("en_core_web_sm"))

pos_server = GalahadServer()
pos_server.add_classifier("SpacyPOS", SpacyPosTagger("en_core_web_sm"))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
