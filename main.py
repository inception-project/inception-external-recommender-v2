from galahad.server import GalahadServer
from galahad.server.contrib.ner.spacy_ner import SpacyNerClassifier
from galahad.server.contrib.sentence_classification.sklearn_sentence_classifier import \
    SklearnSentenceClassifier

app = GalahadServer()
app.add_classifier("sklearn1", SklearnSentenceClassifier())
app.add_classifier("sklearn2", SklearnSentenceClassifier())
app.add_classifier("spacy_ner", SpacyNerClassifier("en_core_web_sm")
)
