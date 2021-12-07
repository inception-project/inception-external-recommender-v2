from galahad.server import GalahadServer
from galahad.server.contrib.ner.spacy_ner import SpacyNerClassifier
from galahad.server.contrib.sentence_classification.sklearn_sentence_classifier import SklearnSentenceClassifier
import uvicorn

app = GalahadServer()
app.add_classifier("spacy_ner", SpacyNerClassifier("en_core_web_sm"))

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
