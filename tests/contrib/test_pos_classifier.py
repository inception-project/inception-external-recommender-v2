from pathlib import Path

from datasets import load_dataset

from galahad.client.formats import build_span_classification_request
from galahad.server.annotations import Annotations
from galahad.server.classifier import AnnotationTypes
from galahad.server.contrib.pos.spacy_pos import SpacyPosClassifier
from galahad.server.dataclasses import Document


def test_spacy_pos():
    clf = SpacyPosClassifier("en_core_web_sm")

    text = "I am jealous. Peter received such a beautifully crafted gift."

    annotations = {
        "t.token": [
            {"begin": 0, "end": 1},
            {"begin": 2, "end": 4},
            {"begin": 5, "end": 12},
            {"begin": 12, "end": 13},
            {"begin": 14, "end": 19},
            {"begin": 20, "end": 28},
            {"begin": 29, "end": 33},
            {"begin": 34, "end": 35},
            {"begin": 36, "end": 47},
            {"begin": 48, "end": 55},
            {"begin": 56, "end": 60},
            {"begin": 60, "end": 61},
        ],
        "t.sentence": [
            {"begin": 0, "end": 13},
            {"begin": 14, "end": 62},
        ],
    }

    doc = Document(**{"text": text, "version": 0, "annotations": annotations})
    predicted_doc = clf.predict("spacy", doc)

    assert len(predicted_doc.annotations["t.annotation"]) == len(annotations["t.token"])


def test_spacy_pos_predict(tmpdir):
    model_directory = Path(tmpdir)

    dataset = load_dataset("conll2003", split="validation")

    classifier = SpacyPosClassifier("en_core_web_sm")
    classifier._model_directory = model_directory
    predict_request = build_span_classification_request(dataset["tokens"])
    response = classifier.predict("spacy", predict_request)

    predicted_annotations = Annotations.from_dict(response.text, response.annotations)
    predictions = predicted_annotations.select(AnnotationTypes.ANNOTATION.value)
    predicted_labels = [p.features[classifier._target_feature] for p in predictions]
    assert len(predicted_labels) == sum(len(sentence) for sentence in dataset["tokens"])
