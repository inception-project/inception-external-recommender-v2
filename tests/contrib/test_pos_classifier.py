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

    annotations["t.annotation"] = [
        {"begin": 0, "end": 1, "features": {"f.value": "PRP"}},
        {"begin": 2, "end": 4, "features": {"f.value": "VBP"}},
        {"begin": 5, "end": 12, "features": {"f.value": "JJ"}},
        {"begin": 12, "end": 13, "features": {"f.value": "."}},
        {"begin": 14, "end": 19, "features": {"f.value": "NNP"}},
        {"begin": 20, "end": 28, "features": {"f.value": "VBD"}},
        {"begin": 29, "end": 33, "features": {"f.value": "PDT"}},
        {"begin": 34, "end": 35, "features": {"f.value": "DT"}},
        {"begin": 36, "end": 47, "features": {"f.value": "RB"}},
        {"begin": 48, "end": 55, "features": {"f.value": "VBN"}},
        {"begin": 56, "end": 60, "features": {"f.value": "NN"}},
        {"begin": 60, "end": 61, "features": {"f.value": "."}},
    ]

    assert predicted_doc == Document(**{"text": text, "version": 0, "annotations": annotations})
