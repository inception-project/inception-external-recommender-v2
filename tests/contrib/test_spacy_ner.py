from pathlib import Path

from datasets import load_dataset

from galahad.formats import build_span_classification_request
from galahad.server.annotations import Annotations
from galahad.server.classifier import AnnotationTypes
from galahad.server.contrib.ner.spacy_ner import SpacyNerTagger


def test_spacy_ner_predict(tmpdir):
    model_directory = Path(tmpdir)

    dataset = load_dataset("conll2003", split="validation")

    classifier = SpacyNerTagger("en_core_web_sm")
    classifier._model_directory = model_directory
    predict_request = build_span_classification_request(dataset["tokens"])
    response = classifier.predict("spacy", predict_request)

    predicted_annotations = Annotations.from_dict(response.text, response.annotations)
    predictions = predicted_annotations.select(AnnotationTypes.ANNOTATION.value)
    predicted_labels = [p.features[classifier._target_feature] for p in predictions]

    assert len(predicted_labels) > 0
