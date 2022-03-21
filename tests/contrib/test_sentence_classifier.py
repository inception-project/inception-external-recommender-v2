from pathlib import Path
from statistics import mean

from datasets import load_dataset

from galahad.formats import build_sentence_classification_document
from galahad.server.annotations import Annotations
from galahad.server.contrib.sentence_classification.sklearn_sentence_classifier import (
    SklearnSentenceClassifier,
)


def test_sklearn_sentence_classifier_train_predict(tmpdir):
    model_directory = Path(tmpdir)
    model_id = "my_test_model"

    train = load_dataset("sms_spam", split="train[:80%]")
    test = load_dataset("sms_spam", split="train[80%:]")

    train_texts, test_texts = train["sms"], test["sms"]
    train_labels, test_labels = train["label"], test["label"]

    classifier = SklearnSentenceClassifier()
    classifier._model_directory = model_directory

    # Train
    train_request = build_sentence_classification_document(train_texts, train_labels)
    classifier.train(model_id, [train_request])

    model_path = classifier._get_model_path(model_id)
    assert model_path.is_file(), f"Expected {model_path} to be an existing file!"

    # Predict
    predict_request = build_sentence_classification_document(test_texts, test_labels)
    prediction_result = classifier.predict(model_id, predict_request)

    predicted_annotations = Annotations.from_dict(prediction_result.text, prediction_result.annotations)

    predictions = predicted_annotations.select(classifier._sentence_annotation_type)
    predicted_labels = [p.features[classifier._target_feature] for p in predictions]

    assert len(predicted_labels) == len(test_labels)
    assert mean(int(e1 == e2) for e1, e2 in zip(predicted_labels, test_labels)) > 0.9
