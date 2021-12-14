from galahad.client.formats import (Span,
                                    build_sentence_classification_document,
                                    build_span_classification_request,
                                    build_span_classification_response)
from galahad.server.annotations import Annotations
from galahad.server.classifier import AnnotationFeatures, AnnotationTypes
from galahad.server.dataclasses import Document


def test_build_sentence_classification_request():
    sentences = ["John likes ice cream", "John hates chocolate."]
    labels = ["positive", "negative"]

    result = build_sentence_classification_document(sentences, labels)

    # Check sentences
    actual_sentences = result.annotations[AnnotationTypes.SENTENCE.value]
    assert len(sentences) == 2
    first_sentence, second_sentence = actual_sentences

    assert first_sentence.begin == 0
    assert first_sentence.end == 20
    assert first_sentence.features == {}

    assert second_sentence.begin == 21
    assert second_sentence.end == 42
    assert second_sentence.end == 42
    assert first_sentence.features == {}

    # Check annotations
    actual_sentence_annotations = result.annotations[AnnotationTypes.ANNOTATION.value]
    assert len(sentences) == 2
    first_annotation, second_annotation = actual_sentence_annotations

    assert first_annotation.begin == 0
    assert first_annotation.end == 20
    assert first_annotation.features == {AnnotationFeatures.VALUE.value: "positive"}

    assert second_annotation.begin == 21
    assert second_annotation.end == 42
    assert second_annotation.features == {AnnotationFeatures.VALUE.value: "negative"}


def test_span_sentence_classification_request():
    tokens = [
        ["John", "studied", "in", "the", "United", "States", "of", "America", "."],
        ["He", "works", "at", "ACME", "Company", "."],
    ]
    spans = [[Span(0, 1, "PER"), Span(4, 8, "LOC")], [Span(3, 5, "ORG")]]

    result = build_span_classification_request(tokens, spans)
    value_feature = AnnotationFeatures.VALUE.value
    actual_span_annotations = result.annotations[AnnotationTypes.ANNOTATION.value]

    first_ner = actual_span_annotations[0]
    assert first_ner.features[value_feature] == "PER"
    assert first_ner.begin == 0
    assert first_ner.end == 4

    second_ner = actual_span_annotations[1]
    assert second_ner.features[value_feature] == "LOC"
    assert second_ner.begin == 20
    assert second_ner.end == 44

    third_ner = actual_span_annotations[2]
    assert third_ner.features[value_feature] == "ORG"
    assert third_ner.begin == 59
    assert third_ner.end == 71


def test_span_sentence_classification_response():
    # tokens = [
    #    ["Ohio", ",", "the", "Boston", "."],
    #    ["Chanel", "No", ".", "5", "."],
    # ]
    spans = [[Span(0, 1, "STATE"), Span(3, 4, "CAP")], [Span(0, 4, "TM")]]

    raw_document = {
        "text": "Ohio, the Boston. Chanel No. 5.",
        "version": 0,
        "annotations": {
            "t.token": [
                {"begin": 0, "end": 4},
                {"begin": 4, "end": 5},
                {"begin": 6, "end": 9},
                {"begin": 10, "end": 16},
                {"begin": 16, "end": 17},
                {"begin": 18, "end": 24},
                {"begin": 25, "end": 27},
                {"begin": 27, "end": 28},
                {"begin": 28, "end": 29},
                {"begin": 29, "end": 30},
            ],
            "t.sentence": [{"begin": 0, "end": 17}, {"begin": 18, "end": 30}],
        }
    }

    document = Document.parse_obj(raw_document)

    result = build_span_classification_response(document, spans)
    actual_annotations = Annotations.from_document(result)
    actual_span_annotations = actual_annotations.select(AnnotationTypes.ANNOTATION.value)
    value_feature = AnnotationFeatures.VALUE.value

    assert result.text == raw_document["text"]

    first_ner = actual_span_annotations[0]
    assert first_ner.features[value_feature] == "STATE"
    assert first_ner.begin == 0
    assert first_ner.end == 4

    second_ner = actual_span_annotations[1]
    assert second_ner.features[value_feature] == "CAP"
    assert second_ner.begin == 10
    assert second_ner.end == 16

    third_ner = actual_span_annotations[2]
    assert third_ner.features[value_feature] == "TM"
    assert third_ner.begin == 18
    assert third_ner.end == 29
