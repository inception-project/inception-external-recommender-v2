from galahad.client.formats import (Span,
                                    build_sentence_classification_document,
                                    build_span_classification_document)
from galahad.server.classifier import AnnotationFeatures, AnnotationTypes


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
    actual_sentence_annotations = result.annotations[AnnotationTypes.SENTENCE_ANNOTATION.value]
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

    result = build_span_classification_document(tokens, spans)
    value_feature = AnnotationFeatures.VALUE.value
    actual_span_annotations = result.annotations[AnnotationTypes.SPAN_ANNOTATION.value]

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
