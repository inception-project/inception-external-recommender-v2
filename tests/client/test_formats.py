from galahad.client.formats import build_sentence_classification_document
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
