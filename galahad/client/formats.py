from typing import List

from galahad.server.annotations import Annotations
from galahad.server.classifier import AnnotationFeatures, AnnotationTypes
from galahad.server.dataclasses import Document


def build_sentence_classification_document(sentences: List[str], labels: List[str], version: int = 0) -> Document:
    assert len(sentences) == len(labels), "Sentences and labels need to have the same length!"

    text = " ".join(sentences)
    annotations = Annotations(text)

    sentence_type = AnnotationTypes.SENTENCE.value
    sentence_annotation_type = AnnotationTypes.SENTENCE_ANNOTATION.value
    value_feature = AnnotationFeatures.VALUE.value

    begin = 0
    for sentence, label in zip(sentences, labels):
        end = begin + len(sentence)

        annotation = annotations.create_annotation(sentence_type, begin, end)
        assert annotations.get_covered_text(annotation) == sentence

        annotation = annotations.create_annotation(sentence_annotation_type, begin, end, {value_feature: label})
        assert annotations.get_covered_text(annotation) == sentence

        begin = end + 1

    document = Document(text=text, annotations=annotations.to_dict(), version=version)
    return document


def build_sequence_tagging_request():
    pass
