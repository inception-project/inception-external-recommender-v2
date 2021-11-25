from dataclasses import dataclass
from typing import List

from galahad.server.annotations import Annotations
from galahad.server.classifier import AnnotationFeatures, AnnotationTypes
from galahad.server.dataclasses import Document



@dataclass
class Span:
    begin: int
    end: int
    value: str


def build_sentence_classification_document(sentences: List[str], labels: List[str], version: int = 0) -> Document:
    assert len(sentences) == len(labels), "Sentences and labels need to have the same length!"

    text = " ".join(sentences)
    annotations = Annotations(text)

    sentence_type = AnnotationTypes.SENTENCE.value
    sentence_annotation_type = AnnotationTypes.ANNOTATION.value
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


def build_span_classification_request(
    sentences: List[List[str]], spans: List[List[Span]] = None, version: int = 0
) -> Document:
    text = " ".join(t for sentence in sentences for t in sentence)
    annotations = Annotations(text)

    token_idx_to_begins = {}
    token_idx_to_ends = {}

    token_type = AnnotationTypes.TOKEN.value
    span_annotation_type = AnnotationTypes.ANNOTATION.value
    value_feature = AnnotationFeatures.VALUE.value

    begin = 0
    for sentence_idx, sentence in enumerate(sentences):
        for token_idx, token_text in enumerate(sentence):
            end = begin + len(token_text)
            token_idx_to_begins[(sentence_idx, token_idx)] = begin
            token_idx_to_ends[(sentence_idx, token_idx)] = end

            annotation = annotations.create_annotation(token_type, begin, end)
            assert annotations.get_covered_text(annotation) == token_text

            begin = end + 1

    for sentence_idx, sentence in enumerate(spans or []):
        for span in sentence:
            begin = token_idx_to_begins[(sentence_idx, span.begin)]
            end = token_idx_to_ends[(sentence_idx, span.end - 1)]

            annotations.create_annotation(span_annotation_type, begin, end, {value_feature: span.value})

    document = Document(text=text, annotations=annotations.to_dict(), version=version)
    return document

#original_doc: Document
def build_span_classification_response(text: str, sentences: List[List[str]], spans: List[List[Span]] = None, version: int = 0) -> Document:
    #text = original_doc.text
    annotations = Annotations(text)

    token_type = AnnotationTypes.TOKEN.value
    span_annotation_type = AnnotationTypes.ANNOTATION.value
    value_feature = AnnotationFeatures.VALUE.value

    current = 0
    start_lists = []
    end_lists = []
    for sentence in sentences:
        starts = []
        ends = []
        for token in sentence:
            token_pos = text.find(token, current)
            starts.append(token_pos)
            current = token_pos + len(token)
            ends.append(current)
            annotations.create_annotation(token_type, token_pos, current)
        start_lists.append(starts)
        end_lists.append(ends)

    for i in range(len(spans)):
        current_starts = start_lists[i]
        current_ends = end_lists[i]
        for span in spans[i]:
            annotations.create_annotation(span_annotation_type, current_starts[span.begin], current_ends[span.end-1], {value_feature: span.value})


    document = Document(text=text, annotations=annotations.to_dict(), version=version)

    return document


