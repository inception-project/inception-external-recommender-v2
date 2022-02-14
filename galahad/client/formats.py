import copy
from dataclasses import dataclass
from typing import List

from galahad.server.annotations import Annotations
from galahad.server.classifier import AnnotationFeatures, AnnotationTypes
from galahad.server.dataclasses import Annotation, Document


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
    end = 0
    for sentence_idx, sentence in enumerate(sentences):
        sentence_start = begin
        for token_idx, token_text in enumerate(sentence):
            end = begin + len(token_text)
            token_idx_to_begins[(sentence_idx, token_idx)] = begin
            token_idx_to_ends[(sentence_idx, token_idx)] = end

            annotation = annotations.create_annotation(token_type, begin, end)
            assert annotations.get_covered_text(annotation) == token_text

            begin = end + 1
        annotations.create_annotation(AnnotationTypes.SENTENCE.value, sentence_start, end)

    for sentence_idx, sentence in enumerate(spans or []):
        for span in sentence:
            begin = token_idx_to_begins[(sentence_idx, span.begin)]
            end = token_idx_to_ends[(sentence_idx, span.end - 1)]

            annotations.create_annotation(span_annotation_type, begin, end, {value_feature: span.value})

    document = Document(text=text, annotations=annotations.to_dict(), version=version)
    return document


def build_span_classification_response(original_doc: Document, spans: List[Span] = None, version: int = 0) -> Document:
    annotated_doc = copy.deepcopy(original_doc)
    annotated_doc.version = version

    assert AnnotationTypes.TOKEN.value in original_doc.annotations
    assert AnnotationTypes.SENTENCE.value in original_doc.annotations

    annotations = Annotations.from_dict(annotated_doc.text, annotated_doc.annotations)

    sentences = annotations.select(AnnotationTypes.SENTENCE.value)

    dummy_text_annotation = {"begin": sentences[0].begin, "end": sentences[-1].end}
    all_tokens = annotations.select_covered(AnnotationTypes.TOKEN.value, Annotation(**dummy_text_annotation))
    for span in spans:
        first_token = all_tokens[span.begin]
        last_token = all_tokens[span.end - 1]

        annotations.create_annotation(
            AnnotationTypes.ANNOTATION.value,
            first_token.begin,
            last_token.end,
            {AnnotationFeatures.VALUE.value: span.value},
        )

    annotated_doc.annotations = annotations.get_annotations()
    return annotated_doc


def build_doc_from_tokens_and_text(text: str, sentences: List[List[str]]) -> Document:
    sentence_list = []
    token_list = []
    position = 0
    subtext = text
    sentence_start = 0
    for sentence in sentences:

        for token in sentence:
            start = subtext.find(token)
            position = position + start
            token_list.append(Annotation(**{"begin": position, "end": position + len(token), "features": {}}))
            subtext = subtext[start + len(token) :]
            position = position + len(token)

        sentence_list.append(Annotation(**{"begin": sentence_start, "end": position, "features": {}}))
        sentence_start = position + 1

    doc = Document(text=text, annotations={"t.token": token_list, "t.sentence": sentence_list}, version=0)
    return doc


def build_token_labeling_response(original_doc: Document, labels: List[str] = None, version: int = 0) -> Document:
    annotated_doc = copy.deepcopy(original_doc)
    annotated_doc.version = version

    assert AnnotationTypes.TOKEN.value in original_doc.annotations
    assert AnnotationTypes.SENTENCE.value in original_doc.annotations
    assert len(original_doc.annotations["t.token"]) == len(labels)

    annotations = Annotations.from_dict(annotated_doc.text, annotated_doc.annotations)

    for token, label in zip(original_doc.annotations["t.token"], labels):
        annotations.create_annotation(
            AnnotationTypes.ANNOTATION.value,
            token.begin,
            token.end,
            {AnnotationFeatures.VALUE.value: label},
        )

    annotated_doc.annotations = annotations.get_annotations()
    return annotated_doc


def build_span_classification_response_per_sentence(
    original_doc: Document, spans: List[List[Span]] = None, version: int = 0
) -> Document:
    annotated_doc = copy.deepcopy(original_doc)
    annotated_doc.version = version

    assert AnnotationTypes.TOKEN.value in original_doc.annotations
    assert AnnotationTypes.SENTENCE.value in original_doc.annotations

    annotations = Annotations.from_dict(annotated_doc.text, annotated_doc.annotations)

    sentences = annotations.select(AnnotationTypes.SENTENCE.value)
    assert len(sentences) == len(spans)

    for sentence, cur_spans in zip(sentences, spans):
        tokens = annotations.select_covered(AnnotationTypes.TOKEN.value, sentence)

        for span in cur_spans:
            first_token = tokens[span.begin]
            last_token = tokens[span.end - 1]

            annotations.create_annotation(
                AnnotationTypes.ANNOTATION.value,
                first_token.begin,
                last_token.end,
                {AnnotationFeatures.VALUE.value: span.value},
            )

    annotated_doc.annotations = annotations.get_annotations()
    return annotated_doc
