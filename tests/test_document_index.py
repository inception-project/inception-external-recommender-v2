from typing import Dict, List

import pytest

from galahad.classifier import AnnotationTypes
from galahad.dataclasses import Document, Layer, Layers
from galahad.document_index import DocumentIndex


@pytest.fixture
def annotations() -> Layers:
    document: Document = Document.parse_obj(Document.Config.schema_extra["example"])
    return document.annotations


def test_select_covered(annotations: Layers):
    token_type = AnnotationTypes.TOKEN.value
    sentence_type = AnnotationTypes.SENTENCE.value

    tokens = annotations[token_type]
    sentences = annotations[sentence_type]

    first_sentence, second_sentence = sentences

    tokens_in_first_sentence = tokens[:6]
    tokens_in_second_sentence = tokens[6:]

    index = DocumentIndex(annotations)

    actual_tokens_in_first_sentence = index.select_covered(token_type, first_sentence)
    actual_tokens_in_second_sentence = index.select_covered(token_type, second_sentence)

    assert actual_tokens_in_first_sentence == tokens_in_first_sentence
    assert actual_tokens_in_second_sentence == tokens_in_second_sentence
