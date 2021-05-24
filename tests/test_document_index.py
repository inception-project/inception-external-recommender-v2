import pytest

from galahad.server.annotations import Annotations
from galahad.server.classifier import AnnotationTypes
from galahad.server.dataclasses import Document


@pytest.fixture
def document() -> Document:
    return Document.parse_obj(Document.Config.schema_extra["example"])


def test_get_covered_text(document: Document):
    annotations = Annotations.from_dict(document.text, document.annotations)

    sentences = annotations.select(AnnotationTypes.SENTENCE.value)
    first_sentence, second_sentence = sentences
    sentence_text1 = "Joe waited for the train ."
    sentence_text_2 = "The train was late ."

    assert annotations.get_covered_text(first_sentence) == sentence_text1
    assert annotations.get_covered_text(second_sentence) == sentence_text_2

    tokens = annotations.select(AnnotationTypes.TOKEN.value)
    token_texts = annotations.text.split(" ")
    assert len(token_texts) == len(tokens)

    for token, expected_token_text in zip(tokens, token_texts):
        assert annotations.get_covered_text(token) == expected_token_text


def test_select(document: Document):
    annotations = Annotations.from_dict(document.text, document.annotations)

    assert len(annotations.select(AnnotationTypes.SENTENCE.value)) == 2
    assert len(annotations.select(AnnotationTypes.TOKEN.value)) == 11


def test_select_covered(document: Document):
    annotations = Annotations.from_dict(document.text, document.annotations)

    token_type = AnnotationTypes.TOKEN.value
    sentence_type = AnnotationTypes.SENTENCE.value

    tokens = annotations.select(token_type)
    sentences = annotations.select(sentence_type)

    first_sentence, second_sentence = sentences

    tokens_in_first_sentence = tokens[:6]
    tokens_in_second_sentence = tokens[6:]

    actual_tokens_in_first_sentence = annotations.select_covered(token_type, first_sentence)
    actual_tokens_in_second_sentence = annotations.select_covered(token_type, second_sentence)

    assert actual_tokens_in_first_sentence == tokens_in_first_sentence
    assert actual_tokens_in_second_sentence == tokens_in_second_sentence
