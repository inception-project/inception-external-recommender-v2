from typing import Optional

import spacy as spacy
from spacy.tokens import Doc

from galahad.client.formats import Span, build_span_classification_document
from galahad.server.annotations import Annotations
from galahad.server.classifier import (AnnotationFeatures, AnnotationTypes,
                                       Classifier)
from galahad.server.dataclasses import Document


class SpacyNerClassifier(Classifier):
    def __init__(self, model_name: str):
        super().__init__()

        self._token_type = AnnotationTypes.TOKEN.value
        self._target_feature = AnnotationFeatures.VALUE.value

        self._model = spacy.load(model_name, disable=["parser"])

    def predict(self, model_id: str, document: Document) -> Optional[Document]:
        # Extract the tokens from the document and create a spacy doc from it
        annotations = Annotations.from_dict(document.text, document.annotations)
        words = [annotations.get_covered_text(token) for token in annotations.select(self._token_type)]

        doc = Doc(self._model.vocab, words=words)

        # Find the named entities
        self._model.get_pipe("ner")(doc)

        # For every entity returned by spacy, create an annotation in the resulting doc
        spans = []
        for named_entity in doc.ents:
            spans.append(Span(named_entity.start, named_entity.end, named_entity.label_))

        return build_span_classification_document([words], [spans])
