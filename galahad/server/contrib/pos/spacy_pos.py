from typing import Optional

try:
    import spacy as spacy
    from spacy.tokens import Doc
except ImportError as error:
    print("Could not import 'spacy', please install it manually via 'pip install spacy'")

from galahad.formats import build_token_labeling_response
from galahad.server.annotations import Annotations
from galahad.server.classifier import (AnnotationFeatures, AnnotationTypes,
                                       Classifier)
from galahad.server.dataclasses import Document


class SpacyPosTagger(Classifier):
    def __init__(self, model_name: str):
        super().__init__()

        self._token_type = AnnotationTypes.TOKEN.value
        self._target_feature = AnnotationFeatures.VALUE.value

        self._model = spacy.load(model_name, disable=["parser"])

    def predict(self, model_id: str, document: Document) -> Optional[Document]:
        # Extract the tokens from the document and create a spacy doc from it
        annotations = Annotations.from_dict(document.text, document.annotations)
        words = [annotations.get_covered_text(token) for token in annotations.select(self._token_type)]

        spacy_doc = Doc(self._model.vocab, words=words)

        self._model.get_pipe("tok2vec")(spacy_doc)
        self._model.get_pipe("tagger")(spacy_doc)

        list_of_pos_tags = []
        for i in range(len(spacy_doc)):
            list_of_pos_tags.append(spacy_doc[i].tag_)

        return build_token_labeling_response(document, list_of_pos_tags)
