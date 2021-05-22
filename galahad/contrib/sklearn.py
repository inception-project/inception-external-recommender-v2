from typing import Dict, List

from galahad.classifier import AnnotationTypes, Classifier, Remapper
from galahad.dataclasses import Document


class SklearnSentenceClassifier(Classifier):
    def __init__(self):
        self._sentence_type = AnnotationTypes.SENTENCE.value
        self._sentence_label_type = AnnotationTypes.SENTENCE_LABEL.value

    def train(self, documents: List[Document], remapper: Remapper):
        sentence_type = remapper.remap(self._sentence_type)
        sentence_label_type = remapper.remap(self._sentence_label_type)

        sentence_texts = []
        for document in documents:
            for sentence in document.annotations.get(sentence_type):
                begin = sentence["begin"]
                end = sentence["end"]
                covered_text = document.text[begin:end]
                sentence_texts.append(covered_text)

    def predict(self, remapper: Remapper):
        pass

    def consumes(self) -> List[str]:
        return [self._sentence_type, self._sentence_label]

    def produces(self) -> List[str]:
        return [self._sentence_label]
