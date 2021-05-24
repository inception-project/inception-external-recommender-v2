import logging
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

from galahad.server.annotations import Annotations
from galahad.server.classifier import (AnnotationFeatures, AnnotationTypes,
                                       Classifier, Remapper)
from galahad.server.dataclasses import Document

logger = logging.getLogger(__name__)


class SklearnSentenceClassifier(Classifier):
    def __init__(self, model_directory: Path):
        super().__init__(model_directory)

        self._sentence_type = AnnotationTypes.SENTENCE.value
        self._sentence_annotation_type = AnnotationTypes.SENTENCE_ANNOTATION.value
        self._target_feature = AnnotationFeatures.VALUE.value

    def train(self, model_id: str, documents: List[Document], remapper: Remapper):
        sentence_type = remapper.remap(self._sentence_type)
        sentence_annotation_type = remapper.remap(self._sentence_annotation_type)
        target_feature = remapper.remap(self._target_feature)

        texts = []
        labels = []

        for document in documents:
            index = Annotations(document.text, document.annotations)

            for sentence in index.select(sentence_type):
                for sentence_label in index.select_covered(sentence_annotation_type, sentence):
                    text = index.get_covered_text(sentence_label)
                    label = sentence_label.features.get(target_feature)

                    if label is None:
                        continue

                    texts.append(text)
                    labels.append(label)

        model = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf", MultinomialNB())])
        model.fit(texts, labels)

        logger.debug(f"Training finished for model with id [%s]", model_id)

        self._save_model(model_id, model)

    def predict(self, model_id: str, remapper: Remapper):
        pass

    def consumes(self) -> List[str]:
        return [self._sentence_type, self._sentence_annotation_type]

    def produces(self) -> List[str]:
        return [self._target_feature]
