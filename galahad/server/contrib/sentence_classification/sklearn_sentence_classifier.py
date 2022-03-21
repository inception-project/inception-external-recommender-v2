import logging
from typing import List, Optional

try:
    from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import Pipeline
except ImportError as error:
    print("Could not import 'sklearn', please install it manually via 'pip install scikit-learn'")

from galahad.client.formats import build_sentence_classification_document
from galahad.server.annotations import Annotations
from galahad.server.classifier import AnnotationFeatures, AnnotationTypes, Classifier
from galahad.server.dataclasses import Document

logger = logging.getLogger(__name__)


class SklearnSentenceClassifier(Classifier):
    def __init__(self):
        super().__init__()

        self._sentence_type = AnnotationTypes.SENTENCE.value
        self._sentence_annotation_type = AnnotationTypes.ANNOTATION.value
        self._target_feature = AnnotationFeatures.VALUE.value

    def train(self, model_id: str, documents: List[Document]):
        texts = []
        labels = []

        for document in documents:
            annotations = Annotations.from_dict(document.text, document.annotations)

            for sentence in annotations.select(self._sentence_type):
                for sentence_label in annotations.select_covered(self._sentence_annotation_type, sentence):
                    text = annotations.get_covered_text(sentence_label)
                    label = sentence_label.features.get(self._target_feature)

                    if label is None:
                        continue

                    texts.append(text)
                    labels.append(label)

        assert len(texts) == len(labels), "Unequal number of sentences and labels"
        if not len(texts):
            logger.debug(f"Empty training set, skipping!")

        model = Pipeline([("vect", CountVectorizer()), ("tfidf", TfidfTransformer()), ("clf", MultinomialNB())])
        model.fit(texts, labels)

        logger.debug(f"Training finished for model with id [%s]", model_id)

        self._save_model(model_id, model)

    def predict(self, model_id: str, document: Document) -> Optional[Document]:
        model: Optional[Pipeline] = self._load_model(model_id)

        if model is None:
            logger.debug("No trained model ready yet!")
            return

        annotations = Annotations.from_dict(document.text, document.annotations)
        texts = [annotations.get_covered_text(sentence) for sentence in annotations.select(self._sentence_type)]
        predicted_labels = model.predict(texts)

        return build_sentence_classification_document(texts, predicted_labels)

    def consumes(self) -> List[str]:
        return [self._sentence_type, self._sentence_annotation_type]

    def produces(self) -> List[str]:
        return [self._target_feature]
