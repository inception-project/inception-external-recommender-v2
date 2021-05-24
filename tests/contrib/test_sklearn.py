from datasets import load_dataset

from galahad.client import build_sentence_classification_request

dataset = load_dataset("sms_spam")


def test_training():
    request = build_sentence_classification_request()
