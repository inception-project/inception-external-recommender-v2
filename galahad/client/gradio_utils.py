from typing import List, Tuple

from nltk import word_tokenize

from galahad.client import formats
from galahad.server.dataclasses import Document


def input_to_doc(sentence: str) -> Document:
    tokens = word_tokenize(sentence)
    return formats.build_doc_from_tokens_and_text(sentence, [tokens])


# Convenience function which represents the annotations of a document as a list of tuples
# each consisting of the covered text with the annotation feature. Gradio's highlightText demands this format.
def annotation_to_gradio(annotated_doc: Document) -> List[Tuple[str]]:
    text_highlights = []
    begin_not_highlighted = 0
    for annotation in annotated_doc["annotations"]["t.annotation"]:
        end_not_highlighted = annotation["begin"] - 1
        if end_not_highlighted >= begin_not_highlighted:
            text_highlights.append((annotated_doc["text"][begin_not_highlighted:end_not_highlighted], None))
        begin_not_highlighted = annotation["end"]
        text_highlights.append(
            (annotated_doc["text"][annotation["begin"] : annotation["end"]], annotation["features"]["f.value"])
        )

    text_highlights.append((annotated_doc["text"][begin_not_highlighted:], None))

    return text_highlights
