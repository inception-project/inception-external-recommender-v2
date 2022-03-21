import argparse
from typing import List, Tuple

import gradio as gr
import nltk
from nltk import word_tokenize

from galahad import formats
from galahad.client import GalahadClient
from galahad.server.dataclasses import Document


def input_to_doc(sentence: str) -> Document:
    tokens = word_tokenize(sentence)
    return formats.build_doc_from_tokens_and_text(sentence, [tokens])


# Convenience function which represents the annotations of a document as a list of tuples
# each consisting of the covered text with the annotation feature. Gradio's highlightText demands this format.
def annotation_to_gradio(annotated_doc: Document) -> List[Tuple[str, str]]:
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


def pos_demo() -> gr.Interface:
    client = GalahadClient("http://127.0.0.1:8000")
    models = [model.name for model in client.list_all_classifiers()]

    def predict(model, sentence):
        annotated_doc = client.predict_on_document(model, "PLACEHOLDER", input_to_doc(sentence))
        return annotation_to_gradio(annotated_doc)

    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.inputs.Dropdown(models, type="value", default=None, label="Choose a model"),
            gr.inputs.Textbox(placeholder="Type sentence here...", label="Input Sentence"),
        ],
        outputs=gr.outputs.HighlightedText(),
        examples=[
            [models[0], "I love chocolate"],
            [models[0], "Peter received such a beautifully crafted gift"],
            [models[0], "Strawberries are sweet and red fruits"],
        ],
        title="Part-of-Speech Tagging",
        description="Part-of-Speech Tagging (POS-tagging) is the task of assigning each word in a sentence "
        "its part-of-speech tag, e.g. noun, verb or adjective.",
        allow_screenshot=False,
        allow_flagging="never",
    )

    return iface


def ner_demo() -> gr.Interface:
    client = GalahadClient("http://127.0.0.1:8000")
    models = [model.name for model in client.list_all_classifiers()]

    def predict(model, sentence):
        # The following method works. The problem is that HighlightedText uses " ".join to
        # concatenate text units with not-None and None features.
        # The displayed result is wrong, but the internal calculations are all correct.
        # Example: "Joe Biden, a child of Delaware." -> "Joe Biden , a child of Delaware ." and "Joe Biden" and "Delaware"
        # are marked as named entities.
        annotated_doc = client.predict_on_document(model, "PLACEHOLDER", input_to_doc(sentence))
        return annotation_to_gradio(annotated_doc)

    iface = gr.Interface(
        fn=predict,
        inputs=[
            gr.inputs.Dropdown(models, type="value", default=None, label="Choose a model"),
            gr.inputs.Textbox(placeholder="Type sentence here...", label="Input Sentence"),
        ],
        outputs=gr.outputs.HighlightedText(),
        examples=[
            [models[0], "Neil Armstrong was a NASA astronaut"],
            [models[0], "The jazz musician Louis Armstrong was born in New Orleans"],
            [models[0], "Lance Armstrong won the Tour de France several times"],
        ],
        title="Named Entity Recognition",
        description="Named Entity Recognition (NER) is the task of identifying persons, places, "
        "institutions etc. in a given sentence.",
        allow_screenshot=False,
        allow_flagging="never",
    )

    return iface


# Before you start this program make sure to run
# uvicorn main:pos_server
# in the terminal
if __name__ == "__main__":
    nltk.download("punkt")

    parser = argparse.ArgumentParser(description="Run the Galahad Gradio demo.")
    parser.add_argument("task", help="Task to run the demo for (pos|ner)")

    args = parser.parse_args()

    task = args.task
    if task == "pos":
        iface = pos_demo()
    elif task == "ner":
        iface = ner_demo()
    else:
        raise Exception("Unknown task: " + task)

    iface.launch()
