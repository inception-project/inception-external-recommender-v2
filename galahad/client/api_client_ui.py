import gradio as gr
import api_client
import formats
from galahad.server import GalahadServer
from galahad.server.contrib.ner.spacy_ner import SpacyNerClassifier

models = ["SpacyNER"]
client = api_client.GalahadClient("http://127.0.0.1:8000")


def predict(model, sentence):

    tokens = sentence.split(" ")
    doc = formats.build_span_classification_request([tokens])
    print(client.list_all_classifiers())
    annotated_doc = client.predict_on_document("SpacyNER", "en_core_web_sm", doc)
    ret = ""
    for annotation in annotated_doc["annotations"]["t.annotation"]:
        ret = ret + annotated_doc["text"][annotation["begin"]:annotation["end"]] + ": " + annotation["features"]["f.value"] + "; "

    return ret.removesuffix("; ")


iface = gr.Interface(fn=predict,
                     inputs=[gr.inputs.Dropdown(models, type="value", default=None, label="Choose a model"),
                             gr.inputs.Textbox(placeholder="Type sentence here...", label="Input Sentence")],
                     outputs="text",
                     examples=[["SpacyNER", "Neil Armstrong was a NASA astronaut"],
                               ["SpacyNER", "The jazz musician Louis Armstrong was born in New Orleans"],
                               ["SpacyNER", "Lance Armstrong won the Tour de France several times"]],
                     title="Named Entity Recognition",
                     description="Named Entity Recognition (NER) is the task of identifying persons, places, "
                                 "institutions etc. in a given sentence.")

iface.launch()
