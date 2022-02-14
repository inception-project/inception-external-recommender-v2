import api_client
import formats
import gradio as gr
from nltk.tokenize import word_tokenize

client = api_client.GalahadClient("http://127.0.0.1:8000")
peter = client.list_all_classifiers()
models = [model.name for model in client.list_all_classifiers()]


def predict(model, sentence):
    tokens = word_tokenize(sentence)
    # The following method works (cf. the test in test_formats). The problem is that HighlightedText uses " ".join so
    # the displayed result is wrong, but the internal calculations are all correct.
    doc = formats.build_doc_from_tokens_and_text(sentence, [tokens])
    annotated_doc = client.predict_on_document(model, "PLACEHOLDER", doc)
    ret = []
    begin = 0
    for annotation in annotated_doc["annotations"]["t.annotation"]:
        end = annotation["begin"] - 1
        if end >= begin:
            ret.append((annotated_doc["text"][begin:end], None))
        begin = annotation["end"]
        ret.append((annotated_doc["text"][annotation["begin"] : annotation["end"]], annotation["features"]["f.value"]))
    ret.append((annotated_doc["text"][begin:], None))
    return ret


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Dropdown(models, type="value", default=None, label="Choose a model"),
        gr.inputs.Textbox(placeholder="Type sentence here...", label="Input Sentence"),
    ],
    outputs=gr.outputs.HighlightedText(),
    examples=[
        ["SpacyNER", "Neil Armstrong was a NASA astronaut"],
        ["SpacyNER", "The jazz musician Louis Armstrong was born in New Orleans"],
        ["SpacyNER", "Lance Armstrong won the Tour de France several times"],
    ],
    title="Named Entity Recognition",
    description="Named Entity Recognition (NER) is the task of identifying persons, places, "
    "institutions etc. in a given sentence.",
    allow_screenshot=False,
    allow_flagging="never",
)

iface.launch()
