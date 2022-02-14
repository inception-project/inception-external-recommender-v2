import api_client
import formats
import gradio as gr
from nltk.tokenize import word_tokenize

client = api_client.GalahadClient("http://127.0.0.1:8000")
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
        ret.append((annotated_doc["text"][annotation["begin"]: annotation["end"]], annotation["features"]["f.value"]))
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

iface.launch()
