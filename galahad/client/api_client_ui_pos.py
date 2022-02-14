import api_client
import gradio as gr

from galahad.client.gradio_utils import annotation_to_gradio, input_to_doc

client = api_client.GalahadClient("http://127.0.0.1:8000")
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

# Before you start this program make sure to run
# uvicorn main:pos_server
# in the terminal
if __name__ == "__main__":
    iface.launch()
