import api_client
import formats
import gradio as gr

client = api_client.GalahadClient("http://127.0.0.1:8000")
peter = client.list_all_classifiers()
models = [model.name for model in client.list_all_classifiers()]


def predict(model, sentence):
    return ""


iface = gr.Interface(
    fn=predict,
    inputs=[
        gr.inputs.Dropdown(models, type="value", default=None, label="Choose a model"),
        gr.inputs.Textbox(placeholder="Type sentence here...", label="Input Sentence"),
    ],
    # outputs=gr.outputs.HighlightedText(),
    outputs="text",
    examples=[
        ["SpacyNER", "Neil Armstrong was a NASA astronaut"],
        ["SpacyNER", "The jazz musician Louis Armstrong was born in New Orleans"],
        ["SpacyNER", "Lance Armstrong won the Tour de France several times"],
    ],
    title="Sentence Classification",
    description="Sentence Classification is the task of",
    allow_screenshot=False,
    allow_flagging="never",
)

iface.launch()
