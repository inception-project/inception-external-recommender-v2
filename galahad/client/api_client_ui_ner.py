import api_client
import gradio as gr

from galahad.client.gradio_utils import annotation_to_gradio, input_to_doc

client = api_client.GalahadClient("http://127.0.0.1:8000")
models = [model.name for model in client.list_all_classifiers()]


# same function as in api_client_ui_pos


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

# Before you start this program make sure to run
# uvicorn main:ner_server
# in the terminal
if __name__ == "__main__":
    iface.launch()
