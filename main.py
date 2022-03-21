import uvicorn

from galahad.server import GalahadServer
from galahad.server.contrib.ner.spacy_ner import SpacyNerTagger
from galahad.server.contrib.pos.spacy_pos import SpacyPosTagger

server = GalahadServer()
server.add_classifier("SpacyPOS", SpacyPosTagger("en_core_web_sm"))
server.add_classifier("SpacyNER", SpacyNerTagger("en_core_web_sm"))


if __name__ == "__main__":
    uvicorn.run(server, host="127.0.0.1", port=8000)
